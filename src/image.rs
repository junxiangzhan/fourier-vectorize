use crate::{bmp, Error, Point2};
use std::{fs::File, io::BufWriter, path::Path};

/// Three floating RGB channels. Quantization is deferred to image output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}
impl Color {
    pub const fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }
    /// Blend toward `other` by `amount` in [0, 1]; used to turn coverage c
    /// into a color: background * (1 - c) + foreground * c.
    pub fn mix(self, other: Self, amount: f64) -> Self {
        Self::rgb(
            self.r * (1.0 - amount) + other.r * amount,
            self.g * (1.0 - amount) + other.g * amount,
            self.b * (1.0 - amount) + other.b * amount,
        )
    }
    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite()
    }
}

/// General, dynamically sized, row-major image. Pixel semantics belong to T.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    width: usize,
    height: usize,
    pixels: Vec<T>,
}

impl<T> Image<T> {
    /// A width x height image with every pixel set to `value`.
    pub fn new(width: usize, height: usize, value: T) -> Result<Self, Error>
    where
        T: Clone,
    {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions);
        }
        // Reject sizes whose byte count would overflow before allocating.
        let len = width.checked_mul(height).ok_or(Error::SizeOverflow)?;
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::SizeOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(Error::SizeOverflow);
        }
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(len)
            .map_err(|_| Error::Allocation)?;
        pixels.resize(len, value);
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn pixels(&self) -> &[T] {
        &self.pixels
    }
    /// Pixel (x, y) is stored at index y * width + x (row-major).
    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x >= self.width || y >= self.height {
            None
        } else {
            self.pixels.get(y * self.width + x)
        }
    }
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        if x >= self.width || y >= self.height {
            None
        } else {
            self.pixels.get_mut(y * self.width + x)
        }
    }
    pub fn row(&self, y: usize) -> Option<&[T]> {
        if y >= self.height {
            None
        } else {
            Some(&self.pixels[y * self.width..(y + 1) * self.width])
        }
    }
    /// Mutable slice of one row; the scanline fill writes whole spans here.
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [T]> {
        if y >= self.height {
            None
        } else {
            Some(&mut self.pixels[y * self.width..(y + 1) * self.width])
        }
    }
    /// A new image of the same size with `transform` applied to each pixel.
    pub fn map<U>(&self, mut transform: impl FnMut(&T) -> U) -> Result<Image<U>, Error> {
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(self.pixels.len())
            .map_err(|_| Error::Allocation)?;
        pixels.extend(self.pixels.iter().map(&mut transform));
        Ok(Image {
            width: self.width,
            height: self.height,
            pixels,
        })
    }
}

impl Image<Color> {
    pub fn save_bmp(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        bmp::validate(self)?;
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bmp::write(self, &mut writer)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    }
}

impl Image<f64> {
    /// Averages each k x k block into one pixel (Part 3 supersampling).
    /// Applied to a 0/1 fill, the mean is the covered fraction of the
    /// block's k*k sub-positions. As a filter, this is a box low-pass filter
    /// applied before the image is decimated back to the output grid.
    pub fn mean_downsample(&self, k: usize) -> Result<Self, Error> {
        if k == 0 || !self.width.is_multiple_of(k) || !self.height.is_multiple_of(k) {
            return Err(Error::InvalidFactor);
        }
        if self.pixels.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFinite);
        }
        let divisor = k.checked_mul(k).ok_or(Error::SizeOverflow)? as f64;
        let mut result = Self::new(self.width / k, self.height / k, 0.0)?;
        for y in 0..result.height {
            for x in 0..result.width {
                let mut mean = 0.0;
                // Output pixel (x, y) covers source rows y*k..(y+1)*k and
                // source columns x*k..(x+1)*k.
                for sy in y * k..(y + 1) * k {
                    for value in &self.row(sy).unwrap()[x * k..(x + 1) * k] {
                        mean += value / divisor;
                    }
                }
                if !mean.is_finite() {
                    return Err(Error::NonFinite);
                }
                *result.get_mut(x, y).unwrap() = mean;
            }
        }
        Ok(result)
    }

    /// Part 4: rotates the bitmap about `center` by `angle` (same sense as
    /// `Point2::rotated_about`) and resamples it onto the same grid.
    ///
    /// Inverse mapping: each output pixel center is rotated back by -angle
    /// to find where it came from in the source, and the source is read
    /// there by bilinear interpolation. Source positions outside the image
    /// read as 0 (empty background).
    pub fn rotated_bilinear(&self, center: Point2, angle: f64) -> Result<Self, Error> {
        if !center.is_finite() || !angle.is_finite() {
            return Err(Error::NonFinite);
        }
        if self.pixels.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFinite);
        }
        let mut result = Self::new(self.width, self.height, 0.0)?;
        for y in 0..self.height {
            for x in 0..self.width {
                let source =
                    Point2::new(x as f64 + 0.5, y as f64 + 0.5).rotated_about(center, -angle);
                // Shift by half a pixel so integer (u, v) are pixel centers.
                *result.get_mut(x, y).unwrap() =
                    self.sample_bilinear(source.x() - 0.5, source.y() - 0.5);
            }
        }
        Ok(result)
    }

    /// Bilinear interpolation between the four pixel centers around (u, v),
    /// where pixel (i, j) has its center at u = i, v = j.
    fn sample_bilinear(&self, u: f64, v: f64) -> f64 {
        let x0 = u.floor();
        let y0 = v.floor();
        let fx = u - x0;
        let fy = v - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        // Weighted mean along x on the two rows, then along y between them.
        let top = self.value_or_zero(x0, y0) * (1.0 - fx) + self.value_or_zero(x0 + 1, y0) * fx;
        let bottom =
            self.value_or_zero(x0, y0 + 1) * (1.0 - fx) + self.value_or_zero(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn value_or_zero(&self, x: i64, y: i64) -> f64 {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            0.0
        } else {
            self.pixels[y as usize * self.width + x as usize]
        }
    }
}
