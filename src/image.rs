use crate::{bmp, Error};
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
    /// Averages each k x k block into one pixel (supersampling).
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
}
