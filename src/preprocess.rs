//! Getting a picture ready: shrinking, edge-preserving smoothing, and cutting
//! a channel into levels.
use crate::{color::lab, Color, Error, Image};

const KMEANS_ROUNDS: usize = 30;
/// Bilateral smoothing: radius (px), spatial sigma (px), colour sigma (ΔE).
const BILATERAL: (usize, f64, f64) = (2, 2.0, 8.0);

/// Centre crop to a multiple of `factor`, then average each factor x factor
/// block (a box filter, per colour channel).
pub(crate) fn shrink(image: &Image<Color>, factor: usize) -> Result<Image<Color>, Error> {
    let (w, h) = (image.width() / factor, image.height() / factor);
    let (x0, y0) = (
        (image.width() - w * factor) / 2,
        (image.height() - h * factor) / 2,
    );
    let mut out = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
    let n = (factor * factor) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0; 3];
            for sy in 0..factor {
                for sx in 0..factor {
                    let c = image
                        .get(x0 + x * factor + sx, y0 + y * factor + sy)
                        .unwrap();
                    sum[0] += c.r;
                    sum[1] += c.g;
                    sum[2] += c.b;
                }
            }
            *out.get_mut(x, y).unwrap() = Color::rgb(sum[0] / n, sum[1] / n, sum[2] / n);
        }
    }
    Ok(out)
}

/// `factor`, or the smallest one that brings the longer side of a
/// `width` x `height` picture to at most `longest` pixels.
pub(crate) fn shrink_factor(width: usize, height: usize, longest: usize, factor: Option<usize>) -> usize {
    factor.unwrap_or_else(|| width.max(height).div_ceil(longest).max(1))
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Edge-preserving smoothing: each pixel becomes a mean of its neighbours
/// within BILATERAL.0 px, weighted by distance (sigma BILATERAL.1 px) and by
/// colour difference (sigma BILATERAL.2 in ΔE). Fine cracks and speckle
/// fade; edges between clearly different colours stay sharp.
pub(crate) fn bilateral(input: &Image<Color>) -> Result<Image<Color>, Error> {
    let (w, h) = (input.width(), input.height());
    let (radius, sigma_s, sigma_r) = BILATERAL;
    let labs: Vec<[f64; 3]> = input.pixels().iter().map(|c| lab(*c)).collect();
    let pixels = input.pixels();
    let r = radius as i64;
    let mut out = Image::new(w, h, Color::rgb(0.0, 0.0, 0.0))?;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (mut sum, mut total) = ([0.0; 3], 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    let weight = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma_s * sigma_s)).exp()
                        * (-dist2(&labs[i], &labs[j]) / (2.0 * sigma_r * sigma_r)).exp();
                    sum[0] += weight * pixels[j].r;
                    sum[1] += weight * pixels[j].g;
                    sum[2] += weight * pixels[j].b;
                    total += weight;
                }
            }
            *out.get_mut(x, y).unwrap() =
                Color::rgb(sum[0] / total, sum[1] / total, sum[2] / total);
        }
    }
    Ok(out)
}

/// 1-D k-means of one channel into `levels` values, seeded at evenly spaced
/// quantiles. Labels are ranked by value, 0 = smallest.
pub(crate) fn kmeans_1d(values: &[f64], levels: usize) -> Vec<usize> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut centres: Vec<f64> = (0..levels)
        .map(|i| sorted[(2 * i + 1) * sorted.len() / (2 * levels)])
        .collect();
    let assign = |centres: &[f64]| -> Vec<usize> {
        values
            .iter()
            .map(|v| {
                (0..centres.len())
                    .min_by(|&a, &b| (v - centres[a]).abs().total_cmp(&(v - centres[b]).abs()))
                    .unwrap()
            })
            .collect()
    };
    let mut labels = assign(&centres);
    for _ in 0..KMEANS_ROUNDS {
        let mut sums = vec![(0.0, 0usize); levels];
        for (v, &l) in values.iter().zip(&labels) {
            sums[l].0 += v;
            sums[l].1 += 1;
        }
        for (c, (s, n)) in centres.iter_mut().zip(&sums) {
            if *n > 0 {
                *c = s / *n as f64;
            }
        }
        let next = assign(&centres);
        if next == labels {
            break;
        }
        labels = next;
    }
    let mut order: Vec<usize> = (0..levels).collect();
    order.sort_by(|&a, &b| centres[a].total_cmp(&centres[b]).then(a.cmp(&b)));
    let mut rank = vec![0; levels];
    for (r, &l) in order.iter().enumerate() {
        rank[l] = r;
    }
    labels.iter().map(|l| rank[*l]).collect()
}

/// Mean value of each level label, dropping levels left empty and keeping
/// the order (0 = smallest).
pub(crate) fn levels_of(values: &[f64], labels: &mut [usize]) -> Vec<f64> {
    let top = labels.iter().copied().max().unwrap_or(0);
    let mut sums = vec![(0.0, 0usize); top + 1];
    for (v, &l) in values.iter().zip(labels.iter()) {
        sums[l].0 += v;
        sums[l].1 += 1;
    }
    let kept: Vec<usize> = (0..=top).filter(|&l| sums[l].1 > 0).collect();
    let mut rank = vec![0; top + 1];
    for (r, &l) in kept.iter().enumerate() {
        rank[l] = r;
    }
    for l in labels.iter_mut() {
        *l = rank[*l];
    }
    kept.iter().map(|&l| sums[l].0 / sums[l].1 as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ssim;

    #[test]
    fn levels_are_ranked_and_bilateral_keeps_edges() {
        let values: Vec<f64> = [0.9, 0.1, 0.5, 0.11, 0.52, 0.88].repeat(5);
        let labels = kmeans_1d(&values, 3);
        assert_eq!(&labels[..6], &[2, 0, 1, 0, 1, 2]);
        // A sharp black/white edge with one noisy pixel on the white side.
        let (w, h) = (16, 16);
        let mut image = Image::new(w, h, Color::rgb(1.0, 1.0, 1.0)).unwrap();
        for y in 0..h {
            for x in 0..5 {
                *image.get_mut(x, y).unwrap() = Color::rgb(0.0, 0.0, 0.0);
            }
        }
        *image.get_mut(7, 3).unwrap() = Color::rgb(0.94, 0.94, 0.94);
        let smooth = bilateral(&image).unwrap();
        assert!(smooth.get(4, 3).unwrap().r < 0.02 && smooth.get(5, 3).unwrap().r > 0.98);
        assert!(smooth.get(7, 3).unwrap().r > 0.95);
        assert!((ssim(&image, &image) - 1.0).abs() < 1e-12);
        assert!(ssim(&smooth, &image) > 0.9);
    }
}
