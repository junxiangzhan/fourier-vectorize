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

/// `shrink` for a picture with alpha, so a hidden pixel adds nothing to the
/// colour of its block, then gives the pixels left hidden colours of their own
/// (see `fill_hidden`). The alpha comes back only if the shrunken picture still
/// has any transparency.
pub(crate) fn shrink_transparent(image: &Image<Color>, alpha: &Image<f64>, factor: usize) -> Result<(Image<Color>, Option<Image<f64>>), Error> {
    let mut premultiplied = image.clone();
    let mut grey = image.clone();
    for (i, a) in alpha.pixels().iter().enumerate() {
        let (x, y) = (i % image.width(), i / image.width());
        let c = premultiplied.get_mut(x, y).unwrap();
        *c = Color::rgb(c.r * a, c.g * a, c.b * a);
        *grey.get_mut(x, y).unwrap() = Color::rgb(*a, *a, *a);
    }
    let (premultiplied, grey) = (shrink(&premultiplied, factor)?, shrink(&grey, factor)?);
    let alpha = grey.map(|c| c.r)?;
    let w = alpha.width();
    let mut colours = premultiplied;
    for (i, a) in alpha.pixels().iter().enumerate() {
        let c = colours.get_mut(i % w, i / w).unwrap();
        if *a > 0.0 {
            *c = Color::rgb(c.r / a, c.g / a, c.b / a);
        }
    }
    if alpha.pixels().iter().all(|a| *a >= 1.0) {
        return Ok((colours, None));
    }
    fill_hidden(&mut colours, &alpha)?;
    Ok((colours, Some(alpha)))
}

/// Gives every clear pixel a colour continuing what is visible, so colour
/// outlines cross the edge of the alpha cleanly and spend little beyond it,
/// whatever colour the file stored there. Each level of a pyramid averages the
/// visible pixels below it, then from the coarsest down a pixel with no colour
/// takes a bilinear blend of the level above. Compared on three cut-out
/// paintings, this drew about as well as or better than copying the nearest visible
/// colour, with a third smaller SVGs on the sharpest cut. Nothing visible: nothing to do.
fn fill_hidden(image: &mut Image<Color>, alpha: &Image<f64>) -> Result<(), Error> {
    let known = image.pixels().iter().zip(alpha.pixels()).map(|(c, a)| (*a > 0.0).then_some(*c)).collect();
    let mut pyramid: Vec<(usize, usize, Vec<Option<Color>>)> = vec![(image.width(), image.height(), known)];
    while let Some((w, h, fine)) = pyramid.last().filter(|(w, h, _)| *w > 1 || *h > 1) {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut coarse = vec![None; cw * ch];
        for (i, cell) in coarse.iter_mut().enumerate() {
            let (x, y) = (2 * (i % cw), 2 * (i / cw));
            let children: Vec<Color> = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                .iter()
                .filter(|(cx, cy)| cx < w && cy < h)
                .filter_map(|(cx, cy)| fine[cy * w + cx])
                .collect();
            if !children.is_empty() {
                let n = children.len() as f64;
                let sum = |f: fn(&Color) -> f64| children.iter().map(f).sum::<f64>() / n;
                *cell = Some(Color::rgb(sum(|c| c.r), sum(|c| c.g), sum(|c| c.b)));
            }
        }
        pyramid.push((cw, ch, coarse));
    }
    if pyramid[pyramid.len() - 1].2[0].is_none() {
        return Ok(());
    }
    while pyramid.len() > 1 {
        let (cw, ch, coarse) = pyramid.pop().unwrap();
        let (w, _, fine) = pyramid.last_mut().unwrap();
        for (i, pixel) in fine.iter_mut().enumerate().filter(|(_, p)| p.is_none()) {
            let u = (((i % *w) as f64 + 0.5) / 2.0 - 0.5).clamp(0.0, cw as f64 - 1.0);
            let v = (((i / *w) as f64 + 0.5) / 2.0 - 0.5).clamp(0.0, ch as f64 - 1.0);
            let (x0, y0) = (u as usize, v as usize);
            let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(ch - 1));
            let (fu, fv) = (u - x0 as f64, v - y0 as f64);
            // Every pixel of a coarser level has its colour by now.
            let at = |x: usize, y: usize| coarse[y * cw + x].unwrap();
            let blend = |f: fn(Color) -> f64| {
                let top = f(at(x0, y0)) * (1.0 - fu) + f(at(x1, y0)) * fu;
                let bottom = f(at(x0, y1)) * (1.0 - fu) + f(at(x1, y1)) * fu;
                top * (1.0 - fv) + bottom * fv
            };
            *pixel = Some(Color::rgb(blend(|c| c.r), blend(|c| c.g), blend(|c| c.b)));
        }
    }
    let (w, _, filled) = &pyramid[0];
    for (i, colour) in filled.iter().enumerate() {
        *image.get_mut(i % w, i / w).unwrap() = colour.unwrap();
    }
    Ok(())
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
    fn a_block_ignores_what_is_clear_and_clear_pixels_continue_what_is_visible() {
        // Red and half-clear green on the left, clear magenta on the right.
        let (w, h) = (8, 4);
        let mut image = Image::new(w, h, Color::rgb(1.0, 0.0, 1.0)).unwrap();
        let mut alpha = Image::new(w, h, 0.0).unwrap();
        for y in 0..h {
            for x in 0..4 {
                *image.get_mut(x, y).unwrap() = if x < 2 { Color::rgb(1.0, 0.0, 0.0) } else { Color::rgb(0.0, 1.0, 0.0) };
                *alpha.get_mut(x, y).unwrap() = if x < 2 { 1.0 } else { 0.5 };
            }
        }
        let (colours, shrunk) = shrink_transparent(&image, &alpha, 2).unwrap();
        let shrunk = shrunk.unwrap();
        assert_eq!(shrunk.pixels()[..2], [1.0, 0.5]);
        assert_eq!(*colours.get(0, 0).unwrap(), Color::rgb(1.0, 0.0, 0.0));
        assert_eq!(*colours.get(1, 0).unwrap(), Color::rgb(0.0, 1.0, 0.0));
        // No magenta is left: every clear pixel blends the visible colours.
        for c in &colours.pixels()[2..4] {
            assert!(c.b == 0.0 && (c.r + c.g - 1.0).abs() < 1e-9, "{c:?}");
        }
        // Visible pixels keep their colours exactly at no reduction.
        let (same, _) = shrink_transparent(&image, &alpha, 1).unwrap();
        assert_eq!(*same.get(0, 0).unwrap(), Color::rgb(1.0, 0.0, 0.0));
        // All clear: nothing to continue, and no panic.
        let clear = Image::new(w, h, 0.0).unwrap();
        assert!(shrink_transparent(&image, &clear, 1).unwrap().1.is_some());
        // All opaque once the clear rim is cropped away: no alpha at all.
        let mut rim = Image::new(5, 4, 1.0).unwrap();
        for y in 0..4 {
            *rim.get_mut(4, y).unwrap() = 0.0;
        }
        let picture = Image::new(5, 4, Color::rgb(0.5, 0.5, 0.5)).unwrap();
        assert!(shrink_transparent(&picture, &rim, 2).unwrap().1.is_none());
    }

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
