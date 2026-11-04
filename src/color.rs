//! Colour: CIE L*a*b*, the ΔE and SSIM measures, and the SVG filter that turns
//! Lab plates back into RGB in a browser.
use crate::{Color, Image};

/// sRGB (0..1) to CIE Lab (D65).
pub(crate) fn lab(c: Color) -> [f64; 3] {
    let linear = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (linear(c.r), linear(c.g), linear(c.b));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f64| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    [
        116.0 * f(y) - 16.0,
        500.0 * (f(x) - f(y)),
        200.0 * (f(y) - f(z)),
    ]
}

pub(crate) fn delta_e(a: Color, b: Color) -> f64 {
    let (a, b) = (lab(a), lab(b));
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// CIE L*a*b* (D65) back to sRGB in [0, 1]: the inverse of `lab`.
pub(crate) fn lab_to_rgb(l: f64, a: f64, b: f64) -> Color {
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
    let finv = |t: f64| {
        if t * t * t > 0.008856 {
            t * t * t
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let (x, y, z) = (0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz));
    let gamma = |v: f64| {
        let v = if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.max(0.0).powf(1.0 / 2.4) - 0.055
        };
        v.clamp(0.0, 1.0)
    };
    Color::rgb(
        gamma(3.2406 * x - 1.5372 * y - 0.4986 * z),
        gamma(-0.9689 * x + 1.8758 * y + 0.0415 * z),
        gamma(0.0557 * x - 0.2040 * y + 1.0570 * z),
    )
}

/// `color` with opacity `alpha` over a white ground.
pub(crate) fn over_white(color: Color, alpha: f64) -> Color {
    let blend = |v: f64| v.clamp(0.0, 1.0) * alpha + (1.0 - alpha);
    Color::rgb(blend(color.r), blend(color.g), blend(color.b))
}

pub(crate) fn hex(color: Color) -> String {
    let c = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", c(color.r), c(color.g), c(color.b))
}

/// Structural similarity of lightness L*: mean SSIM over non-overlapping
/// 8 x 8 blocks (C1 = (0.01 * 100)^2, C2 = (0.03 * 100)^2 on the 0..100 L*
/// scale; NaN below 8 x 8 px). Mean ΔE scores each pixel's colour alone; SSIM asks whether
/// local light and shade, i.e. the structure, survived.
pub(crate) fn ssim(a: &Image<Color>, b: &Image<Color>) -> f64 {
    let (w, h) = (a.width(), a.height());
    let la: Vec<f64> = a.pixels().iter().map(|c| lab(*c)[0]).collect();
    let lb: Vec<f64> = b.pixels().iter().map(|c| lab(*c)[0]).collect();
    let (c1, c2) = (1.0, 9.0);
    let (mut total, mut blocks) = (0.0, 0usize);
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let index = |k: usize| (by * 8 + k / 8) * w + bx * 8 + k % 8;
            let n = 64.0;
            let ma = (0..64).map(|k| la[index(k)]).sum::<f64>() / n;
            let mb = (0..64).map(|k| lb[index(k)]).sum::<f64>() / n;
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for k in 0..64 {
                let (x, y) = (la[index(k)] - ma, lb[index(k)] - mb);
                va += x * x;
                vb += y * y;
                cov += x * y;
            }
            let (va, vb, cov) = (va / n, vb / n, cov / n);
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            blocks += 1;
        }
    }
    // Smaller than one block: no structure to compare.
    if blocks == 0 {
        return f64::NAN;
    }
    total / blocks as f64
}

/// In the browser SVG, each plate lights only its own channel: red, green or
/// blue (r = L*/100, g = (a* + 128)/255, b = (b* + 128)/255), combined with
/// `screen`, which is exact when each plate lights only its own channel. An
/// SVG filter then rebuilds RGB in four primitives: a matrix to f(X), f(Y),
/// f(Z); a per-channel cube; a matrix from XYZ to linear RGB; the sRGB gamma
/// curve. SVG filters clamp every result to [0, 1] and have no piecewise
/// functions, so the short linear pieces of the Lab and sRGB curves near black
/// are left out.
const LAB_TO_F: [[f64; 4]; 3] = [
    // f(X) = f(Y) + a*/500, f(Y) = (L* + 16)/116, f(Z) = f(Y) - b*/200,
    // written in the encoded channels: [r, g, b, constant].
    [
        100.0 / 116.0,
        255.0 / 500.0,
        0.0,
        16.0 / 116.0 - 128.0 / 500.0,
    ],
    [100.0 / 116.0, 0.0, 0.0, 16.0 / 116.0],
    [
        100.0 / 116.0,
        0.0,
        -255.0 / 200.0,
        16.0 / 116.0 + 128.0 / 200.0,
    ],
];
/// Normalized XYZ (X/Xn, Y/Yn, Z/Zn) to linear sRGB, D65.
const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.2406 * 0.95047, -1.5372, -0.4986 * 1.08883],
    [-0.9689 * 0.95047, 1.8758, 0.0415 * 1.08883],
    [0.0557 * 0.95047, -0.2040, 1.0570 * 1.08883],
];

/// The <filter> element for those steps.
pub(crate) fn svg_lab_filter(w: usize, h: usize) -> String {
    let m1: Vec<String> = LAB_TO_F
        .iter()
        .map(|r| format!("{:.6} {:.6} {:.6} 0 {:.6}", r[0], r[1], r[2], r[3]))
        .collect();
    let m2: Vec<String> = XYZ_TO_RGB
        .iter()
        .map(|r| format!("{:.6} {:.6} {:.6} 0 0", r[0], r[1], r[2]))
        .collect();
    let funcs = |attributes: &str| {
        ["R", "G", "B"]
            .iter()
            .map(|c| format!("<feFunc{c} type=\"gamma\" {attributes}/>"))
            .collect::<String>()
    };
    format!(
        "  <defs><filter id=\"lab\" filterUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" color-interpolation-filters=\"sRGB\">\n    <feColorMatrix type=\"matrix\" values=\"{} 0 0 0 1 0\"/>\n    <feComponentTransfer>{}</feComponentTransfer>\n    <feColorMatrix type=\"matrix\" values=\"{} 0 0 0 1 0\"/>\n    <feComponentTransfer>{}</feComponentTransfer>\n  </filter></defs>\n",
        m1.join(" "),
        funcs("amplitude=\"1\" exponent=\"3\" offset=\"0\""),
        m2.join(" "),
        funcs("amplitude=\"1.055\" exponent=\"0.416667\" offset=\"-0.055\"")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The colour a browser computes from the plate values through the filter,
    /// primitive by primitive, with the same clamping. With `eight_bit`, every
    /// intermediate result is also rounded to 1/255, as when a renderer keeps
    /// 8-bit buffers between filter primitives.
    fn svg_filter_colour(l: f64, a: f64, b: f64, eight_bit: bool) -> Color {
        let clamp = |v: f64| {
            let v = v.clamp(0.0, 1.0);
            if eight_bit {
                (v * 255.0).round() / 255.0
            } else {
                v
            }
        };
        let encoded = [
            clamp(l / 100.0),
            clamp((a + 128.0) / 255.0),
            clamp((b + 128.0) / 255.0),
        ];
        let f: Vec<f64> = LAB_TO_F
            .iter()
            .map(|row| clamp(row[0] * encoded[0] + row[1] * encoded[1] + row[2] * encoded[2] + row[3]))
            .collect();
        let xyz: Vec<f64> = f.iter().map(|v| clamp(v * v * v)).collect();
        let rgb: Vec<f64> = XYZ_TO_RGB
            .iter()
            .map(|row| {
                let linear = clamp(row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]);
                clamp(1.055 * linear.powf(1.0 / 2.4) - 0.055)
            })
            .collect();
        Color::rgb(rgb[0], rgb[1], rgb[2])
    }
    
    #[test]
    fn svg_filter_chain_matches_lab_away_from_black() {
        // Mid-tone colours: the filter's missing linear pieces near black
        // do not apply, so it should agree with the exact conversion.
        for c in [
            Color::rgb(0.8, 0.6, 0.5),
            Color::rgb(0.3, 0.5, 0.7),
            Color::rgb(0.9, 0.9, 0.85),
        ] {
            let [l, a, b] = lab(c);
            assert!(delta_e(svg_filter_colour(l, a, b, false), lab_to_rgb(l, a, b)) < 0.05);
            // 8-bit buffers between primitives cost a few ΔE at most.
            assert!(delta_e(svg_filter_colour(l, a, b, true), lab_to_rgb(l, a, b)) < 5.0);
        }
        let filter = svg_lab_filter(10, 20);
        assert_eq!(filter.matches("<feColorMatrix").count(), 2);
        assert_eq!(filter.matches("<feFunc").count(), 6);
    }

    #[test]
    fn lab_round_trips_back_to_rgb() {
        for c in [
            Color::rgb(0.9, 0.1, 0.1),
            Color::rgb(0.2, 0.5, 0.8),
            Color::rgb(0.95, 0.93, 0.88),
            Color::rgb(0.01, 0.02, 0.03),
        ] {
            let [l, a, b] = lab(c);
            let back = lab_to_rgb(l, a, b);
            assert!(
                (back.r - c.r).abs() < 2e-3
                    && (back.g - c.g).abs() < 2e-3
                    && (back.b - c.b).abs() < 2e-3
            );
        }
    }
}
