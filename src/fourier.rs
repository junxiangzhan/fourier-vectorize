//! Complex numbers and a radix-2 fast Fourier transform, written from
//! scratch for the Part 4 vectorizing experiments.
use crate::Error;
use std::ops::{Add, Mul, Sub};

/// A complex number re + i*im. A 2D point (x, y) is also read as x + i*y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}
impl Complex {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    /// e^(i*angle) = cos(angle) + i*sin(angle): a point on the unit circle.
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn norm_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}
impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// In-place discrete Fourier transform X[f] = sum_n x[n] e^(-2*pi*i*f*n/N).
/// `inverse` computes x[n] = (1/N) sum_f X[f] e^(+2*pi*i*f*n/N) instead.
/// The length must be a power of two.
pub fn fft(data: &mut [Complex], inverse: bool) -> Result<(), Error> {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(Error::InvalidParameter("FFT length must be a power of two"));
    }
    if data.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
        return Err(Error::NonFinite);
    }
    // 1. Reorder by bit-reversed index, so each butterfly pass below can
    //    combine neighbouring blocks in place.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            data.swap(i, j);
        }
    }
    // 2. Cooley-Tukey: merge pairs of half-length transforms into transforms
    //    of length 2, 4, 8, ... N. Each merge is X = E + w*O and E - w*O.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut length = 2;
    while length <= n {
        let angle = sign * std::f64::consts::TAU / length as f64;
        for block in data.chunks_mut(length) {
            let (even, odd) = block.split_at_mut(length / 2);
            for (j, (e, o)) in even.iter_mut().zip(odd.iter_mut()).enumerate() {
                // Twiddle factor computed directly (not by repeated
                // multiplication) so rounding does not accumulate.
                let twiddled = Complex::from_angle(angle * j as f64) * *o;
                *o = *e - twiddled;
                *e = *e + twiddled;
            }
        }
        length *= 2;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for value in data.iter_mut() {
            *value = value.scale(scale);
        }
    }
    Ok(())
}
