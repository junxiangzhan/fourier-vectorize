use std::{
    array,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

/// Fixed-size row-major matrix. Incompatible multiplication is a type error.
///
/// ```compile_fail
/// use multimedia::Matrix;
/// let _ = Matrix::<2, 3>::zeros() * Matrix::<2, 2>::zeros();
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<const R: usize, const C: usize> {
    rows: [[f64; C]; R],
}
/// Column vector: an N x 1 matrix, so the same multiplication applies.
pub type Vector<const N: usize> = Matrix<N, 1>;

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub const fn from_rows(rows: [[f64; C]; R]) -> Self {
        Self { rows }
    }
    pub const fn zeros() -> Self {
        Self {
            rows: [[0.0; C]; R],
        }
    }
    pub fn rows(&self) -> &[[f64; C]; R] {
        &self.rows
    }
    pub fn transpose(self) -> Matrix<C, R> {
        Matrix::from_rows(array::from_fn(|c| array::from_fn(|r| self[(r, c)])))
    }
}
impl<const N: usize> Matrix<N, N> {
    pub fn identity() -> Self {
        Self::from_rows(array::from_fn(|r| {
            array::from_fn(|c| if r == c { 1.0 } else { 0.0 })
        }))
    }
}
impl<const N: usize> Matrix<N, 1> {
    pub fn from_array(values: [f64; N]) -> Self {
        Self::from_rows(values.map(|v| [v]))
    }
    pub fn dot(self, other: Self) -> f64 {
        (0..N).map(|n| self[(n, 0)] * other[(n, 0)]).sum()
    }
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }
    /// Euclidean length; repeated hypot avoids overflow in the squares.
    pub fn norm(self) -> f64 {
        (0..N).fold(0.0_f64, |length, n| length.hypot(self[(n, 0)]))
    }
}
impl Matrix<2, 2> {
    /// Standard rotation matrix [[cos, -sin], [sin, cos]]. In screen
    /// coordinates (y down) a positive angle turns visually clockwise.
    pub fn rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_rows([[c, -s], [s, c]])
    }
}
impl<const R: usize, const C: usize> Index<(usize, usize)> for Matrix<R, C> {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.rows[r][c]
    }
}
impl<const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<R, C> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.rows[r][c]
    }
}
impl<const R: usize, const C: usize> Add for Matrix<R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_rows(array::from_fn(|r| {
            array::from_fn(|c| self[(r, c)] + rhs[(r, c)])
        }))
    }
}
impl<const R: usize, const C: usize> Sub for Matrix<R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_rows(array::from_fn(|r| {
            array::from_fn(|c| self[(r, c)] - rhs[(r, c)])
        }))
    }
}
impl<const R: usize, const C: usize> Mul<f64> for Matrix<R, C> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::from_rows(self.rows.map(|row| row.map(|v| v * rhs)))
    }
}
impl<const R: usize, const C: usize> Div<f64> for Matrix<R, C> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::from_rows(self.rows.map(|row| row.map(|v| v / rhs)))
    }
}
impl<const R: usize, const C: usize> Mul<Matrix<R, C>> for f64 {
    type Output = Matrix<R, C>;
    fn mul(self, rhs: Matrix<R, C>) -> Self::Output {
        rhs * self
    }
}
/// Matrix product (R x K) * (K x C) = (R x C); entry (r, c) is the dot
/// product of row r and column c. Mismatched inner sizes do not compile.
impl<const R: usize, const K: usize, const C: usize> Mul<Matrix<K, C>> for Matrix<R, K> {
    type Output = Matrix<R, C>;
    fn mul(self, rhs: Matrix<K, C>) -> Self::Output {
        Matrix::from_rows(array::from_fn(|r| {
            array::from_fn(|c| (0..K).map(|k| self[(r, k)] * rhs[(k, c)]).sum())
        }))
    }
}
