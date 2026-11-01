//! Hilbert space-filling curve: visits every pixel of a 2^m x 2^m grid
//! exactly once, each step moving to a 4-neighbour, so pixels that are close
//! along the curve are also close in the image.
use crate::Error;

/// Position (x, y) of step `d` along the Hilbert curve on an n x n grid,
/// n a power of two. Iterative version of the standard recursive
/// construction: each pair of bits of `d` picks a quadrant, and the
/// sub-curve in that quadrant is rotated/reflected so that the pieces join.
pub fn d2xy(n: usize, d: usize) -> Result<(usize, usize), Error> {
    if n == 0 || !n.is_power_of_two() {
        return Err(Error::InvalidParameter(
            "Hilbert grid size must be a power of two",
        ));
    }
    if d >= n.checked_mul(n).ok_or(Error::SizeOverflow)? {
        return Err(Error::InvalidParameter("Hilbert index outside the grid"));
    }
    let (mut x, mut y) = (0, 0);
    let mut t = d;
    let mut s = 1;
    while s < n {
        // Quadrant of this level: rx = right half, ry = lower half.
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);
        // Orient the sub-curve built so far for this quadrant.
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    Ok((x, y))
}
