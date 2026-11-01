//! Small, dependency-free geometry and rasterization toolkit.
pub mod bmp;
pub mod cluster;
mod error;
pub mod fit;
pub mod fourier;
pub mod geometry;
pub mod hilbert;
pub mod image;
pub mod math;
pub mod path;
pub mod raster;

pub use error::Error;
pub use fourier::{fft, Complex};
pub use geometry::{distance_to_segment, CubicBezier, Point2, Polyline};
pub use image::{Color, Image};
pub use math::{Matrix, Vector};
pub use path::{FlattenedPath, Path, Segment, Subpath};
pub use raster::{winding_at, Rasterizer};
