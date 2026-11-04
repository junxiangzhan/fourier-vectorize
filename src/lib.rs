//! Draws a picture as Fourier series. Each Lab channel is cut into levels,
//! each level's outlines are fitted with a Fourier series, and the result is
//! drawn straight from the series, as a browser SVG or a bitmap, at any scale.
//!
//! [`fit_image`] writes the coefficients; [`render`] draws them. Both let the
//! user trade size against accuracy: see [`Prune`] and [`SvgDetail`].
mod bezier;
pub mod bmp;
mod color;
mod decode;
mod drawing;
mod error;
mod fit;
mod fourier;
mod image;
mod outline;
mod packed;
mod pipeline;
mod preprocess;
mod prune;
mod raster;

pub use drawing::SvgDetail;
pub use error::Error;
pub use fit::FitKind;
pub use fourier::{fft, Complex};
pub use image::{Color, Image};
pub use pipeline::{fit_image, pack, render, FitOptions, Fitted, DEFAULT_MOVEMENT};
pub use prune::{Prune, PruneBy};
pub use raster::Rasterizer;
