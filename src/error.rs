use std::{fmt, io};

/// Every failure the library and experiments can report. Invalid input is
/// returned as an error instead of being silently corrected.
#[derive(Debug)]
pub enum Error {
    InvalidDimensions,
    InvalidFactor,
    SizeOverflow,
    Allocation,
    NonFinite,
    OpenPath,
    InvalidParameter(&'static str),
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "image dimensions must be positive"),
            Self::InvalidFactor => write!(
                f,
                "sampling factor must be positive and divide the image dimensions"
            ),
            Self::SizeOverflow => write!(
                f,
                "requested dimensions or subdivision exceed supported sizes"
            ),
            Self::Allocation => write!(f, "unable to allocate requested storage"),
            Self::NonFinite => write!(f, "geometry and color values must be finite"),
            Self::OpenPath => write!(f, "filling requires closed subpaths"),
            Self::InvalidParameter(message) => write!(f, "{message}"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
