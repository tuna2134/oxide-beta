use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidShape(String),
    Execution(String),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(message) => write!(formatter, "invalid shape: {message}"),
            Self::Execution(message) => write!(formatter, "CUDA execution failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}
