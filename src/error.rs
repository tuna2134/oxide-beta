use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidShape(String),
    DeviceMismatch,
    CudaUnavailable,
    Execution(String),
    Trace(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(message) => write!(f, "invalid shape: {message}"),
            Self::DeviceMismatch => f.write_str("all tensors must be on the same device"),
            Self::CudaUnavailable => {
                f.write_str("CUDA support is disabled; rebuild with --features cuda")
            }
            Self::Execution(message) => write!(f, "execution failed: {message}"),
            Self::Trace(message) => write!(f, "JIT trace failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}
