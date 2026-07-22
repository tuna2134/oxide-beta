pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid shape: {0}")]
    InvalidShape(String),

    #[error("all tensors must be on the same device")]
    DeviceMismatch,

    #[error("CUDA support is disabled; rebuild with --features cuda")]
    CudaUnavailable,

    #[error("execution failed: {0}")]
    Execution(String),

    #[error("JIT trace failed: {0}")]
    Trace(String),

    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid SafeTensors data: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] oxide_torch_cuda::Error),
}

impl Error {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }
}
