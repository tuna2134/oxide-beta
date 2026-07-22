pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid shape: {0}")]
    InvalidShape(String),

    #[error("CUDA execution failed: {0}")]
    Execution(String),

    #[cfg(feature = "cuda")]
    #[error("failed to load {component} shared library or symbol: {source}")]
    DynamicLibrary {
        component: &'static str,
        #[source]
        source: libloading::Error,
    },
}
