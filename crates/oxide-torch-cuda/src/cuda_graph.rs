//! Minimal CUDA Driver graph-capture wrapper.
#![allow(unsafe_code)]

use crate::{Error, Result};
use cuda_core::{CudaContext, CudaStream};
use libloading::Library;
use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

type CuResult = c_int;
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
type CuGraphNode = *mut c_void;
type CuStream = *mut c_void;

type StreamBeginCapture = unsafe extern "C" fn(CuStream, c_uint) -> CuResult;
type StreamEndCapture = unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult;
type GraphInstantiate =
    unsafe extern "C" fn(*mut CuGraphExec, CuGraph, *mut CuGraphNode, *mut i8, usize) -> CuResult;
type GraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut CuGraphExec, CuGraph, c_ulonglong) -> CuResult;
type GraphLaunch = unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult;
type GraphDestroy = unsafe extern "C" fn(CuGraph) -> CuResult;
type GraphExecDestroy = unsafe extern "C" fn(CuGraphExec) -> CuResult;

const STREAM_CAPTURE_MODE_THREAD_LOCAL: c_uint = 1;

/// Owns an instantiated CUDA graph and the driver library containing its ABI.
pub struct CudaGraphExec {
    graph: NonNull<c_void>,
    executable: NonNull<c_void>,
    graph_destroy: GraphDestroy,
    executable_destroy: GraphExecDestroy,
    launch: GraphLaunch,
    context: Arc<CudaContext>,
    _library: Library,
}

impl CudaGraphExec {
    /// Captures all work enqueued by `record` onto `stream` and instantiates it.
    ///
    /// # Errors
    ///
    /// Returns an error when the CUDA Driver API is unavailable or capture,
    /// recording, or graph instantiation fails.
    pub fn capture<E>(
        stream: &CudaStream,
        record: impl FnOnce() -> std::result::Result<(), E>,
    ) -> Result<Self>
    where
        E: std::fmt::Display,
    {
        stream.synchronize().map_err(cuda_error)?;
        let library = ["libcuda.so.1", "libcuda.so"]
            .into_iter()
            .find_map(|name| unsafe { Library::new(name).ok() })
            .ok_or_else(|| Error::Execution("CUDA driver library was not found".into()))?;

        // SAFETY: symbol names and signatures follow the CUDA Driver API.
        let begin: StreamBeginCapture = unsafe {
            *library
                .get(b"cuStreamBeginCapture_v2\0")
                .or_else(|_| library.get(b"cuStreamBeginCapture\0"))
                .map_err(ffi_error)?
        };
        // SAFETY: same ABI guarantee as above.
        let end: StreamEndCapture =
            unsafe { *library.get(b"cuStreamEndCapture\0").map_err(ffi_error)? };
        // SAFETY: same ABI guarantee as above.
        let launch: GraphLaunch = unsafe { *library.get(b"cuGraphLaunch\0").map_err(ffi_error)? };
        // SAFETY: same ABI guarantee as above.
        let graph_destroy: GraphDestroy =
            unsafe { *library.get(b"cuGraphDestroy\0").map_err(ffi_error)? };
        // SAFETY: same ABI guarantee as above.
        let executable_destroy: GraphExecDestroy =
            unsafe { *library.get(b"cuGraphExecDestroy\0").map_err(ffi_error)? };

        // SAFETY: `stream` is live and bound to the model CUDA context.
        check(
            unsafe { begin(stream.cu_stream().cast(), STREAM_CAPTURE_MODE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )?;
        if let Err(error) = record() {
            // End capture even on a recording error so the stream does not
            // remain permanently invalidated for subsequent fallback work.
            let mut discarded = std::ptr::null_mut();
            // SAFETY: capture was successfully begun on this stream.
            let _ = unsafe { end(stream.cu_stream().cast(), &mut discarded) };
            if !discarded.is_null() {
                // SAFETY: a non-null graph returned by the driver is owned here.
                let _ = unsafe { graph_destroy(discarded) };
            }
            return Err(Error::Execution(format!(
                "CUDA graph recording failed: {error}"
            )));
        }

        let mut graph = std::ptr::null_mut();
        // SAFETY: capture is active and `graph` is valid writable storage.
        check(
            unsafe { end(stream.cu_stream().cast(), &mut graph) },
            "cuStreamEndCapture",
        )?;
        let graph = NonNull::new(graph)
            .ok_or_else(|| Error::Execution("CUDA capture returned a null graph".into()))?;

        let mut executable = std::ptr::null_mut();
        // Prefer the modern flags API but retain compatibility with CUDA 12
        // drivers exposing only `cuGraphInstantiate_v2`.
        let instantiate_result = unsafe {
            if let Ok(symbol) =
                library.get::<GraphInstantiateWithFlags>(b"cuGraphInstantiateWithFlags\0")
            {
                symbol(&mut executable, graph.as_ptr(), 0)
            } else {
                let instantiate: GraphInstantiate = *library
                    .get(b"cuGraphInstantiate_v2\0")
                    .or_else(|_| library.get(b"cuGraphInstantiate\0"))
                    .map_err(ffi_error)?;
                instantiate(
                    &mut executable,
                    graph.as_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            }
        };
        if let Err(error) = check(instantiate_result, "cuGraphInstantiate") {
            // SAFETY: `graph` is owned and has not been destroyed.
            let _ = unsafe { graph_destroy(graph.as_ptr()) };
            return Err(error);
        }
        let executable = NonNull::new(executable).ok_or_else(|| {
            // SAFETY: `graph` is owned and has not been destroyed.
            let _ = unsafe { graph_destroy(graph.as_ptr()) };
            Error::Execution("CUDA graph instantiation returned null".into())
        })?;

        Ok(Self {
            graph,
            executable,
            graph_destroy,
            executable_destroy,
            launch,
            context: Arc::clone(stream.context()),
            _library: library,
        })
    }

    /// Enqueues one replay on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error when the context cannot be bound or graph launch fails.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        self.context.bind_to_thread().map_err(cuda_error)?;
        // SAFETY: both handles remain live for `self` and `stream` lifetimes.
        check(
            unsafe { (self.launch)(self.executable.as_ptr(), stream.cu_stream().cast()) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() {
            return;
        }
        // Destroy executable before its source graph. CUDA permits destroying
        // the source earlier, but this order keeps ownership straightforward.
        // SAFETY: both handles are uniquely owned and destroyed exactly once.
        let _ = unsafe { (self.executable_destroy)(self.executable.as_ptr()) };
        // SAFETY: graph is uniquely owned and destroyed exactly once.
        let _ = unsafe { (self.graph_destroy)(self.graph.as_ptr()) };
    }
}

fn check(status: CuResult, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Execution(format!(
            "{operation} failed with CUDA status {status}"
        )))
    }
}

fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(format!("CUDA graph synchronization failed: {error}"))
}

fn ffi_error(error: libloading::Error) -> Error {
    Error::Execution(format!("failed to load CUDA graph symbol: {error}"))
}
