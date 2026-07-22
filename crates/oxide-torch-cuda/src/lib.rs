#![cfg_attr(feature = "cuda", feature(core_intrinsics))]

//! CUDA kernels, graph execution, cuBLAS, and cuDNN primitives for oxide-torch.

#[cfg(feature = "cuda")]
pub mod cublas;
#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cudnn;
mod error;
#[cfg(feature = "cuda")]
mod kernel_module;

pub use error::{Error, Result};
#[cfg(feature = "cuda")]
pub use kernel_module::module as kernels;

#[cfg(feature = "cuda")]
pub fn load_kernels(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> std::result::Result<kernels::LoadedModule, cuda_host::embedded::EmbeddedModuleError> {
    match kernels::load(context) {
        Ok(module) => Ok(module),
        Err(cuda_host::embedded::EmbeddedModuleError::Ltoir(_)) => {
            load_kernels_through_ptx(context)
        }
        Err(error) => Err(error),
    }
}

#[cfg(feature = "cuda")]
fn load_kernels_through_ptx(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> std::result::Result<kernels::LoadedModule, cuda_host::embedded::EmbeddedModuleError> {
    use cuda_host::embedded::{
        ArtifactPayloadKind, EmbeddedModuleError, artifact_bundles_from_current_exe,
    };

    let bundle_name = env!("CARGO_PKG_NAME");
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == bundle_name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: bundle_name.to_owned(),
        })?;

    let ptx = if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        ptx.to_vec()
    } else if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        cuda_host::ltoir::build_ptx_from_nvvm_ir_with_compile_options(
            nvvm_ir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )?
    } else if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        cuda_host::ltoir::link_ltoir_to_ptx_with_compile_options(
            ltoir,
            &bundle.name,
            &bundle.target,
            bundle.compile_options,
        )?
    } else {
        return Err(EmbeddedModuleError::UnsupportedPayload { name: bundle.name });
    };

    let module = context.load_module_from_image(&ptx)?;
    kernels::from_module(module).map_err(EmbeddedModuleError::Driver)
}
