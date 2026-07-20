//! Minimal dynamically-loaded cuDNN convolution backend.

// All unsafe operations in this module are calls through symbols loaded from
// NVIDIA's cudnn.h ABI or conversions of cuda-oxide device/stream handles.
#![allow(unsafe_code)]

use crate::{Error, Result};
use cuda_core::{CudaStream, DeviceBuffer};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::ptr;

type Handle = *mut c_void;
type TensorDesc = *mut c_void;
type FilterDesc = *mut c_void;
type ConvDesc = *mut c_void;
type Status = c_int;

const SUCCESS: Status = 0;
const DATA_FLOAT: c_int = 0;
const TENSOR_NCHW: c_int = 0;
const CROSS_CORRELATION: c_int = 1;
const FWD_IMPLICIT_PRECOMP_GEMM: c_int = 1;
const BWD_FILTER_ALGO_1: c_int = 1;
const BWD_DATA_ALGO_1: c_int = 1;

type Create = unsafe extern "C" fn(*mut Handle) -> Status;
type Destroy = unsafe extern "C" fn(Handle) -> Status;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> Status;
type CreateTensor = unsafe extern "C" fn(*mut TensorDesc) -> Status;
type DestroyTensor = unsafe extern "C" fn(TensorDesc) -> Status;
type SetTensor4d =
    unsafe extern "C" fn(TensorDesc, c_int, c_int, c_int, c_int, c_int, c_int) -> Status;
type CreateFilter = unsafe extern "C" fn(*mut FilterDesc) -> Status;
type DestroyFilter = unsafe extern "C" fn(FilterDesc) -> Status;
type SetFilter4d =
    unsafe extern "C" fn(FilterDesc, c_int, c_int, c_int, c_int, c_int, c_int) -> Status;
type CreateConv = unsafe extern "C" fn(*mut ConvDesc) -> Status;
type DestroyConv = unsafe extern "C" fn(ConvDesc) -> Status;
type SetConv2d = unsafe extern "C" fn(
    ConvDesc,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> Status;
type SetGroupCount = unsafe extern "C" fn(ConvDesc, c_int) -> Status;
type Workspace = unsafe extern "C" fn(
    Handle,
    TensorDesc,
    FilterDesc,
    ConvDesc,
    TensorDesc,
    c_int,
    *mut usize,
) -> Status;
type Forward = unsafe extern "C" fn(
    Handle,
    *const c_void,
    TensorDesc,
    *const c_void,
    FilterDesc,
    *const c_void,
    ConvDesc,
    c_int,
    *mut c_void,
    usize,
    *const c_void,
    TensorDesc,
    *mut c_void,
) -> Status;
type AddTensor = unsafe extern "C" fn(
    Handle,
    *const c_void,
    TensorDesc,
    *const c_void,
    *const c_void,
    TensorDesc,
    *mut c_void,
) -> Status;
type BackwardData = unsafe extern "C" fn(
    Handle,
    *const c_void,
    FilterDesc,
    *const c_void,
    TensorDesc,
    *const c_void,
    ConvDesc,
    c_int,
    *mut c_void,
    usize,
    *const c_void,
    TensorDesc,
    *mut c_void,
) -> Status;
type BackwardFilter = unsafe extern "C" fn(
    Handle,
    *const c_void,
    TensorDesc,
    *const c_void,
    TensorDesc,
    *const c_void,
    ConvDesc,
    c_int,
    *mut c_void,
    usize,
    *const c_void,
    FilterDesc,
    *mut c_void,
) -> Status;
type BackwardBias = unsafe extern "C" fn(
    Handle,
    *const c_void,
    TensorDesc,
    *const c_void,
    *const c_void,
    TensorDesc,
    *mut c_void,
) -> Status;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ConvShape {
    pub batch: usize,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
    pub out_channels: usize,
    pub out_height: usize,
    pub out_width: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

struct Plan {
    x: TensorDesc,
    y: TensorDesc,
    w: FilterDesc,
    bias: TensorDesc,
    conv: ConvDesc,
    forward_workspace: DeviceBuffer<u8>,
    data_workspace: DeviceBuffer<u8>,
    filter_workspace: DeviceBuffer<u8>,
}

struct Api {
    destroy: Destroy,
    set_stream: SetStream,
    create_tensor: CreateTensor,
    destroy_tensor: DestroyTensor,
    set_tensor: SetTensor4d,
    create_filter: CreateFilter,
    destroy_filter: DestroyFilter,
    set_filter: SetFilter4d,
    create_conv: CreateConv,
    destroy_conv: DestroyConv,
    set_conv: SetConv2d,
    set_groups: SetGroupCount,
    fwd_workspace: Workspace,
    data_workspace: Workspace,
    filter_workspace: Workspace,
    forward: Forward,
    add_tensor: AddTensor,
    backward_data: BackwardData,
    backward_filter: BackwardFilter,
    backward_bias: BackwardBias,
}

pub(crate) struct Cudnn {
    handle: Handle,
    api: Api,
    plans: HashMap<ConvShape, Plan>,
    _library: Library,
}

// SAFETY: access is serialized by the per-device Executor mutex. cuDNN handles
// are rebound to the executor's stream before every operation.
unsafe impl Send for Cudnn {}

impl Cudnn {
    pub(crate) fn try_new() -> Option<Self> {
        for name in ["libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"] {
            // SAFETY: symbols are copied as function pointers and the library
            // is retained in Cudnn for longer than every pointer.
            let Ok(library) = (unsafe { Library::new(name) }) else {
                continue;
            };
            if let Ok(value) = unsafe { Self::from_library(library) } {
                return Some(value);
            }
        }
        None
    }

    unsafe fn from_library(library: Library) -> std::result::Result<Self, ()> {
        macro_rules! load {
            ($name:literal, $ty:ty) => {{
                // SAFETY: names and signatures match cudnn.h's legacy API.
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }.map_err(|_| ())?
            }};
        }
        let create: Create = load!("cudnnCreate", Create);
        let api = Api {
            destroy: load!("cudnnDestroy", Destroy),
            set_stream: load!("cudnnSetStream", SetStream),
            create_tensor: load!("cudnnCreateTensorDescriptor", CreateTensor),
            destroy_tensor: load!("cudnnDestroyTensorDescriptor", DestroyTensor),
            set_tensor: load!("cudnnSetTensor4dDescriptor", SetTensor4d),
            create_filter: load!("cudnnCreateFilterDescriptor", CreateFilter),
            destroy_filter: load!("cudnnDestroyFilterDescriptor", DestroyFilter),
            set_filter: load!("cudnnSetFilter4dDescriptor", SetFilter4d),
            create_conv: load!("cudnnCreateConvolutionDescriptor", CreateConv),
            destroy_conv: load!("cudnnDestroyConvolutionDescriptor", DestroyConv),
            set_conv: load!("cudnnSetConvolution2dDescriptor", SetConv2d),
            set_groups: load!("cudnnSetConvolutionGroupCount", SetGroupCount),
            fwd_workspace: load!("cudnnGetConvolutionForwardWorkspaceSize", Workspace),
            data_workspace: load!("cudnnGetConvolutionBackwardDataWorkspaceSize", Workspace),
            filter_workspace: load!("cudnnGetConvolutionBackwardFilterWorkspaceSize", Workspace),
            forward: load!("cudnnConvolutionForward", Forward),
            add_tensor: load!("cudnnAddTensor", AddTensor),
            backward_data: load!("cudnnConvolutionBackwardData", BackwardData),
            backward_filter: load!("cudnnConvolutionBackwardFilter", BackwardFilter),
            backward_bias: load!("cudnnConvolutionBackwardBias", BackwardBias),
        };
        let mut handle = ptr::null_mut();
        check(unsafe { create(&mut handle) }).map_err(|_| ())?;
        Ok(Self {
            handle,
            api,
            plans: HashMap::new(),
            _library: library,
        })
    }

    fn plan(&mut self, shape: ConvShape, stream: &CudaStream) -> Result<&mut Plan> {
        if !self.plans.contains_key(&shape) {
            let plan = self.create_plan(shape, stream)?;
            self.plans.insert(shape, plan);
        }
        Ok(self.plans.get_mut(&shape).expect("inserted cuDNN plan"))
    }

    fn create_plan(&self, s: ConvShape, stream: &CudaStream) -> Result<Plan> {
        let mut x = ptr::null_mut();
        let mut y = ptr::null_mut();
        let mut bias = ptr::null_mut();
        let mut w = ptr::null_mut();
        let mut conv = ptr::null_mut();
        unsafe {
            check((self.api.create_tensor)(&mut x))?;
            check((self.api.create_tensor)(&mut y))?;
            check((self.api.create_tensor)(&mut bias))?;
            check((self.api.create_filter)(&mut w))?;
            check((self.api.create_conv)(&mut conv))?;
            check((self.api.set_tensor)(
                x,
                TENSOR_NCHW,
                DATA_FLOAT,
                i(s.batch)?,
                i(s.in_channels)?,
                i(s.height)?,
                i(s.width)?,
            ))?;
            check((self.api.set_tensor)(
                y,
                TENSOR_NCHW,
                DATA_FLOAT,
                i(s.batch)?,
                i(s.out_channels)?,
                i(s.out_height)?,
                i(s.out_width)?,
            ))?;
            check((self.api.set_tensor)(
                bias,
                TENSOR_NCHW,
                DATA_FLOAT,
                1,
                i(s.out_channels)?,
                1,
                1,
            ))?;
            check((self.api.set_filter)(
                w,
                DATA_FLOAT,
                TENSOR_NCHW,
                i(s.out_channels)?,
                i(s.in_channels / s.groups)?,
                i(s.kernel)?,
                i(s.kernel)?,
            ))?;
            check((self.api.set_conv)(
                conv,
                i(s.padding)?,
                i(s.padding)?,
                i(s.stride)?,
                i(s.stride)?,
                1,
                1,
                CROSS_CORRELATION,
                DATA_FLOAT,
            ))?;
            check((self.api.set_groups)(conv, i(s.groups)?))?;
        }
        let mut fw = 0;
        let mut dw = 0;
        let mut ww = 0;
        unsafe {
            check((self.api.fwd_workspace)(
                self.handle,
                x,
                w,
                conv,
                y,
                FWD_IMPLICIT_PRECOMP_GEMM,
                &mut fw,
            ))?;
            check((self.api.data_workspace)(
                self.handle,
                w,
                y,
                conv,
                x,
                BWD_DATA_ALGO_1,
                &mut dw,
            ))?;
            check((self.api.filter_workspace)(
                self.handle,
                x,
                y,
                conv,
                w,
                BWD_FILTER_ALGO_1,
                &mut ww,
            ))?;
        }
        Ok(Plan {
            x,
            y,
            w,
            bias,
            conv,
            forward_workspace: DeviceBuffer::zeroed(stream, fw.max(1)).map_err(cuda_error)?,
            data_workspace: DeviceBuffer::zeroed(stream, dw.max(1)).map_err(cuda_error)?,
            filter_workspace: DeviceBuffer::zeroed(stream, ww.max(1)).map_err(cuda_error)?,
        })
    }

    pub(crate) fn forward(
        &mut self,
        s: ConvShape,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.bind(stream)?;
        let handle = self.handle;
        let api = &self.api as *const Api;
        let p = self.plan(s, stream)?;
        let (one, zero) = (1.0f32, 0.0f32);
        let workspace_size = p.forward_workspace.len();
        unsafe {
            let a = &*api;
            check((a.forward)(
                handle,
                ptr_of(&one),
                p.x,
                dev(input),
                p.w,
                dev(weight),
                p.conv,
                FWD_IMPLICIT_PRECOMP_GEMM,
                dev_mut(&mut p.forward_workspace),
                workspace_size,
                ptr_of(&zero),
                p.y,
                dev_mut(output),
            ))?;
            check((a.add_tensor)(
                handle,
                ptr_of(&one),
                p.bias,
                dev(bias),
                ptr_of(&one),
                p.y,
                dev_mut(output),
            ))
        }
    }

    pub(crate) fn backward(
        &mut self,
        s: ConvShape,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        dx: &mut DeviceBuffer<f32>,
        dw: &mut DeviceBuffer<f32>,
        db: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.bind(stream)?;
        let handle = self.handle;
        let api = &self.api as *const Api;
        let p = self.plan(s, stream)?;
        let (one, zero) = (1.0f32, 0.0f32);
        let data_workspace_size = p.data_workspace.len();
        let filter_workspace_size = p.filter_workspace.len();
        unsafe {
            let a = &*api;
            check((a.backward_data)(
                handle,
                ptr_of(&one),
                p.w,
                dev(weight),
                p.y,
                dev(gradient),
                p.conv,
                BWD_DATA_ALGO_1,
                dev_mut(&mut p.data_workspace),
                data_workspace_size,
                ptr_of(&zero),
                p.x,
                dev_mut(dx),
            ))?;
            check((a.backward_filter)(
                handle,
                ptr_of(&one),
                p.x,
                dev(input),
                p.y,
                dev(gradient),
                p.conv,
                BWD_FILTER_ALGO_1,
                dev_mut(&mut p.filter_workspace),
                filter_workspace_size,
                ptr_of(&zero),
                p.w,
                dev_mut(dw),
            ))?;
            check((a.backward_bias)(
                handle,
                ptr_of(&one),
                p.y,
                dev(gradient),
                ptr_of(&zero),
                p.bias,
                dev_mut(db),
            ))
        }
    }

    fn bind(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check((self.api.set_stream)(
                self.handle,
                stream.cu_stream().cast(),
            ))
        }
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        unsafe {
            for (_, p) in self.plans.drain() {
                let _ = (self.api.destroy_tensor)(p.x);
                let _ = (self.api.destroy_tensor)(p.y);
                let _ = (self.api.destroy_tensor)(p.bias);
                let _ = (self.api.destroy_filter)(p.w);
                let _ = (self.api.destroy_conv)(p.conv);
            }
            let _ = (self.api.destroy)(self.handle);
        }
    }
}

fn check(status: Status) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(Error::Execution(format!("cuDNN status {status}")))
    }
}
fn i(value: usize) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| Error::Execution("cuDNN dimension exceeds c_int".into()))
}
fn ptr_of<T>(value: &T) -> *const c_void {
    std::ptr::from_ref(value).cast()
}
fn dev<T>(value: &DeviceBuffer<T>) -> *const c_void {
    value.cu_deviceptr() as usize as *const c_void
}
fn dev_mut<T>(value: &mut DeviceBuffer<T>) -> *mut c_void {
    value.cu_deviceptr() as usize as *mut c_void
}
fn cuda_error(error: impl std::fmt::Display) -> Error {
    Error::Execution(error.to_string())
}
