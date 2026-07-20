//! Hugging Face `SafeTensors` loading with transparent dtype conversion.

use crate::{Device, Error, Result, Tensor};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

/// A decoded row-major tensor from a `SafeTensors` checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl LoadedTensor {
    /// Converts the loaded tensor into an oxide-torch tensor on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape is invalid or the device is unavailable.
    pub fn into_tensor(self, device: Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(self.data, self.shape)?.to(device))
    }
}

/// Loader for one `SafeTensors` file or a Hugging Face sharded checkpoint.
///
/// Files are opened only when a tensor is requested, so a large Gemma model is
/// not duplicated in host memory during construction.
#[derive(Clone, Debug)]
pub struct SafeTensorLoader {
    root: PathBuf,
    weight_map: HashMap<String, PathBuf>,
}

impl SafeTensorLoader {
    /// Opens a `.safetensors` file, an index JSON, or a model directory.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, unreadable, or malformed checkpoint files.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            if index.exists() {
                return Self::from_index(&index);
            }
            let single = path.join("model.safetensors");
            if single.exists() {
                return Self::from_file(&single);
            }
            return Err(load_error(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                path.display()
            )));
        }
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            Self::from_index(path)
        } else {
            Self::from_file(path)
        }
    }

    fn from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .map_err(|error| load_error(format!("failed to read {}: {error}", path.display())))?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(safetensor_error)?;
        let file_name = path
            .file_name()
            .ok_or_else(|| load_error("SafeTensors path has no file name"))?;
        let weight_map = tensors
            .names()
            .into_iter()
            .map(|name| (name.to_owned(), PathBuf::from(file_name)))
            .collect();
        Ok(Self {
            root: path.parent().unwrap_or_else(|| Path::new(".")).to_owned(),
            weight_map,
        })
    }

    fn from_index(path: &Path) -> Result<Self> {
        let json = fs::read(path)
            .map_err(|error| load_error(format!("failed to read {}: {error}", path.display())))?;
        let index: ShardIndex = serde_json::from_slice(&json)
            .map_err(|error| load_error(format!("invalid SafeTensors index: {error}")))?;
        let root = path.parent().unwrap_or_else(|| Path::new(".")).to_owned();
        let mut validated = BTreeSet::new();
        for shard in index.weight_map.values() {
            if validated.insert(shard) && !root.join(shard).is_file() {
                return Err(load_error(format!("missing SafeTensors shard {shard}")));
            }
        }
        Ok(Self {
            root,
            weight_map: index
                .weight_map
                .into_iter()
                .map(|(name, shard)| (name, PathBuf::from(shard)))
                .collect(),
        })
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn contains_any<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> bool {
        names.into_iter().any(|name| self.contains(name))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    /// Loads one tensor and converts F32/F16/BF16 to f32.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is missing, malformed, or has an unsupported dtype.
    pub fn load(&self, name: &str) -> Result<LoadedTensor> {
        let shard = self
            .weight_map
            .get(name)
            .ok_or_else(|| load_error(format!("tensor `{name}` is missing")))?;
        let path = self.root.join(shard);
        let bytes = fs::read(&path)
            .map_err(|error| load_error(format!("failed to read {}: {error}", path.display())))?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(safetensor_error)?;
        let view = tensors.tensor(name).map_err(safetensor_error)?;
        let shape = view.shape().to_vec();
        let data = decode_f32(view.dtype(), view.data())?;
        let expected = shape.iter().try_fold(1usize, |size, dimension| {
            size.checked_mul(*dimension)
                .ok_or_else(|| load_error(format!("tensor `{name}` shape overflows usize")))
        })?;
        if data.len() != expected {
            return Err(load_error(format!(
                "tensor `{name}` contains {} values, shape requires {expected}",
                data.len()
            )));
        }
        Ok(LoadedTensor { shape, data })
    }

    /// Tries aliases in order. This handles `model.*`, `language_model.*`, and
    /// base-model checkpoints without forcing a physical conversion step.
    ///
    /// # Errors
    ///
    /// Returns an error if no alias exists or the selected tensor cannot be decoded.
    pub fn load_any<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Result<LoadedTensor> {
        let names: Vec<_> = names.into_iter().collect();
        for name in &names {
            if self.contains(name) {
                return self.load(name);
            }
        }
        Err(load_error(format!(
            "none of the tensor aliases exist: {}",
            names.join(", ")
        )))
    }
}

fn decode_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = |width| {
        if bytes.len() % width == 0 {
            Ok(bytes.chunks_exact(width))
        } else {
            Err(load_error("misaligned SafeTensors payload"))
        }
    };
    match dtype {
        Dtype::F32 => chunks(4).map(|values| {
            values
                .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte chunk")))
                .collect()
        }),
        Dtype::F16 => chunks(2).map(|values| {
            values
                .map(|value| f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
                .collect()
        }),
        Dtype::BF16 => chunks(2).map(|values| {
            values
                .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
                .collect()
        }),
        other => Err(load_error(format!(
            "unsupported SafeTensors dtype {other:?}; Gemma weights must be F32, F16, or BF16"
        ))),
    }
}

#[allow(clippy::needless_pass_by_value)]
fn safetensor_error(error: safetensors::SafeTensorError) -> Error {
    load_error(format!("invalid SafeTensors data: {error}"))
}

fn load_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_common_hugging_face_float_types() {
        let f32_bytes = [1.5_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        assert_eq!(decode_f32(Dtype::F32, &f32_bytes).unwrap(), [1.5, -2.0]);

        let f16_bytes = [
            f16::from_f32(1.5).to_bits().to_le_bytes(),
            f16::from_f32(-2.0).to_bits().to_le_bytes(),
        ]
        .concat();
        assert_eq!(decode_f32(Dtype::F16, &f16_bytes).unwrap(), [1.5, -2.0]);

        let bf16_bytes = [
            bf16::from_f32(1.5).to_bits().to_le_bytes(),
            bf16::from_f32(-2.0).to_bits().to_le_bytes(),
        ]
        .concat();
        assert_eq!(decode_f32(Dtype::BF16, &bf16_bytes).unwrap(), [1.5, -2.0]);
    }
}
