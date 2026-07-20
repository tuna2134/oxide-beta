//! Dependency-free reader for the canonical uncompressed MNIST IDX files.

use crate::{Error, Result, Tensor};
use std::fs;
use std::path::Path;

const ROWS: usize = 28;
const COLUMNS: usize = 28;
const IMAGE_MAGIC: u32 = 2051;
const LABEL_MAGIC: u32 = 2049;

#[derive(Clone, Debug)]
pub struct Mnist {
    train_images: Vec<u8>,
    train_labels: Vec<u8>,
    test_images: Vec<u8>,
    test_labels: Vec<u8>,
}

impl Mnist {
    /// Loads the four canonical, uncompressed IDX files from `directory`.
    ///
    /// # Errors
    ///
    /// Returns an error for missing files, malformed IDX headers, or inconsistent counts.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let (train_images, train_count) = parse_images(&read(
            &directory.join("train-images-idx3-ubyte"),
        )?)?;
        let train_labels = parse_labels(&read(&directory.join("train-labels-idx1-ubyte"))?)?;
        let (test_images, test_count) =
            parse_images(&read(&directory.join("t10k-images-idx3-ubyte"))?)?;
        let test_labels = parse_labels(&read(&directory.join("t10k-labels-idx1-ubyte"))?)?;
        if train_count != train_labels.len() || test_count != test_labels.len() {
            return Err(Error::Execution(
                "MNIST image and label counts do not match".into(),
            ));
        }
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Creates normalized training batches.
    ///
    /// # Errors
    ///
    /// Returns an error when `batch_size` is zero.
    pub fn train_batches(&self, batch_size: usize, shuffle: bool) -> Result<MnistBatches<'_>> {
        MnistBatches::new(
            &self.train_images,
            &self.train_labels,
            batch_size,
            shuffle,
        )
    }

    /// Creates normalized test batches in dataset order.
    ///
    /// # Errors
    ///
    /// Returns an error when `batch_size` is zero.
    pub fn test_batches(&self, batch_size: usize) -> Result<MnistBatches<'_>> {
        MnistBatches::new(&self.test_images, &self.test_labels, batch_size, false)
    }

    #[must_use]
    pub fn train_len(&self) -> usize {
        self.train_labels.len()
    }

    #[must_use]
    pub fn test_len(&self) -> usize {
        self.test_labels.len()
    }
}

pub struct MnistBatches<'a> {
    images: &'a [u8],
    labels: &'a [u8],
    indices: Vec<usize>,
    position: usize,
    batch_size: usize,
}

impl<'a> MnistBatches<'a> {
    fn new(images: &'a [u8], labels: &'a [u8], batch_size: usize, shuffle: bool) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::InvalidShape("batch_size must be non-zero".into()));
        }
        let mut indices: Vec<_> = (0..labels.len()).collect();
        if shuffle {
            deterministic_shuffle(&mut indices);
        }
        Ok(Self {
            images,
            labels,
            indices,
            position: 0,
            batch_size,
        })
    }
}

impl Iterator for MnistBatches<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];
        self.position = end;
        let mut images = Vec::with_capacity(batch_indices.len() * ROWS * COLUMNS);
        let mut labels = Vec::with_capacity(batch_indices.len());
        for &index in batch_indices {
            let start = index * ROWS * COLUMNS;
            images.extend(
                self.images[start..start + ROWS * COLUMNS]
                    .iter()
                    .map(|&pixel| (f32::from(pixel) / 255.0 - 0.1307) / 0.3081),
            );
            labels.push(f32::from(self.labels[index]));
        }
        Some(
            Tensor::from_vec(images, vec![batch_indices.len(), 1, ROWS, COLUMNS]).and_then(
                |images| {
                    Tensor::from_vec(labels, vec![batch_indices.len()])
                        .map(|labels| (images, labels))
                },
            ),
        )
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|error| {
        Error::Execution(format!("failed to read MNIST file {}: {error}", path.display()))
    })
}

fn parse_images(bytes: &[u8]) -> Result<(Vec<u8>, usize)> {
    if bytes.len() < 16 || be_u32(bytes, 0)? != IMAGE_MAGIC {
        return Err(Error::Execution("invalid MNIST image IDX header".into()));
    }
    let count = usize::try_from(be_u32(bytes, 4)?)
        .map_err(|_| Error::Execution("MNIST image count overflow".into()))?;
    let rows = usize::try_from(be_u32(bytes, 8)?)
        .map_err(|_| Error::Execution("MNIST row count overflow".into()))?;
    let columns = usize::try_from(be_u32(bytes, 12)?)
        .map_err(|_| Error::Execution("MNIST column count overflow".into()))?;
    if rows != ROWS || columns != COLUMNS {
        return Err(Error::Execution(format!(
            "expected 28x28 MNIST images, got {rows}x{columns}"
        )));
    }
    let payload_len = count
        .checked_mul(rows * columns)
        .ok_or_else(|| Error::Execution("MNIST image payload overflow".into()))?;
    if bytes.len() != 16 + payload_len {
        return Err(Error::Execution("invalid MNIST image payload length".into()));
    }
    Ok((bytes[16..].to_vec(), count))
}

fn parse_labels(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 8 || be_u32(bytes, 0)? != LABEL_MAGIC {
        return Err(Error::Execution("invalid MNIST label IDX header".into()));
    }
    let count = usize::try_from(be_u32(bytes, 4)?)
        .map_err(|_| Error::Execution("MNIST label count overflow".into()))?;
    if bytes.len() != 8 + count || bytes[8..].iter().any(|label| *label > 9) {
        return Err(Error::Execution("invalid MNIST label payload".into()));
    }
    Ok(bytes[8..].to_vec())
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
        .map(u32::from_be_bytes)
        .ok_or_else(|| Error::Execution("truncated MNIST IDX header".into()))
}

fn deterministic_shuffle(values: &mut [usize]) {
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    for index in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let swap_with = usize::try_from(state % (index as u64 + 1)).unwrap_or(0);
        values.swap(index, swap_with);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_batches_idx_data() {
        let mut images = Vec::new();
        images.extend_from_slice(&IMAGE_MAGIC.to_be_bytes());
        images.extend_from_slice(&2_u32.to_be_bytes());
        images.extend_from_slice(&28_u32.to_be_bytes());
        images.extend_from_slice(&28_u32.to_be_bytes());
        images.extend(vec![0_u8; 28 * 28]);
        images.extend(vec![255_u8; 28 * 28]);
        let (images, count) = parse_images(&images).unwrap();
        assert_eq!(count, 2);

        let mut labels = Vec::new();
        labels.extend_from_slice(&LABEL_MAGIC.to_be_bytes());
        labels.extend_from_slice(&2_u32.to_be_bytes());
        labels.extend_from_slice(&[3, 7]);
        let labels = parse_labels(&labels).unwrap();
        let mut batches = MnistBatches::new(&images, &labels, 2, false).unwrap();
        let (batch, targets) = batches.next().unwrap().unwrap();
        assert_eq!(batch.shape(), &[2, 1, 28, 28]);
        assert_eq!(targets.to_vec().unwrap(), vec![3.0, 7.0]);
    }
}

