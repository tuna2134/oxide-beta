//! Optimizers for trainable modules.

use crate::nn::{Parameter, Trainable};
use crate::{Error, Result};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
static NEXT_OPTIMIZER_ID: AtomicU64 = AtomicU64::new(1);

pub trait Optimizer {
    /// Clears gradients on every model parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter gradient lock is poisoned.
    fn zero_grad(&self, model: &dyn Trainable) -> Result<()>;

    /// Applies one optimizer update.
    ///
    /// # Errors
    ///
    /// Returns an error if gradients or parameter data cannot be read or replaced.
    fn step(&mut self, model: &mut dyn Trainable) -> Result<()>;
}

#[derive(Debug)]
struct State {
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
}

/// `AdamW` with bias correction and decoupled weight decay.
#[derive(Debug)]
pub struct AdamW {
    #[cfg(feature = "cuda")]
    id: u64,
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
    states: HashMap<u64, State>,
}

impl AdamW {
    /// Creates an `AdamW` optimizer.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-positive/non-finite learning rate or negative decay.
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(Error::Execution(
                "AdamW learning rate must be positive and finite".into(),
            ));
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(Error::Execution(
                "AdamW weight decay must be non-negative and finite".into(),
            ));
        }
        Ok(Self {
            #[cfg(feature = "cuda")]
            id: NEXT_OPTIMIZER_ID.fetch_add(1, Ordering::Relaxed),
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 0,
            states: HashMap::new(),
        })
    }

    fn update_parameter(&mut self, parameter: &mut Parameter) -> Result<()> {
        #[cfg(feature = "cuda")]
        if matches!(parameter.value().device(), crate::Device::Cuda(_)) {
            let step_i32 = i32::try_from(self.step)
                .map_err(|_| Error::Execution("AdamW step counter exceeded i32".into()))?;
            return crate::cuda::adamw_step(
                parameter,
                self.id,
                crate::cuda::AdamWHyperparameters {
                    learning_rate: self.learning_rate,
                    weight_decay: self.weight_decay,
                    beta1: self.beta1,
                    beta2: self.beta2,
                    first_correction: 1.0 - self.beta1.powi(step_i32),
                    second_correction: 1.0 - self.beta2.powi(step_i32),
                    epsilon: self.epsilon,
                },
            );
        }

        let Some(gradient) = parameter.value().grad()? else {
            return Ok(());
        };
        let mut data = parameter.value().to_vec()?;
        if data.len() != gradient.len() {
            return Err(Error::Execution("parameter gradient size mismatch".into()));
        }
        let state = self.states.entry(parameter.id()).or_insert_with(|| State {
            first_moment: vec![0.0; data.len()],
            second_moment: vec![0.0; data.len()],
        });
        let step_i32 = i32::try_from(self.step)
            .map_err(|_| Error::Execution("AdamW step counter exceeded i32".into()))?;
        let first_correction = 1.0 - self.beta1.powi(step_i32);
        let second_correction = 1.0 - self.beta2.powi(step_i32);
        for (((value, grad), first), second) in data
            .iter_mut()
            .zip(gradient)
            .zip(&mut state.first_moment)
            .zip(&mut state.second_moment)
        {
            *first = self.beta1 * *first + (1.0 - self.beta1) * grad;
            *second = self.beta2 * *second + (1.0 - self.beta2) * grad * grad;
            let normalized =
                (*first / first_correction) / ((*second / second_correction).sqrt() + self.epsilon);
            *value -= self.learning_rate * (normalized + self.weight_decay * *value);
        }
        parameter.replace_data(data)
    }
}

impl Optimizer for AdamW {
    fn zero_grad(&self, model: &dyn Trainable) -> Result<()> {
        let mut result = Ok(());
        model.visit_parameters(&mut |parameter| {
            if result.is_ok() {
                result = parameter.value().zero_grad();
            }
        });
        result
    }

    fn step(&mut self, model: &mut dyn Trainable) -> Result<()> {
        self.step = self
            .step
            .checked_add(1)
            .ok_or_else(|| Error::Execution("AdamW step counter overflow".into()))?;
        let mut result = Ok(());
        model.visit_parameters_mut(&mut |parameter| {
            if result.is_ok() {
                result = self.update_parameter(parameter);
            }
        });
        result
    }
}

#[cfg(feature = "cuda")]
impl Drop for AdamW {
    fn drop(&mut self) {
        crate::cuda::release_optimizer(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::nn::{Conv2d, Module};

    #[test]
    fn adamw_reduces_a_tiny_classification_loss() {
        let mut layer = Conv2d::new(1, 2, 1, 1, 1, crate::Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![-1.0, 1.0], vec![2, 1, 1, 1]).unwrap();
        let targets = Tensor::from_vec(vec![0.0, 1.0], vec![2]).unwrap();
        let initial = layer
            .forward(&input)
            .unwrap()
            .reshape(vec![2, 2])
            .unwrap()
            .cross_entropy(&targets)
            .unwrap()
            .item()
            .unwrap();
        let mut optimizer = AdamW::new(0.05, 0.0).unwrap();
        for _ in 0..10 {
            optimizer.zero_grad(&layer).unwrap();
            let loss = layer
                .forward(&input)
                .unwrap()
                .reshape(vec![2, 2])
                .unwrap()
                .cross_entropy(&targets)
                .unwrap();
            loss.backward().unwrap();
            optimizer.step(&mut layer).unwrap();
        }
        let final_loss = layer
            .forward(&input)
            .unwrap()
            .reshape(vec![2, 2])
            .unwrap()
            .cross_entropy(&targets)
            .unwrap()
            .item()
            .unwrap();
        assert!(
            final_loss < initial,
            "{final_loss} should be below {initial}"
        );
    }
}
