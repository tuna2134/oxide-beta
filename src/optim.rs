//! Optimizers for trainable modules.

use crate::nn::{Parameter, Trainable};
use crate::{Error, Result};
use std::collections::HashMap;

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

/// AdamW with bias correction and decoupled weight decay.
#[derive(Debug)]
pub struct AdamW {
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
    states: HashMap<u64, State>,
}

impl AdamW {
    /// Creates an AdamW optimizer.
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
