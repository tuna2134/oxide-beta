use oxide_torch::{Error, Result};

#[derive(Clone, Debug)]
pub(super) struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: usize,
    pub output: usize,
}

impl Linear {
    pub fn apply(&self, values: &[f32], rows: usize) -> Result<Vec<f32>> {
        if values.len() != rows * self.input
            || self.weight.len() != self.output * self.input
            || self.bias.len() != self.output
        {
            return Err(Error::InvalidShape("invalid BERT linear operands".into()));
        }
        let mut result = vec![0.0; rows * self.output];
        for row in 0..rows {
            for output in 0..self.output {
                let mut sum = self.bias[output];
                for input in 0..self.input {
                    sum +=
                        values[row * self.input + input] * self.weight[output * self.input + input];
                }
                result[row * self.output + output] = sum;
            }
        }
        Ok(result)
    }
}

#[allow(clippy::cast_precision_loss)]
pub(super) fn layer_norm(
    values: &mut [f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) {
    for row in 0..rows {
        let slice = &mut values[row * hidden..(row + 1) * hidden];
        let mean = slice.iter().sum::<f32>() / hidden as f32;
        let variance = slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / hidden as f32;
        let scale = (variance + epsilon).sqrt().recip();
        for index in 0..hidden {
            slice[index] = (slice[index] - mean) * scale * weight[index] + bias[index];
        }
    }
}

pub(super) fn gelu(value: f32) -> f32 {
    let inner = (2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value.powi(3));
    0.5 * value * (1.0 + inner.tanh())
}
