#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::if_not_else,
    clippy::unused_self,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use crate::tensor::Op;
use crate::{CustomInput, Error, Result, Tensor};

trait CpuPrimitive {
    fn forward(&self, inputs: &[CustomInput<'_>]) -> Result<Vec<f32>>;

    fn backward(
        &self,
        inputs: &[CustomInput<'_>],
        output_gradient: &[f32],
    ) -> Result<Vec<Option<Vec<f32>>>>;
}

/// Applies a row-major affine projection using `[output, input]` weights.
///
/// # Errors
///
/// Returns an error for incompatible shapes or devices.
pub fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor, output: usize) -> Result<Tensor> {
    let mut shape = x.shape().to_vec();
    let input = *shape
        .last()
        .ok_or_else(|| Error::InvalidShape("linear requires rank >= 1".into()))?;
    if weight.shape() != [output, input] || bias.shape() != [output] {
        return Err(Error::InvalidShape(
            "linear parameter shape mismatch".into(),
        ));
    }
    *shape
        .last_mut()
        .ok_or_else(|| Error::InvalidShape("linear requires rank >= 1".into()))? = output;
    if x.device() != weight.device() || x.device() != bias.device() {
        return Err(Error::DeviceMismatch);
    }
    Ok(Tensor::new(
        shape,
        x.device(),
        Op::Linear {
            input: x.clone(),
            weight: weight.clone(),
            bias: bias.clone(),
        },
    ))
}

#[derive(Debug)]
struct LinearOp;
impl CpuPrimitive for LinearOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        let input = i[0].shape[i[0].shape.len() - 1];
        let output = i[1].shape[0];
        let rows = i[0].values.len() / input;
        let mut y = vec![0.; rows * output];
        for r in 0..rows {
            for o in 0..output {
                let mut s = i[2].values[o];
                for k in 0..input {
                    s += i[0].values[r * input + k] * i[1].values[o * input + k];
                }
                y[r * output + o] = s;
            }
        }
        Ok(y)
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let n = i[0].shape[i[0].shape.len() - 1];
        let o = i[1].shape[0];
        let rows = i[0].values.len() / n;
        let (mut dx, mut dw, mut db) = (
            vec![0.; i[0].values.len()],
            vec![0.; i[1].values.len()],
            vec![0.; o],
        );
        for r in 0..rows {
            for j in 0..o {
                let q = g[r * o + j];
                db[j] += q;
                for k in 0..n {
                    dx[r * n + k] += q * i[1].values[j * n + k];
                    dw[j * n + k] += q * i[0].values[r * n + k];
                }
            }
        }
        Ok(vec![Some(dx), Some(dw), Some(db)])
    }
}

/// Applies the tanh approximation of GELU elementwise.
///
/// # Errors
///
/// Returns an error if graph construction fails.
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(
        x.shape().to_vec(),
        x.device(),
        Op::Gelu(x.clone()),
    ))
}
#[derive(Debug)]
struct GeluOp;
impl CpuPrimitive for GeluOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        Ok(i[0].values.iter().map(|&x| gelu_value(x)).collect())
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let dx = i[0]
            .values
            .iter()
            .zip(g)
            .map(|(&x, &q)| {
                let c = (2. / std::f32::consts::PI).sqrt();
                let u = c * (x + 0.044715 * x * x * x);
                q * 0.5
                    * ((1. + u.tanh())
                        + x * (1. - u.tanh().powi(2)) * c * (1. + 3. * 0.044715 * x * x))
            })
            .collect();
        Ok(vec![Some(dx)])
    }
}
fn gelu_value(x: f32) -> f32 {
    0.5 * x * (1. + ((2. / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}
/// Applies hyperbolic tangent elementwise.
///
/// # Errors
///
/// Returns an error if graph construction fails.
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    Ok(Tensor::new(
        x.shape().to_vec(),
        x.device(),
        Op::Tanh(x.clone()),
    ))
}
#[derive(Debug)]
struct TanhOp;
impl CpuPrimitive for TanhOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        Ok(i[0].values.iter().map(|x| x.tanh()).collect())
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(vec![Some(
            i[0].values
                .iter()
                .zip(g)
                .map(|(x, q)| q * (1. - x.tanh().powi(2)))
                .collect(),
        )])
    }
}

/// Gathers embedding rows for integral token IDs stored in a tensor.
///
/// # Errors
///
/// Returns an error for an invalid embedding-table shape or device mismatch.
pub fn embedding(ids: &Tensor, weight: &Tensor, hidden: usize) -> Result<Tensor> {
    if weight.shape().len() != 2 || weight.shape()[1] != hidden {
        return Err(Error::InvalidShape(
            "embedding weight must have shape [vocabulary, hidden]".into(),
        ));
    }
    let mut shape = ids.shape().to_vec();
    shape.push(hidden);
    if ids.device() != weight.device() {
        return Err(Error::DeviceMismatch);
    }
    Ok(Tensor::new(
        shape,
        ids.device(),
        Op::Embedding {
            ids: ids.clone(),
            weight: weight.clone(),
        },
    ))
}
#[derive(Debug)]
struct EmbeddingOp;
impl CpuPrimitive for EmbeddingOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        let h = i[1].shape[1];
        let mut y = Vec::with_capacity(i[0].values.len() * h);
        for &id in i[0].values {
            let id = index(id, i[1].shape[0])?;
            y.extend_from_slice(&i[1].values[id * h..(id + 1) * h]);
        }
        Ok(y)
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let h = i[1].shape[1];
        let mut dw = vec![0.; i[1].values.len()];
        for (p, &id) in i[0].values.iter().enumerate() {
            let id = index(id, i[1].shape[0])?;
            for k in 0..h {
                dw[id * h + k] += g[p * h + k];
            }
        }
        Ok(vec![None, Some(dw)])
    }
}
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn index(x: f32, n: usize) -> Result<usize> {
    if !x.is_finite() || x < 0. || x.fract() != 0. || x as usize >= n {
        Err(Error::InvalidShape("embedding id out of range".into()))
    } else {
        Ok(x as usize)
    }
}

/// Normalizes the final dimension and applies an affine transform.
///
/// # Errors
///
/// Returns an error for incompatible parameter shapes or devices.
pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden = x
        .shape()
        .last()
        .copied()
        .ok_or_else(|| Error::InvalidShape("layer_norm requires rank >= 1".into()))?;
    if w.shape() != [hidden] || b.shape() != [hidden] {
        return Err(Error::InvalidShape(
            "layer_norm parameters must match the final dimension".into(),
        ));
    }
    if x.device() != w.device() || x.device() != b.device() {
        return Err(Error::DeviceMismatch);
    }
    Ok(Tensor::new(
        x.shape().to_vec(),
        x.device(),
        Op::LayerNorm {
            input: x.clone(),
            weight: w.clone(),
            bias: b.clone(),
            epsilon: eps,
        },
    ))
}
#[derive(Debug)]
struct LayerNormOp(f32);
impl CpuPrimitive for LayerNormOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        let n = i[1].values.len();
        let mut y = vec![0.; i[0].values.len()];
        for (r, row) in i[0].values.chunks_exact(n).enumerate() {
            let m = row.iter().sum::<f32>() / n as f32;
            let v = row.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
            let z = (v + self.0).sqrt().recip();
            for k in 0..n {
                y[r * n + k] = (row[k] - m) * z * i[1].values[k] + i[2].values[k];
            }
        }
        Ok(y)
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let n = i[1].values.len();
        let mut dx = vec![0.; i[0].values.len()];
        let (mut dw, mut db) = (vec![0.; n], vec![0.; n]);
        for (r, row) in i[0].values.chunks_exact(n).enumerate() {
            let m = row.iter().sum::<f32>() / n as f32;
            let v = row.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
            let inv = (v + self.0).sqrt().recip();
            let mut s1 = 0.;
            let mut s2 = 0.;
            for k in 0..n {
                let xh = (row[k] - m) * inv;
                let q = g[r * n + k] * i[1].values[k];
                s1 += q;
                s2 += q * xh;
                dw[k] += g[r * n + k] * xh;
                db[k] += g[r * n + k];
            }
            for k in 0..n {
                let xh = (row[k] - m) * inv;
                let q = g[r * n + k] * i[1].values[k];
                dx[r * n + k] = inv * (q - (s1 + xh * s2) / n as f32);
            }
        }
        Ok(vec![Some(dx), Some(dw), Some(db)])
    }
}

/// Selects the first sequence element from a `[batch, sequence, hidden]` tensor.
///
/// # Errors
///
/// Returns an error unless the input has the expected rank and hidden size.
pub fn select_first(x: &Tensor, hidden: usize) -> Result<Tensor> {
    if x.shape().len() != 3 || x.shape()[2] != hidden || x.shape()[1] == 0 {
        return Err(Error::InvalidShape(
            "select_first expects [batch, non-empty sequence, hidden]".into(),
        ));
    }
    let batch = x.shape()[0];
    Ok(Tensor::new(
        vec![batch, hidden],
        x.device(),
        Op::SelectFirst(x.clone()),
    ))
}
#[derive(Debug)]
struct ClsOp;
impl CpuPrimitive for ClsOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        let (b, s, h) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let mut y = vec![0.; b * h];
        for n in 0..b {
            y[n * h..(n + 1) * h].copy_from_slice(&i[0].values[n * s * h..n * s * h + h]);
        }
        Ok(y)
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let (b, s, h) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let mut dx = vec![0.; i[0].values.len()];
        for n in 0..b {
            dx[n * s * h..n * s * h + h].copy_from_slice(&g[n * h..(n + 1) * h]);
        }
        Ok(vec![Some(dx)])
    }
}

/// Applies multi-head self-attention with learned Q/K/V projections.
///
/// # Errors
///
/// Returns an error for invalid tensor, projection, mask, or head shapes.
pub fn scaled_dot_product_attention(
    x: &Tensor,
    mask: &Tensor,
    qw: &Tensor,
    qb: &Tensor,
    kw: &Tensor,
    kb: &Tensor,
    vw: &Tensor,
    vb: &Tensor,
    heads: usize,
) -> Result<Tensor> {
    if x.shape().len() != 3 {
        return Err(Error::InvalidShape(
            "attention input must be [batch, sequence, hidden]".into(),
        ));
    }
    let (batch, sequence, hidden) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    if heads == 0 || hidden % heads != 0 || mask.shape() != [batch, sequence] {
        return Err(Error::InvalidShape(
            "attention mask or head configuration is invalid".into(),
        ));
    }
    if [qw, kw, vw]
        .iter()
        .any(|weight| weight.shape() != [hidden, hidden])
        || [qb, kb, vb].iter().any(|bias| bias.shape() != [hidden])
    {
        return Err(Error::InvalidShape(
            "attention projections must use [hidden, hidden] weights and [hidden] biases".into(),
        ));
    }
    if [mask, qw, qb, kw, kb, vw, vb]
        .iter()
        .any(|tensor| tensor.device() != x.device())
    {
        return Err(Error::DeviceMismatch);
    }
    Ok(Tensor::new(
        x.shape().to_vec(),
        x.device(),
        Op::ScaledDotProductAttention {
            input: x.clone(),
            mask: mask.clone(),
            query_weight: qw.clone(),
            query_bias: qb.clone(),
            key_weight: kw.clone(),
            key_bias: kb.clone(),
            value_weight: vw.clone(),
            value_bias: vb.clone(),
            heads,
        },
    ))
}
#[derive(Debug)]
struct AttentionOp {
    heads: usize,
}
impl AttentionOp {
    fn projections(&self, i: &[CustomInput<'_>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = i[0].values;
        let h = i[0].shape[2];
        let rows = x.len() / h;
        let apply = |w: &[f32], b: &[f32]| {
            let mut y = vec![0.; rows * h];
            for r in 0..rows {
                for o in 0..h {
                    let mut z = b[o];
                    for k in 0..h {
                        z += x[r * h + k] * w[o * h + k];
                    }
                    y[r * h + o] = z;
                }
            }
            y
        };
        (
            apply(i[2].values, i[3].values),
            apply(i[4].values, i[5].values),
            apply(i[6].values, i[7].values),
        )
    }
    fn probabilities(&self, i: &[CustomInput<'_>], q: &[f32], k: &[f32]) -> Vec<f32> {
        let (b, s, h) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let d = h / self.heads;
        let scale = (d as f32).sqrt().recip();
        let mut p = vec![0.; b * self.heads * s * s];
        for n in 0..b {
            for a in 0..self.heads {
                for u in 0..s {
                    let base = ((n * self.heads + a) * s + u) * s;
                    let mut mx = f32::NEG_INFINITY;
                    for v in 0..s {
                        if i[1].values[n * s + v] != 0. {
                            let mut z = 0.;
                            for j in 0..d {
                                z +=
                                    q[(n * s + u) * h + a * d + j] * k[(n * s + v) * h + a * d + j];
                            }
                            p[base + v] = z * scale;
                            mx = mx.max(p[base + v]);
                        } else {
                            p[base + v] = f32::NEG_INFINITY;
                        }
                    }
                    let mut sum = 0.;
                    for v in 0..s {
                        p[base + v] = (p[base + v] - mx).exp();
                        sum += p[base + v];
                    }
                    for v in 0..s {
                        p[base + v] /= sum;
                    }
                }
            }
        }
        p
    }
}
impl CpuPrimitive for AttentionOp {
    fn forward(&self, i: &[CustomInput<'_>]) -> Result<Vec<f32>> {
        let (q, k, v) = self.projections(i);
        let p = self.probabilities(i, &q, &k);
        let (b, s, h) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let d = h / self.heads;
        let mut y = vec![0.; b * s * h];
        for n in 0..b {
            for a in 0..self.heads {
                for u in 0..s {
                    for vpos in 0..s {
                        let prob = p[((n * self.heads + a) * s + u) * s + vpos];
                        for j in 0..d {
                            y[(n * s + u) * h + a * d + j] +=
                                prob * v[(n * s + vpos) * h + a * d + j];
                        }
                    }
                }
            }
        }
        Ok(y)
    }
    fn backward(&self, i: &[CustomInput<'_>], g: &[f32]) -> Result<Vec<Option<Vec<f32>>>> {
        let (q, k, v) = self.projections(i);
        let p = self.probabilities(i, &q, &k);
        let (b, s, h) = (i[0].shape[0], i[0].shape[1], i[0].shape[2]);
        let d = h / self.heads;
        let scale = (d as f32).sqrt().recip();
        let (mut dq, mut dk, mut dv) = (vec![0.; q.len()], vec![0.; k.len()], vec![0.; v.len()]);
        for n in 0..b {
            for a in 0..self.heads {
                for u in 0..s {
                    let base = ((n * self.heads + a) * s + u) * s;
                    let mut dp = vec![0.; s];
                    for z in 0..s {
                        for j in 0..d {
                            dp[z] +=
                                g[(n * s + u) * h + a * d + j] * v[(n * s + z) * h + a * d + j];
                            dv[(n * s + z) * h + a * d + j] +=
                                p[base + z] * g[(n * s + u) * h + a * d + j];
                        }
                    }
                    let dot = (0..s).map(|z| dp[z] * p[base + z]).sum::<f32>();
                    for z in 0..s {
                        let ds = p[base + z] * (dp[z] - dot) * scale;
                        for j in 0..d {
                            dq[(n * s + u) * h + a * d + j] += ds * k[(n * s + z) * h + a * d + j];
                            dk[(n * s + z) * h + a * d + j] += ds * q[(n * s + u) * h + a * d + j];
                        }
                    }
                }
            }
        }
        let mut grads = vec![None; 8];
        let mut dx = vec![0.; i[0].values.len()];
        for (slot, proj) in [(2, dq), (4, dk), (6, dv)] {
            let mut dw = vec![0.; i[slot].values.len()];
            let mut db = vec![0.; h];
            for r in 0..b * s {
                for o in 0..h {
                    let z = proj[r * h + o];
                    db[o] += z;
                    for j in 0..h {
                        dx[r * h + j] += z * i[slot].values[o * h + j];
                        dw[o * h + j] += z * i[0].values[r * h + j];
                    }
                }
            }
            grads[slot] = Some(dw);
            grads[slot + 1] = Some(db);
        }
        grads[0] = Some(dx);
        Ok(grads)
    }
}

/// Compact execution descriptor used after the public graph has already been
/// lowered from concrete [`Op`] variants.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Primitive {
    Linear,
    Gelu,
    Tanh,
    Embedding,
    LayerNorm { epsilon: f32 },
    SelectFirst,
    ScaledDotProductAttention { heads: usize },
}

pub(crate) fn forward(primitive: Primitive, inputs: &[CustomInput<'_>]) -> Result<Vec<f32>> {
    match primitive {
        Primitive::Linear => LinearOp.forward(inputs),
        Primitive::Gelu => GeluOp.forward(inputs),
        Primitive::Tanh => TanhOp.forward(inputs),
        Primitive::Embedding => EmbeddingOp.forward(inputs),
        Primitive::LayerNorm { epsilon } => LayerNormOp(epsilon).forward(inputs),
        Primitive::SelectFirst => ClsOp.forward(inputs),
        Primitive::ScaledDotProductAttention { heads } => AttentionOp { heads }.forward(inputs),
    }
}

pub(crate) fn backward(
    primitive: Primitive,
    inputs: &[CustomInput<'_>],
    output_gradient: &[f32],
) -> Result<Vec<Option<Vec<f32>>>> {
    match primitive {
        Primitive::Linear => LinearOp.backward(inputs, output_gradient),
        Primitive::Gelu => GeluOp.backward(inputs, output_gradient),
        Primitive::Tanh => TanhOp.backward(inputs, output_gradient),
        Primitive::Embedding => EmbeddingOp.backward(inputs, output_gradient),
        Primitive::LayerNorm { epsilon } => LayerNormOp(epsilon).backward(inputs, output_gradient),
        Primitive::SelectFirst => ClsOp.backward(inputs, output_gradient),
        Primitive::ScaledDotProductAttention { heads } => {
            AttentionOp { heads }.backward(inputs, output_gradient)
        }
    }
}
