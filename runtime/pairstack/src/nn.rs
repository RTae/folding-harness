//! The primitives the four triangle operations are built from: layer norm,
//! a linear projection, sigmoid, and softmax over a chosen axis.
//!
//! Reductions accumulate in f64 and round to f32 once. That keeps the
//! deviation measured against the fixtures dominated by the oracle's own
//! rounding rather than by this code's summation order.
//!
//! Each primitive is validated in tests against the dumped fixture tensors
//! that exercise it, before any operation is built on it.

use crate::tensor::Tensor;
use crate::Error;

/// Normalises over the last axis, then scales and shifts by the learned
/// parameters. This is `nn.LayerNorm` with `eps = 1e-5` and the biased
/// variance, matching foundry's use.
pub fn layer_norm(
    x: &Tensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<Tensor, Error> {
    let Some(&width) = x.shape().last() else {
        return Err(Error::Mismatch {
            context: "layer_norm".to_string(),
            expected: "a tensor with at least one axis".to_string(),
            got: "an empty shape".to_string(),
        });
    };
    if weight.len() != width {
        return Err(Error::Mismatch {
            context: "layer_norm weight".to_string(),
            expected: format!("{width} values for the last axis"),
            got: format!("{} values", weight.len()),
        });
    }
    if let Some(b) = bias {
        if b.len() != width {
            return Err(Error::Mismatch {
                context: "layer_norm bias".to_string(),
                expected: format!("{width} values for the last axis"),
                got: format!("{} values", b.len()),
            });
        }
    }

    let mut data = vec![0.0f32; x.len()];
    for (row, dst) in x
        .data()
        .chunks_exact(width)
        .zip(data.chunks_exact_mut(width))
    {
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / width as f64;
        let var = row
            .iter()
            .map(|v| {
                let d = f64::from(*v) - mean;
                d * d
            })
            .sum::<f64>()
            / width as f64;
        let denom = (var + f64::from(eps)).sqrt() as f32;
        for (c, v) in row.iter().enumerate() {
            let mut y = (*v - mean as f32) / denom * weight[c];
            if let Some(b) = bias {
                y += b[c];
            }
            dst[c] = y;
        }
    }
    Ok(Tensor::internal(x.shape().to_vec(), data))
}

/// Applies `y = x @ weight^T + bias`, the arithmetic of `nn.Linear`. The
/// weight is stored row-major with shape `(out_features, in_features)`, the
/// layout the fixture weight files use. The bias is optional, because foundry
/// builds several of these projections with `bias=False`.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&[f32]>) -> Result<Tensor, Error> {
    let ws = weight.shape();
    if ws.len() != 2 {
        return Err(Error::Mismatch {
            context: "linear weight".to_string(),
            expected: "a matrix with shape (out_features, in_features)".to_string(),
            got: format!("shape {ws:?}"),
        });
    }
    let (out_features, in_features) = (ws[0], ws[1]);
    let Some(&x_last) = x.shape().last() else {
        return Err(Error::Mismatch {
            context: "linear input".to_string(),
            expected: "a tensor with at least one axis".to_string(),
            got: "an empty shape".to_string(),
        });
    };
    if x_last != in_features {
        return Err(Error::Mismatch {
            context: "linear input".to_string(),
            expected: format!("a last axis of {in_features} to match the weight"),
            got: format!("{x_last}"),
        });
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(Error::Mismatch {
                context: "linear bias".to_string(),
                expected: format!("{out_features} values"),
                got: format!("{} values", b.len()),
            });
        }
    }

    let rows = x.len() / in_features;
    let mut data = vec![0.0f32; rows * out_features];
    for (r, row) in x.data().chunks_exact(in_features).enumerate() {
        let dst = &mut data[r * out_features..(r + 1) * out_features];
        if let Some(b) = bias {
            dst.copy_from_slice(b);
        }
        for (o, slot) in dst.iter_mut().enumerate() {
            let w_row = &weight.data()[o * in_features..(o + 1) * in_features];
            let dot: f64 = w_row
                .iter()
                .zip(row)
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            *slot += dot as f32;
        }
    }

    let mut shape = x.shape().to_vec();
    let last = shape.len() - 1;
    shape[last] = out_features;
    Ok(Tensor::internal(shape, data))
}

/// The logistic function, elementwise.
#[must_use]
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Normalises exponentials along one axis, matching `F.softmax(x, dim=axis)`.
/// Subtracts the axis maximum before exponentiating, as PyTorch does.
#[must_use]
pub fn softmax(x: &Tensor, axis: usize) -> Tensor {
    let shape = x.shape();
    assert!(
        axis < shape.len(),
        "softmax axis {axis} outside {:?}",
        shape
    );
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut data = x.data().to_vec();
    for o in 0..outer {
        for i in 0..inner {
            let base = o * n * inner + i;
            let line_stride = inner;
            let mut max = f32::NEG_INFINITY;
            for t in 0..n {
                max = max.max(data[base + t * line_stride]);
            }
            let mut sum = 0.0f64;
            for t in 0..n {
                let e = (data[base + t * line_stride] - max).exp();
                data[base + t * line_stride] = e;
                sum += f64::from(e);
            }
            let inv = 1.0 / sum as f32;
            for t in 0..n {
                data[base + t * line_stride] *= inv;
            }
        }
    }
    Tensor::internal(shape.to_vec(), data)
}

#[cfg(test)]
mod tests {
    use super::{layer_norm, linear, sigmoid, softmax};
    use crate::check::{assert_close, Deviation, Tolerance, TRIANGLE_ATTN, TRIANGLE_MUL};
    use crate::fixtures::Fixtures;
    use crate::tensor::Tensor;

    /// The layer-norm epsilon every foundry triangle operation uses.
    const EPS: f32 = 1e-5;

    fn param(fx: &Fixtures, op: &str, key: &str) -> Vec<f32> {
        fx.weight(op, key).data().to_vec()
    }

    /// One `check` per dumped tensor. The closure returns the deviation so a
    /// test can also assert on the margin.
    struct Stage<'a> {
        fx: &'a Fixtures,
    }

    impl<'a> Stage<'a> {
        fn check(&self, op: &str, name: &str, got: &Tensor, tol: Tolerance) -> Deviation {
            assert_close(
                &format!("{op}.{name}"),
                got,
                &self.fx.tensor(&format!("{op}.{name}")),
                tol,
            )
        }
    }

    #[test]
    fn layer_norm_reproduces_every_dumped_normalisation() {
        let fx = Fixtures::load();
        let stage = Stage { fx: &fx };
        for op in [
            "tri_mul_outgoing",
            "tri_mul_incoming",
            "tri_attn_start",
            "tri_attn_end",
        ] {
            let tol = if op.starts_with("tri_mul") {
                TRIANGLE_MUL
            } else {
                TRIANGLE_ATTN
            };
            let input = fx.tensor(&format!("{op}.input"));
            let (wk, bk) = if op.starts_with("tri_mul") {
                ("norm_in.weight", "norm_in.bias")
            } else {
                ("norm.weight", "norm.bias")
            };
            let got =
                layer_norm(&input, &param(&fx, op, wk), Some(&param(&fx, op, bk)), EPS).unwrap();
            let d = stage.check(op, "norm_pair", &got, tol);
            assert!(
                d.max_excess < tol.atol,
                "{op}.norm_pair max_abs {}",
                d.max_abs
            );
        }
        for op in ["tri_mul_outgoing", "tri_mul_incoming"] {
            let raw = fx.tensor(&format!("{op}.einsum_raw"));
            let got = layer_norm(
                &raw,
                &param(&fx, op, "norm_out.weight"),
                Some(&param(&fx, op, "norm_out.bias")),
                EPS,
            )
            .unwrap();
            stage.check(op, "norm_out_applied", &got, TRIANGLE_MUL);
        }
    }

    #[test]
    fn linear_reproduces_the_triangle_mul_projections() {
        let fx = Fixtures::load();
        let stage = Stage { fx: &fx };
        for op in ["tri_mul_outgoing", "tri_mul_incoming"] {
            let norm_pair = fx.tensor(&format!("{op}.norm_pair"));
            let p = linear(&norm_pair, &fx.weight(op, "p_in.weight"), None).unwrap();
            stage.check(op, "p_combined", &p, TRIANGLE_MUL);
            let g = linear(&norm_pair, &fx.weight(op, "g_in.weight"), None).unwrap();
            stage.check(op, "g_combined", &g, TRIANGLE_MUL);

            let raw = fx.tensor(&format!("{op}.norm_out_applied"));
            let proj = linear(&raw, &fx.weight(op, "p_out.weight"), None).unwrap();
            stage.check(op, "out_proj", &proj, TRIANGLE_MUL);
            let logit = linear(&norm_pair, &fx.weight(op, "g_out.weight"), None).unwrap();
            stage.check(op, "out_gate_logit", &logit, TRIANGLE_MUL);
        }
    }

    #[test]
    fn linear_reproduces_the_triangle_attention_projections() {
        let fx = Fixtures::load();
        let stage = Stage { fx: &fx };
        for op in ["tri_attn_start", "tri_attn_end"] {
            let norm_pair = fx.tensor(&format!("{op}.norm_pair"));
            let bias = linear(&norm_pair, &fx.weight(op, "to_b.weight"), None).unwrap();
            stage.check(op, "bias", &bias, TRIANGLE_ATTN);

            let qkv = fx.tensor(&format!("{op}.qkv_pair"));
            let q = linear(&qkv, &fx.weight(op, "to_q.weight"), None).unwrap();
            stage.check(op, "query_proj", &q, TRIANGLE_ATTN);
            let k = linear(&qkv, &fx.weight(op, "to_k.weight"), None).unwrap();
            stage.check(op, "key_proj", &k, TRIANGLE_ATTN);
            let v = linear(&qkv, &fx.weight(op, "to_v.weight"), None).unwrap();
            stage.check(op, "value_proj", &v, TRIANGLE_ATTN);
            let g = linear(
                &qkv,
                &fx.weight(op, "to_g.weight"),
                Some(&param(&fx, op, "to_g.bias")),
            )
            .unwrap();
            stage.check(op, "gate_logit", &g, TRIANGLE_ATTN);

            let gated_out = fx.tensor(&format!("{op}.gated_out"));
            let out = linear(
                &gated_out,
                &fx.weight(op, "to_out.weight"),
                Some(&param(&fx, op, "to_out.bias")),
            )
            .unwrap();
            stage.check(op, "output", &out, TRIANGLE_ATTN);
        }
    }

    #[test]
    fn sigmoid_reproduces_every_dumped_gate() {
        let fx = Fixtures::load();
        let stage = Stage { fx: &fx };
        for op in ["tri_mul_outgoing", "tri_mul_incoming"] {
            let g = fx.tensor(&format!("{op}.g_combined"));
            let (low, high) = g.split_last_half().unwrap();
            let left = sigmoid(&low);
            stage.check(op, "left_gate", &left, TRIANGLE_MUL);
            let right = sigmoid(&high);
            stage.check(op, "right_gate", &right, TRIANGLE_MUL);
            let logit = fx.tensor(&format!("{op}.out_gate_logit"));
            let gate = sigmoid(&logit);
            stage.check(op, "out_gate", &gate, TRIANGLE_MUL);
        }
        for op in ["tri_attn_start", "tri_attn_end"] {
            let logit = fx.tensor(&format!("{op}.gate_logit"));
            let gate = sigmoid(&logit);
            stage.check(op, "gate", &gate, TRIANGLE_ATTN);
        }
    }

    #[test]
    fn softmax_over_the_key_axis_reproduces_the_attention_weights() {
        let fx = Fixtures::load();
        let stage = Stage { fx: &fx };
        for op in ["tri_attn_start", "tri_attn_end"] {
            // attn_with_bias is (B, i, j, k, h). foundry softmaxes over dim
            // -2, which is the k axis at index 3.
            let logits = fx.tensor(&format!("{op}.attn_with_bias"));
            let weights = softmax(&logits, 3);
            stage.check(op, "attn_weights", &weights, TRIANGLE_ATTN);
        }
    }
}
