//! Triangle attention, starting-node and ending-node.
//!
//! Mirrors foundry `TriangleAttention.forward` and `_forward_vanilla` at
//! commit `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`,
//! `models/rf3/src/rf3/model/layers/attention.py:82-111` and `179-200`:
//!
//! ```text
//! pair = LayerNorm(pair)
//! bias = to_b(pair)                     # before any transpose
//! if ending: pair = transpose(pair)
//! gate  = sigmoid(to_g(pair))
//! q, k, v = to_q(pair), to_k(pair), to_v(pair)      # no bias terms
//! q = q * (1 / sqrt(d_head))
//! attn = einsum("bijhd,bikhd->bijkh", q, k) + bias
//! attn = softmax(attn, over k)
//! out  = gate * einsum("bijkh,bikhd->bijhd", attn, v)
//! if ending: out = transpose(out)
//! out  = to_out(out)
//! ```
//!
//! The load-bearing detail here is the first line of the ending-node path:
//! the bias is projected from the untransposed normalised pair, and the pair
//! is transposed only afterwards, for the query, key, value and gate. The
//! dumped fixtures distinguish `norm_pair` from `qkv_pair` for exactly this
//! reason, and a named test pins the bias orientation.

use crate::nn::{layer_norm, linear, sigmoid, softmax};
use crate::tensor::Tensor;
use crate::Error;

/// The layer-norm epsilon foundry uses everywhere in these operations.
const EPS: f32 = 1e-5;

/// Which node the attention updates. The two differ only in where the
/// transposes happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangleNode {
    /// No transpose. The query, key, value and gate read the normalised pair
    /// in its own orientation.
    Starting,
    /// The bias reads the untransposed normalised pair. The query, key, value
    /// and gate read its transpose, and the gated result is transposed back
    /// before the output projection.
    Ending,
}

/// The weights of one triangle attention, under foundry's `state_dict` names.
/// `to_q`, `to_k`, `to_v` and `to_b` have no bias term. `to_g` and `to_out`
/// carry both parameters, and the layer norm carries both.
#[derive(Debug, Clone)]
pub struct TriangleAttnWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    /// Shape `(n_head * d_head, d_pair)`, row-major.
    pub to_q_weight: Tensor,
    /// Shape `(n_head * d_head, d_pair)`, row-major.
    pub to_k_weight: Tensor,
    /// Shape `(n_head * d_head, d_pair)`, row-major.
    pub to_v_weight: Tensor,
    /// Shape `(n_head, d_pair)`, row-major.
    pub to_b_weight: Tensor,
    /// Shape `(n_head * d_head, d_pair)`, row-major.
    pub to_g_weight: Tensor,
    pub to_g_bias: Vec<f32>,
    /// Shape `(d_pair, n_head * d_head)`, row-major.
    pub to_out_weight: Tensor,
    pub to_out_bias: Vec<f32>,
    pub n_head: usize,
    pub d_head: usize,
}

/// Every intermediate the forward pass produces, under the names the oracle
/// fixtures use. Tests check each one against its dumped tensor, so a failure
/// names the stage that broke.
#[derive(Debug, Clone)]
pub struct TriangleAttnStages {
    pub norm_pair: Tensor,
    pub bias: Tensor,
    pub qkv_pair: Tensor,
    pub query_proj: Tensor,
    pub key_proj: Tensor,
    pub value_proj: Tensor,
    pub gate_logit: Tensor,
    pub gate: Tensor,
    pub query_scaled: Tensor,
    pub key_heads: Tensor,
    pub attn_logits: Tensor,
    pub attn_with_bias: Tensor,
    pub attn_weights: Tensor,
    pub value_heads: Tensor,
    pub attn_out_heads: Tensor,
    pub gated_out: Tensor,
    pub output: Tensor,
}

/// The query scale foundry applies: `1 / math.sqrt(d_hidden)` computed in f64
/// and rounded once to f32, matching the Python scalar conversion.
fn scaling(d_head: usize) -> f32 {
    (1.0 / (d_head as f64).sqrt()) as f32
}

/// Runs the forward pass and keeps every intermediate.
pub fn triangle_attn_stages(
    pair: &Tensor,
    w: &TriangleAttnWeights,
    node: TriangleNode,
) -> Result<TriangleAttnStages, Error> {
    let shape = pair.shape();
    if shape.len() != 4 {
        return Err(Error::Mismatch {
            context: "triangle_attn input".to_string(),
            expected: "shape (B, L, L, d_pair)".to_string(),
            got: format!("shape {shape:?}"),
        });
    }
    let (nb, l) = (shape[0], shape[1]);
    if nb == 0 {
        return Err(Error::Mismatch {
            context: "triangle_attn input".to_string(),
            expected: "a nonempty batch axis".to_string(),
            got: "shape with B = 0".to_string(),
        });
    }
    if shape[2] != l {
        return Err(Error::Mismatch {
            context: "triangle_attn input".to_string(),
            expected: "a square pair representation (B, L, L, d_pair)".to_string(),
            got: format!("shape {shape:?}"),
        });
    }
    let (h, dh) = (w.n_head, w.d_head);

    let norm_pair = layer_norm(pair, &w.norm_weight, Some(&w.norm_bias), EPS)?;
    // The bias is always projected from the untransposed normalised pair.
    let bias = linear(&norm_pair, &w.to_b_weight, None)?;
    let qkv_pair = match node {
        TriangleNode::Starting => norm_pair.clone(),
        TriangleNode::Ending => norm_pair.transpose_axes(1, 2),
    };

    let gate_logit = linear(&qkv_pair, &w.to_g_weight, Some(&w.to_g_bias))?;
    let gate = sigmoid(&gate_logit);
    let query_proj = linear(&qkv_pair, &w.to_q_weight, None)?;
    let key_proj = linear(&qkv_pair, &w.to_k_weight, None)?;
    let value_proj = linear(&qkv_pair, &w.to_v_weight, None)?;

    let head_shape = |t: &Tensor| -> Result<Tensor, Error> {
        let mut s = t.shape().to_vec();
        s.pop();
        s.push(h);
        s.push(dh);
        t.reshaped(s)
    };
    let query_scaled = head_shape(&query_proj)?.map(|v| v * scaling(dh));
    let key_heads = head_shape(&key_proj)?;
    let value_heads = head_shape(&value_proj)?;

    let attn_logits = qk_logits(&query_scaled, &key_heads)?;
    let attn_with_bias = add_bias(&attn_logits, &bias)?;
    // foundry softmaxes over dim -2, the k axis of (B, i, j, k, h).
    let attn_weights = softmax(&attn_with_bias, 3);
    let attn_out_heads = weighted_values(&attn_weights, &value_heads)?;

    let mut flat = attn_out_heads.shape().to_vec();
    flat.pop();
    flat.pop();
    flat.push(h * dh);
    let gated = gate.mul(&attn_out_heads.reshaped(flat)?)?;
    let gated_out = match node {
        TriangleNode::Starting => gated,
        TriangleNode::Ending => gated.transpose_axes(1, 2),
    };
    let output = linear(&gated_out, &w.to_out_weight, Some(&w.to_out_bias))?;

    Ok(TriangleAttnStages {
        norm_pair,
        bias,
        qkv_pair,
        query_proj,
        key_proj,
        value_proj,
        gate_logit,
        gate,
        query_scaled,
        key_heads,
        attn_logits,
        attn_with_bias,
        attn_weights,
        value_heads,
        attn_out_heads,
        gated_out,
        output,
    })
}

/// Runs the forward pass and returns the value a pairformer block adds to its
/// residual.
pub fn triangle_attn(
    pair: &Tensor,
    w: &TriangleAttnWeights,
    node: TriangleNode,
) -> Result<Tensor, Error> {
    Ok(triangle_attn_stages(pair, w, node)?.output)
}

/// `einsum("bijhd,bikhd->bijkh", query, key)`. Both operands are
/// `(B, L, L, h, d)` and the result is `(B, L, L, L, h)`. Accumulates in f64.
fn qk_logits(query: &Tensor, key: &Tensor) -> Result<Tensor, Error> {
    let (qs, ks) = (query.shape(), key.shape());
    if qs.len() != 5 || qs != ks || qs[1] != qs[2] {
        return Err(Error::Mismatch {
            context: "query-key contraction".to_string(),
            expected: "two equal square shapes (B, L, L, h, d)".to_string(),
            got: format!("{qs:?} and {ks:?}"),
        });
    }
    let (nb, l, h, d) = (qs[0], qs[1], qs[3], qs[4]);
    let q = query.data();
    let k = key.data();
    let mut acc = vec![0.0f64; nb * l * l * l * h];
    for b in 0..nb {
        for i in 0..l {
            for j in 0..l {
                for head in 0..h {
                    let out_base = (((b * l + i) * l + j) * l) * h;
                    let q_base = ((b * l + i) * l + j) * h * d + head * d;
                    let q_slot = &q[q_base..q_base + d];
                    for kk in 0..l {
                        let k_base = ((b * l + i) * l + kk) * h * d + head * d;
                        let k_slot = &k[k_base..k_base + d];
                        let dot: f64 = q_slot
                            .iter()
                            .zip(k_slot)
                            .map(|(qv, kv)| f64::from(*qv) * f64::from(*kv))
                            .sum();
                        acc[out_base + kk * h + head] += dot;
                    }
                }
            }
        }
    }
    let data: Vec<f32> = acc.iter().map(|v| *v as f32).collect();
    Ok(Tensor::internal(vec![nb, l, l, l, h], data))
}

/// Adds the bias to the logits, matching foundry's
/// `attn + bias.unsqueeze(1).expand(-1, L, -1, -1, -1)`. The logits are
/// `(B, i, j, k, h)`, the bias is `(B, j, k, h)`, and the bias keeps its
/// untransposed orientation for both node kinds.
fn add_bias(logits: &Tensor, bias: &Tensor) -> Result<Tensor, Error> {
    let ls = logits.shape();
    let bs = bias.shape();
    let ok = ls.len() == 5
        && bs.len() == 4
        && ls[1] == ls[2]
        && ls[2] == ls[3]
        && bs[0] == ls[0]
        && bs[1] == ls[2]
        && bs[2] == ls[3]
        && bs[3] == ls[4];
    if !ok {
        return Err(Error::Mismatch {
            context: "attention bias addition".to_string(),
            expected: format!(
                "a bias of shape (B, j, k, h) for logits {ls:?}, broadcast over the i axis"
            ),
            got: format!("bias {bs:?}"),
        });
    }
    let (l, h) = (ls[1], ls[4]);
    let bias_data = bias.data();
    let data: Vec<f32> = logits
        .data()
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            // idx decomposes row-major as (((b*l + i)*l + j)*l + k)*h + head.
            let head = idx % h;
            let k = (idx / h) % l;
            let j = (idx / (h * l)) % l;
            let b = idx / (h * l * l * l);
            v + bias_data[((b * l + j) * l + k) * h + head]
        })
        .collect();
    Ok(Tensor::internal(ls.to_vec(), data))
}

/// `einsum("bijkh,bikhd->bijhd", attn, value)`. The weights are
/// `(B, L, L, L, h)`, the values `(B, L, L, h, d)`, and the result
/// `(B, L, L, h, d)`. Accumulates in f64.
fn weighted_values(attn: &Tensor, value: &Tensor) -> Result<Tensor, Error> {
    let (attn_shape, vs) = (attn.shape(), value.shape());
    if attn_shape.len() != 5 || vs.len() != 5 || attn_shape[0] != vs[0] || attn_shape[1] != vs[1] {
        return Err(Error::Mismatch {
            context: "attention-value contraction".to_string(),
            expected: "weights (B, i, j, k, h) and values (B, i, k, h, d) over one batch"
                .to_string(),
            got: format!("{attn_shape:?} and {vs:?}"),
        });
    }
    let (nb, l, h) = (attn_shape[0], attn_shape[1], attn_shape[4]);
    if attn_shape[2] != attn_shape[1]
        || attn_shape[3] != attn_shape[1]
        || attn_shape[3] != vs[2]
        || attn_shape[4] != vs[3]
    {
        return Err(Error::Mismatch {
            context: "attention-value contraction".to_string(),
            expected: "square sequence axes and a k axis and head count the two operands share"
                .to_string(),
            got: format!("{attn_shape:?} and {vs:?}"),
        });
    }
    let d = vs[4];
    let a = attn.data();
    let v = value.data();
    let mut acc = vec![0.0f64; nb * l * l * h * d];
    for b in 0..nb {
        for i in 0..l {
            for j in 0..l {
                for head in 0..h {
                    let out_base = ((b * l + i) * l + j) * h * d + head * d;
                    for kk in 0..l {
                        let w = f64::from(a[(((b * l + i) * l + j) * l + kk) * h + head]);
                        let v_base = ((b * l + i) * l + kk) * h * d + head * d;
                        let out_slot = &mut acc[out_base..out_base + d];
                        for (slot, vv) in out_slot.iter_mut().zip(&v[v_base..v_base + d]) {
                            *slot += w * f64::from(*vv);
                        }
                    }
                }
            }
        }
    }
    let data: Vec<f32> = acc.iter().map(|v2| *v2 as f32).collect();
    Ok(Tensor::internal(vs.to_vec(), data))
}

#[cfg(test)]
mod tests {
    use super::{triangle_attn, triangle_attn_stages, TriangleNode};
    use crate::check::{assert_close, assert_far, Deviation, TRIANGLE_ATTN};
    use crate::fixtures::Fixtures;
    use crate::tensor::Tensor;

    /// Checks every dumped intermediate of one operation and returns the
    /// deviations, so the report test can print them.
    fn check_intermediates(
        fx: &Fixtures,
        op: &str,
        node: TriangleNode,
    ) -> Vec<(&'static str, Deviation)> {
        let w = fx.triangle_attn_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let s = triangle_attn_stages(&input, &w, node).unwrap();
        let cases: Vec<(&'static str, &Tensor)> = vec![
            ("norm_pair", &s.norm_pair),
            ("bias", &s.bias),
            ("qkv_pair", &s.qkv_pair),
            ("query_proj", &s.query_proj),
            ("key_proj", &s.key_proj),
            ("value_proj", &s.value_proj),
            ("gate_logit", &s.gate_logit),
            ("gate", &s.gate),
            ("query_scaled", &s.query_scaled),
            ("key_heads", &s.key_heads),
            ("attn_logits", &s.attn_logits),
            ("attn_with_bias", &s.attn_with_bias),
            ("attn_weights", &s.attn_weights),
            ("value_heads", &s.value_heads),
            ("attn_out_heads", &s.attn_out_heads),
            ("gated_out", &s.gated_out),
        ];
        let mut out = Vec::new();
        for (name, got) in cases {
            let d = assert_close(
                &format!("{op}.{name}"),
                got,
                &fx.tensor(&format!("{op}.{name}")),
                TRIANGLE_ATTN,
            );
            out.push((name, d));
        }
        out
    }

    fn check_output(fx: &Fixtures, op: &str, node: TriangleNode) -> Deviation {
        let w = fx.triangle_attn_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let got = triangle_attn(&input, &w, node).unwrap();
        assert_close(
            &format!("{op}.output"),
            &got,
            &fx.tensor(&format!("{op}.output")),
            TRIANGLE_ATTN,
        )
    }

    #[test]
    fn starting_intermediates_match_the_dumped_stages() {
        let fx = Fixtures::load();
        check_intermediates(&fx, "tri_attn_start", TriangleNode::Starting);
    }

    #[test]
    fn starting_output_matches_the_dumped_output() {
        let fx = Fixtures::load();
        check_output(&fx, "tri_attn_start", TriangleNode::Starting);
    }

    #[test]
    fn ending_intermediates_match_the_dumped_stages() {
        let fx = Fixtures::load();
        check_intermediates(&fx, "tri_attn_end", TriangleNode::Ending);
    }

    #[test]
    fn ending_output_matches_the_dumped_output() {
        let fx = Fixtures::load();
        check_output(&fx, "tri_attn_end", TriangleNode::Ending);
    }

    /// Trap one. foundry projects the ending-node bias from the untransposed
    /// normalised pair, at `attention.py:85-88`, and transposes the pair only
    /// afterwards. Computing the bias from the transposed pair instead is
    /// equivalent to transposing the correct bias, because `to_b` acts on the
    /// channel axis and commutes with the i/j swap. This test pins the
    /// orientation: the computed bias equals the dumped untransposed bias,
    /// and the dumped bias differs from its own transpose by far more than the
    /// tolerance, so the first comparison has discriminating power.
    #[test]
    fn the_ending_bias_is_projected_from_the_untransposed_pair() {
        let fx = Fixtures::load();
        let op = "tri_attn_end";
        let w = fx.triangle_attn_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let s = triangle_attn_stages(&input, &w, TriangleNode::Ending).unwrap();

        let dumped = fx.tensor(&format!("{op}.bias"));
        assert_close(
            &format!("{op}.bias vs the untransposed dump"),
            &s.bias,
            &dumped,
            TRIANGLE_ATTN,
        );

        let transposed = dumped.transpose_axes(1, 2);
        // Measured margin: max abs deviation 3.86, two orders of magnitude
        // above the 1e-2 attention tolerance.
        assert_far(
            &format!("{op}.bias must differ from its own transpose"),
            &transposed,
            &dumped,
            0.5,
        );

        // The starting node shares the orientation claim on its own fixtures.
        let op = "tri_attn_start";
        let w = fx.triangle_attn_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let s = triangle_attn_stages(&input, &w, TriangleNode::Starting).unwrap();
        assert_close(
            &format!("{op}.bias"),
            &s.bias,
            &fx.tensor(&format!("{op}.bias")),
            TRIANGLE_ATTN,
        );
    }

    /// The ending node reads its query, key, value and gate from the
    /// transposed pair, and the fixture records that tensor as `qkv_pair`.
    /// This test pins the transpose itself: for the ending node `qkv_pair`
    /// equals the transpose of `norm_pair` and differs from `norm_pair`, so a
    /// missing or doubled transpose fails.
    #[test]
    fn the_ending_qkv_pair_is_the_transposed_norm_pair() {
        let fx = Fixtures::load();
        let op = "tri_attn_end";
        let qkv = fx.tensor(&format!("{op}.qkv_pair"));
        let norm = fx.tensor(&format!("{op}.norm_pair"));

        assert_close(
            &format!("{op}.qkv_pair vs transpose(norm_pair)"),
            &norm.transpose_axes(1, 2),
            &qkv,
            TRIANGLE_ATTN,
        );
        // Measured margin: max abs deviation 5.50 between norm_pair and
        // qkv_pair at the same index.
        assert_far(
            &format!("{op}.qkv_pair must differ from norm_pair"),
            &norm,
            &qkv,
            0.5,
        );
    }

    /// The attention fixtures come from a real RF3 checkpoint, so the dumped
    /// `output` and `gate` tensors carry signal and their comparisons can
    /// fail. Foundry's default initialisation zero-fills `to_out.weight`,
    /// `to_out.bias` and `to_g.weight`, at `attention.py:74-79`, and fixtures
    /// built from it once made both comparisons vacuous: the dumped output
    /// was zero everywhere and the dumped gate was the constant `sigmoid(1)`.
    /// This test pins the property that keeps them honest. A regeneration
    /// that reintroduces degenerate weights fails here before any output
    /// comparison can pass against zero again.
    #[test]
    fn the_attention_fixtures_are_non_degenerate() {
        let fx = Fixtures::load();
        for op in ["tri_attn_start", "tri_attn_end"] {
            // Measured stddevs: to_out.weight 0.43 and 0.44, to_out.bias 0.17
            // and 0.21, to_g.weight 0.49 and 0.50, all above the 1e-2 bound.
            for key in ["to_out.weight", "to_out.bias", "to_g.weight"] {
                let s = stddev(fx.weight(op, key).data());
                assert!(
                    s > 1e-2,
                    "{op}.{key} has stddev {s}, expected real spread. Foundry's default \
                     initialisation zero-fills this weight. If the fixtures were rebuilt \
                     from it, the {op}.output and {op}.gate comparisons pass vacuously. \
                     Regenerate them from a trained checkpoint."
                );
            }

            let output = fx.tensor(&format!("{op}.output"));
            assert!(
                output.data().iter().any(|v| *v != 0.0),
                "{op}.output is all zero, so comparing against it catches nothing. \
                 Regenerate the fixtures from a trained checkpoint."
            );
            // Measured stddevs 18.2 (starting) and 17.1 (ending).
            let s = stddev(output.data());
            assert!(
                s > 1.0,
                "{op}.output has stddev {s}, expected above 1.0. An output this flat \
                 cannot discriminate a numerical error."
            );

            // Measured range 1.0 for both, from about 2e-8 up to 1.0.
            let gate = fx.tensor(&format!("{op}.gate"));
            let (lo, hi) = gate
                .data()
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            assert!(
                hi - lo > 0.5,
                "{op}.gate spans only {lo} to {hi}. A constant gate means to_g is still \
                 at its default initialisation, and the gate comparison passes for any \
                 implementation that ignores its input."
            );
        }
    }

    fn stddev(data: &[f32]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let var = data
            .iter()
            .map(|v| {
                let d = f64::from(*v) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        var.sqrt()
    }
}
