//! Triangle multiplication, outgoing and incoming.
//!
//! Mirrors foundry `TriangleMultiplication._forward_vanilla` at commit
//! `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`,
//! `models/rf3/src/rf3/model/layers/attention.py:289-328`:
//!
//! ```text
//! pair_norm = LayerNorm(pair)
//! left, right           = split(p_in(pair_norm))
//! left_gate, right_gate = split(sigmoid(g_in(pair_norm)))
//! left  = left_gate * left
//! right = right_gate * right
//! outgoing: out = einsum("bikd,bjkd->bijd", left, right / L)
//! incoming: out = einsum("bkid,bkjd->bijd", left, right / L)
//! out = sigmoid(g_out(pair_norm)) * p_out(LayerNorm(out))
//! ```
//!
//! Two details here are load-bearing, and each has a named test. The output
//! gate reads the normalised input `pair_norm`, not the raw pair, and it
//! multiplies the output projection's result, not the normalised pair. Only
//! the right einsum operand is divided by the sequence length `L`.

use crate::nn::{layer_norm, linear, sigmoid};
use crate::tensor::Tensor;
use crate::Error;

/// The layer-norm epsilon foundry uses everywhere in these operations.
const EPS: f32 = 1e-5;

/// Which contraction the operation runs. The two directions differ only in the
/// einsum subscripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d] / L`
    Outgoing,
    /// `out[b,i,j,d] = sum_k left[b,k,i,d] * right[b,k,j,d] / L`
    Incoming,
}

/// The weights of one triangle multiplication, under foundry's `state_dict`
/// names. `p_in`, `g_in`, `p_out` and `g_out` have no bias term, matching
/// foundry's `bias=False` projections. The two layer norms carry both
/// parameters.
#[derive(Debug, Clone)]
pub struct TriangleMulWeights {
    pub norm_in_weight: Vec<f32>,
    pub norm_in_bias: Vec<f32>,
    /// Shape `(2 * d_hidden, d_pair)`, row-major.
    pub p_in_weight: Tensor,
    /// Shape `(2 * d_hidden, d_pair)`, row-major.
    pub g_in_weight: Tensor,
    pub norm_out_weight: Vec<f32>,
    pub norm_out_bias: Vec<f32>,
    /// Shape `(d_pair, d_hidden)`, row-major.
    pub p_out_weight: Tensor,
    /// Shape `(d_pair, d_pair)`, row-major.
    pub g_out_weight: Tensor,
}

/// Every intermediate the forward pass produces, under the names the oracle
/// fixtures use. Tests check each one against its dumped tensor, so a failure
/// names the stage that broke.
#[derive(Debug, Clone)]
pub struct TriangleMulStages {
    pub norm_pair: Tensor,
    pub p_combined: Tensor,
    pub g_combined: Tensor,
    pub left_gate: Tensor,
    pub right_gate: Tensor,
    pub left_gated: Tensor,
    pub right_gated_over_l: Tensor,
    pub einsum_raw: Tensor,
    pub norm_out_applied: Tensor,
    pub out_proj: Tensor,
    pub out_gate_logit: Tensor,
    pub out_gate: Tensor,
    pub output: Tensor,
}

/// Runs the forward pass and keeps every intermediate.
pub fn triangle_mul_stages(
    pair: &Tensor,
    w: &TriangleMulWeights,
    direction: Direction,
) -> Result<TriangleMulStages, Error> {
    let shape = pair.shape();
    if shape.len() != 4 {
        return Err(Error::Mismatch {
            context: "triangle_mul input".to_string(),
            expected: "shape (B, L, L, d_pair)".to_string(),
            got: format!("shape {shape:?}"),
        });
    }
    let seq_len = shape[1] as f32;

    let norm_pair = layer_norm(pair, &w.norm_in_weight, Some(&w.norm_in_bias), EPS)?;
    let p_combined = linear(&norm_pair, &w.p_in_weight, None)?;
    let (left_p, right_p) = p_combined.split_last_half()?;
    let g_combined = linear(&norm_pair, &w.g_in_weight, None)?;
    let (left_g, right_g) = g_combined.split_last_half()?;
    let left_gate = sigmoid(&left_g);
    let right_gate = sigmoid(&right_g);
    let left_gated = left_gate.mul(&left_p)?;
    let right_gated = right_gate.mul(&right_p)?;
    // foundry divides the right operand by float(L), elementwise. Division by
    // L is not the same rounding as multiplication by 1/L, so divide.
    let right_gated_over_l = right_gated.map(|v| v / seq_len);

    let einsum_raw = contract(direction, &left_gated, &right_gated_over_l)?;
    let norm_out_applied =
        layer_norm(&einsum_raw, &w.norm_out_weight, Some(&w.norm_out_bias), EPS)?;
    let out_proj = linear(&norm_out_applied, &w.p_out_weight, None)?;
    // The gate reads the normalised input, not the raw pair.
    let out_gate_logit = linear(&norm_pair, &w.g_out_weight, None)?;
    let out_gate = sigmoid(&out_gate_logit);
    // The gate multiplies the projection result, not the normalised pair.
    let output = out_gate.mul(&out_proj)?;

    Ok(TriangleMulStages {
        norm_pair,
        p_combined,
        g_combined,
        left_gate,
        right_gate,
        left_gated,
        right_gated_over_l,
        einsum_raw,
        norm_out_applied,
        out_proj,
        out_gate_logit,
        out_gate,
        output,
    })
}

/// Runs the forward pass and returns the value a pairformer block adds to its
/// residual.
pub fn triangle_mul(
    pair: &Tensor,
    w: &TriangleMulWeights,
    direction: Direction,
) -> Result<Tensor, Error> {
    Ok(triangle_mul_stages(pair, w, direction)?.output)
}

/// The einsum contraction, with the subscript order foundry uses. Both
/// operands are `(B, L, L, D)` and so is the result. Accumulates in f64.
fn contract(direction: Direction, left: &Tensor, right: &Tensor) -> Result<Tensor, Error> {
    let ls = left.shape();
    let rs = right.shape();
    if ls.len() != 4 || ls != rs || ls[1] != ls[2] {
        return Err(Error::Mismatch {
            context: "triangle multiplication contraction".to_string(),
            expected: "two equal square pair shapes (B, L, L, D)".to_string(),
            got: format!("{ls:?} and {rs:?}"),
        });
    }
    let (nb, l, d) = (ls[0], ls[1], ls[3]);
    let lo = left.data();
    let ro = right.data();
    let mut acc = vec![0.0f64; nb * l * l * d];
    for b in 0..nb {
        for i in 0..l {
            for j in 0..l {
                let out_base = ((b * l + i) * l + j) * d;
                for k in 0..l {
                    let (left_base, right_base) = match direction {
                        // left[b,i,k,d] pairs with right[b,j,k,d]
                        Direction::Outgoing => {
                            (((b * l + i) * l + k) * d, ((b * l + j) * l + k) * d)
                        }
                        // left[b,k,i,d] pairs with right[b,k,j,d]
                        Direction::Incoming => {
                            (((b * l + k) * l + i) * d, ((b * l + k) * l + j) * d)
                        }
                    };
                    let out_slot = &mut acc[out_base..out_base + d];
                    let left_slot = &lo[left_base..left_base + d];
                    let right_slot = &ro[right_base..right_base + d];
                    for (slot, (lv, rv)) in
                        out_slot.iter_mut().zip(left_slot.iter().zip(right_slot))
                    {
                        *slot += f64::from(*lv) * f64::from(*rv);
                    }
                }
            }
        }
    }
    let data: Vec<f32> = acc.iter().map(|v| *v as f32).collect();
    Ok(Tensor::internal(ls.to_vec(), data))
}

#[cfg(test)]
mod tests {
    use super::{triangle_mul, triangle_mul_stages, Direction};
    use crate::check::{assert_close, assert_far, Deviation, TRIANGLE_MUL};
    use crate::fixtures::Fixtures;
    use crate::tensor::Tensor;

    /// Checks every dumped intermediate of one operation and returns the
    /// deviations, so the report test can print them.
    fn check_intermediates(
        fx: &Fixtures,
        op: &str,
        direction: Direction,
    ) -> Vec<(&'static str, Deviation)> {
        let w = fx.triangle_mul_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let s = triangle_mul_stages(&input, &w, direction).unwrap();
        let mut out = Vec::new();
        let cases: Vec<(&'static str, &Tensor)> = vec![
            ("norm_pair", &s.norm_pair),
            ("p_combined", &s.p_combined),
            ("g_combined", &s.g_combined),
            ("left_gate", &s.left_gate),
            ("right_gate", &s.right_gate),
            ("left_gated", &s.left_gated),
            ("right_gated_over_l", &s.right_gated_over_l),
            ("einsum_raw", &s.einsum_raw),
            ("norm_out_applied", &s.norm_out_applied),
            ("out_proj", &s.out_proj),
            ("out_gate_logit", &s.out_gate_logit),
            ("out_gate", &s.out_gate),
        ];
        for (name, got) in cases {
            let d = assert_close(
                &format!("{op}.{name}"),
                got,
                &fx.tensor(&format!("{op}.{name}")),
                TRIANGLE_MUL,
            );
            out.push((name, d));
        }
        out
    }

    fn check_output(fx: &Fixtures, op: &str, direction: Direction) -> Deviation {
        let w = fx.triangle_mul_weights(op);
        let input = fx.tensor(&format!("{op}.input"));
        let got = triangle_mul(&input, &w, direction).unwrap();
        assert_close(
            &format!("{op}.output"),
            &got,
            &fx.tensor(&format!("{op}.output")),
            TRIANGLE_MUL,
        )
    }

    #[test]
    fn outgoing_intermediates_match_the_dumped_stages() {
        let fx = Fixtures::load();
        check_intermediates(&fx, "tri_mul_outgoing", Direction::Outgoing);
    }

    #[test]
    fn outgoing_output_matches_the_dumped_output() {
        let fx = Fixtures::load();
        check_output(&fx, "tri_mul_outgoing", Direction::Outgoing);
    }

    #[test]
    fn incoming_intermediates_match_the_dumped_stages() {
        let fx = Fixtures::load();
        check_intermediates(&fx, "tri_mul_incoming", Direction::Incoming);
    }

    #[test]
    fn incoming_output_matches_the_dumped_output() {
        let fx = Fixtures::load();
        check_output(&fx, "tri_mul_incoming", Direction::Incoming);
    }

    /// Trap two. foundry computes `sigmoid(g_out(pair_norm))` from the
    /// normalised input and multiplies the output projection's result with it,
    /// at `attention.py:325-326`. Gating the normalised pair instead shifts
    /// magnitudes quietly and produces no error. This test pins the right
    /// operand of the gate against the implementation's own stages: the
    /// computed output equals the computed gate times the computed projection,
    /// rounding for rounding, and differs from the computed gate times the
    /// normalised pair by far more than the tolerance, so an implementation
    /// that picked the wrong operand fails. The dumped tensors are checked the
    /// same way, which proves the fixture itself has discriminating power.
    #[test]
    fn the_output_gate_multiplies_the_projection_not_the_normalised_pair() {
        let fx = Fixtures::load();
        for (op, direction) in [
            ("tri_mul_outgoing", Direction::Outgoing),
            ("tri_mul_incoming", Direction::Incoming),
        ] {
            let input = fx.tensor(&format!("{op}.input"));
            let s = triangle_mul_stages(&input, &fx.triangle_mul_weights(op), direction).unwrap();

            let projection_side = s.out_gate.mul(&s.out_proj).unwrap();
            let d = assert_close(
                &format!("{op}: computed output vs computed out_gate * out_proj"),
                &projection_side,
                &s.output,
                TRIANGLE_MUL,
            );
            assert_eq!(
                d.max_abs, 0.0,
                "{op}: both sides are the same elementwise f32 product; a nonzero \
                 deviation means the output was not built from out_gate * out_proj"
            );

            let norm_side = s.out_gate.mul(&s.norm_pair).unwrap();
            // Measured margin: 3.50 outgoing, 3.10 incoming, three orders of
            // magnitude above the 1e-3 multiplication tolerance.
            assert_far(
                &format!("{op}: out_gate * norm_pair must differ from the output"),
                &norm_side,
                &s.output,
                0.5,
            );

            // The dumped tensors carry the same identity, so the fixture can
            // tell the two operands apart.
            let dumped_gate = fx.tensor(&format!("{op}.out_gate"));
            let dumped_proj = fx.tensor(&format!("{op}.out_proj"));
            let dumped_norm = fx.tensor(&format!("{op}.norm_pair"));
            let dumped_output = fx.tensor(&format!("{op}.output"));
            assert_far(
                &format!("{op}: dumped out_gate * norm_pair must differ from the dumped output"),
                &dumped_gate.mul(&dumped_norm).unwrap(),
                &dumped_output,
                0.5,
            );
            assert_close(
                &format!("{op}: dumped out_gate * out_proj vs dumped output"),
                &dumped_gate.mul(&dumped_proj).unwrap(),
                &dumped_output,
                TRIANGLE_MUL,
            );
        }
    }
}
