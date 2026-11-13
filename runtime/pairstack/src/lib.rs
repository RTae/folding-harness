//! CPU reference implementation of the four AlphaFold 3 family triangle
//! operations: outgoing and incoming triangle multiplication, and starting-node
//! and ending-node triangle attention.
//!
//! The arithmetic mirrors foundry's vanilla PyTorch path at commit
//! `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`,
//! `models/rf3/src/rf3/model/layers/attention.py`. Two details are load-bearing
//! and are each pinned by a named test:
//!
//! - The ending-node attention projects its bias from the untransposed
//!   normalised pair and transposes the pair only afterwards, for the query,
//!   key, value and gate projections. See [`tri_attn`].
//! - The triangle-multiplication output gate is `sigmoid(g_out(pair_norm))`,
//!   taken from the normalised input, and multiplies the output projection's
//!   result. Only the right einsum operand is divided by the sequence length.
//!   See [`tri_mul`].
//!
//! Every operation is validated in tests against the fp32 reference tensors
//! committed under `tools/pairstack-oracle/fixtures/`, at BioNeMo's tolerances
//! for an fp32 reference: `atol=1e-3`, `rtol=1e-4` for multiplication and
//! `atol=1e-2`, `rtol=1e-3` for attention. Cross-language fp32 is not
//! bit-exact, because summation order differs from PyTorch's.
//!
//! The crate is standalone. The root `Cargo.toml` declares no workspace and
//! does not list this crate, so a build of the shipping `fh` crate never
//! compiles it. The library itself has no dependencies.

mod error;
mod nn;
mod tensor;
mod tri_attn;
mod tri_mul;

#[cfg(test)]
mod check;
#[cfg(test)]
mod fixtures;

pub use error::Error;
pub use nn::{layer_norm, linear, sigmoid, softmax};
pub use tensor::Tensor;
pub use tri_attn::{
    triangle_attn, triangle_attn_stages, TriangleAttnStages, TriangleAttnWeights, TriangleNode,
};
pub use tri_mul::{
    triangle_mul, triangle_mul_stages, Direction, TriangleMulStages, TriangleMulWeights,
};

#[cfg(test)]
mod tests {
    use crate::check::{assert_close, Deviation};
    use crate::fixtures::Fixtures;
    use crate::tensor::Tensor;
    use crate::tri_attn::{triangle_attn_stages, TriangleNode};
    use crate::tri_mul::{triangle_mul_stages, Direction};

    /// Runs all four operations, checks every dumped tensor against its
    /// computed counterpart, and prints the measured maximum absolute and
    /// relative deviation per tensor. Run with `--nocapture` to see the table.
    /// The stage tests carry the same comparisons under finer-grained names;
    /// this one exists so the margin against the tolerance stays visible.
    #[test]
    fn deviation_report() {
        let fx = Fixtures::load();
        let mut rows: Vec<(String, Deviation)> = Vec::new();

        for (op, direction) in [
            ("tri_mul_outgoing", Direction::Outgoing),
            ("tri_mul_incoming", Direction::Incoming),
        ] {
            let input = fx.tensor(&format!("{op}.input"));
            let s = triangle_mul_stages(&input, &fx.triangle_mul_weights(op), direction).unwrap();
            let tol = crate::check::TRIANGLE_MUL;
            let stages: Vec<(&str, &Tensor)> = vec![
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
                ("output", &s.output),
            ];
            for (name, got) in stages {
                let d = assert_close(
                    &format!("{op}.{name}"),
                    got,
                    &fx.tensor(&format!("{op}.{name}")),
                    tol,
                );
                rows.push((format!("{op}.{name}"), d));
            }
        }

        for (op, node) in [
            ("tri_attn_start", TriangleNode::Starting),
            ("tri_attn_end", TriangleNode::Ending),
        ] {
            let input = fx.tensor(&format!("{op}.input"));
            let s = triangle_attn_stages(&input, &fx.triangle_attn_weights(op), node).unwrap();
            let tol = crate::check::TRIANGLE_ATTN;
            let stages: Vec<(&str, &Tensor)> = vec![
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
                ("output", &s.output),
            ];
            for (name, got) in stages {
                let d = assert_close(
                    &format!("{op}.{name}"),
                    got,
                    &fx.tensor(&format!("{op}.{name}")),
                    tol,
                );
                rows.push((format!("{op}.{name}"), d));
            }
        }

        println!("{:<44} {:>12} {:>12}", "tensor", "max_abs", "max_rel");
        for (name, d) in &rows {
            println!("{name:<44} {:>12.3e} {:>12.3e}", d.max_abs, d.max_rel);
        }
        let worst_abs = rows
            .iter()
            .map(|(name, d)| (d.max_abs, name))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap();
        println!(
            "checked {} tensors; worst max_abs {} at {}",
            rows.len(),
            worst_abs.0,
            worst_abs.1
        );
    }
}
