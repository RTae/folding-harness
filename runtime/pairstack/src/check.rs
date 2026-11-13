//! Test-only comparison helpers.
//!
//! `assert_close` checks a computed tensor against a dumped fixture at
//! BioNeMo's fp32-reference tolerances, recorded in `docs/pair-stack-design.md`
//! under Numerical validation. Cross-language fp32 is not bit-exact, because
//! summation order differs between PyTorch's einsum and the loops here, so
//! these tolerances and not equality are the passing condition. The panic
//! message names the tensor and reports the measured margins.

use crate::tensor::Tensor;

/// The passing condition, in the `torch.allclose` form:
/// `|actual - expected| <= atol + rtol * |expected|` for every element.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub atol: f32,
    pub rtol: f32,
}

/// BioNeMo's tolerance for triangle multiplication against an fp32 reference,
/// `tests/_torch/test_triangle_nodes.py:211`.
pub const TRIANGLE_MUL: Tolerance = Tolerance {
    atol: 1e-3,
    rtol: 1e-4,
};

/// BioNeMo's tolerance for triangle attention against an fp32 reference,
/// `tests/_torch/test_triangle_attention.py:112`.
pub const TRIANGLE_ATTN: Tolerance = Tolerance {
    atol: 1e-2,
    rtol: 1e-3,
};

/// The measured distance between two equally shaped tensors.
#[derive(Debug, Clone, Copy)]
pub struct Deviation {
    /// The largest `|actual - expected|` over all elements.
    pub max_abs: f32,
    /// The largest `|actual - expected| / |expected|` over the elements whose
    /// expected value is at least `1e-6` in magnitude. Relative deviation is
    /// meaningless where the reference is near zero, and those elements are
    /// already covered by `max_abs` against `atol`.
    pub max_rel: f32,
    /// The largest `|actual - expected| - rtol * |expected|`, the quantity the
    /// `atol` bound is applied to. Negative means inside the tolerance with
    /// that much margin.
    pub max_excess: f32,
}

/// Elements below this magnitude are skipped when measuring relative
/// deviation.
const REL_FLOOR: f32 = 1e-6;

/// Measures the deviation between two equally shaped tensors.
pub fn deviation(actual: &Tensor, expected: &Tensor, rtol: f32) -> Deviation {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape disagreement: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_excess = f32::NEG_INFINITY;
    for (a, e) in actual.data().iter().zip(expected.data()) {
        let abs = (*a - *e).abs();
        max_abs = max_abs.max(abs);
        max_excess = max_excess.max(abs - rtol * e.abs());
        if e.abs() >= REL_FLOOR {
            max_rel = max_rel.max(abs / e.abs());
        }
    }
    Deviation {
        max_abs,
        max_rel,
        max_excess,
    }
}

/// Fails unless every element satisfies
/// `|actual - expected| <= atol + rtol * |expected|`. The failure message
/// names the tensor and reports the measured maximum absolute and relative
/// deviation, so the margin is visible in a failing run too.
pub fn assert_close(what: &str, actual: &Tensor, expected: &Tensor, tol: Tolerance) -> Deviation {
    let d = deviation(actual, expected, tol.rtol);
    assert!(
        d.max_excess <= tol.atol,
        "{what}: outside tolerance. max_abs {} max_rel {} vs atol {} rtol {}; \
         the worst element exceeds the bound by {}",
        d.max_abs,
        d.max_rel,
        tol.atol,
        tol.rtol,
        d.max_excess - tol.atol
    );
    d
}

/// Fails unless the two tensors differ by more than `min_abs` somewhere. Used
/// by the trap tests to prove a comparison has discriminating power: the wrong
/// variant must be detectably wrong, not accidentally equal.
pub fn assert_far(what: &str, a: &Tensor, b: &Tensor, min_abs: f32) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape disagreement");
    let max_abs = a
        .data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > min_abs,
        "{what}: the two tensors differ by only {max_abs}, expected more than {min_abs}. \
         The companion equality check would pass for the wrong variant too."
    );
}
