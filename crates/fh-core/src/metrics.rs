//! Scoring. The harness minimises a single weighted loss so that different
//! folding backends can be compared on one axis.

use serde::{Deserialize, Serialize};

/// Metrics for one folded candidate. Field names are backend-neutral: a
/// backend adapter is responsible for mapping its own output onto these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Interface predicted TM-score, 0..1, higher is better.
    pub iptm: f64,
    /// Whole-complex predicted TM-score, 0..1.
    pub ptm: f64,
    /// Mean per-residue confidence over the whole complex, 0..100.
    pub plddt_mean: f64,
    /// Mean per-residue confidence restricted to interface residues, 0..100.
    pub interface_plddt: f64,
    /// Mean predicted aligned error across the interface, in angstroms.
    pub pae_interface: f64,
    /// Number of inter-chain residue contacts under the contact threshold.
    pub interface_contacts: u32,
    /// Steric clashes detected in the predicted structure.
    pub clashes: u32,
    /// Largest exposed hydrophobic patch, normalised 0..1.
    pub hydrophobic_patch: f64,
    /// Net formal charge of the binder at neutral pH.
    pub net_charge: f64,
    /// Per-residue confidence for the binder chains, used by the structure
    /// viewer. Not part of the loss.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plddt: Vec<f32>,
}

impl Metrics {
    /// Neutral placeholder, useful in tests.
    pub fn zeroed() -> Self {
        Metrics {
            iptm: 0.0,
            ptm: 0.0,
            plddt_mean: 0.0,
            interface_plddt: 0.0,
            pae_interface: 30.0,
            interface_contacts: 0,
            clashes: 0,
            hydrophobic_patch: 0.0,
            net_charge: 0.0,
            plddt: Vec::new(),
        }
    }
}

/// Weights for the loss terms. Every term is normalised to roughly 0..1 before
/// weighting, so the weights are directly comparable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LossWeights {
    pub iptm: f64,
    pub interface_plddt: f64,
    pub pae_interface: f64,
    pub interface_contacts: f64,
    pub clashes: f64,
    pub hydrophobic_patch: f64,
    pub net_charge: f64,
    pub plddt_mean: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        LossWeights {
            iptm: 3.0,
            interface_plddt: 2.0,
            pae_interface: 1.0,
            interface_contacts: 1.0,
            clashes: 2.0,
            hydrophobic_patch: 0.5,
            net_charge: 0.25,
            plddt_mean: 0.5,
        }
    }
}

impl LossWeights {
    pub fn total_weight(&self) -> f64 {
        self.iptm
            + self.interface_plddt
            + self.pae_interface
            + self.interface_contacts
            + self.clashes
            + self.hydrophobic_patch
            + self.net_charge
            + self.plddt_mean
    }
}

/// One named contribution to the loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossTerm {
    pub name: &'static str,
    /// The metric's normalised penalty, 0..1, before weighting.
    pub penalty: f64,
    pub weight: f64,
    /// `penalty * weight`.
    pub contribution: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossBreakdown {
    pub terms: Vec<LossTerm>,
    pub total: f64,
}

fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Normalised penalties, each 0..1 and lower-is-better.
fn penalties(m: &Metrics) -> [(&'static str, f64); 8] {
    [
        ("iptm", clamp01(1.0 - m.iptm)),
        ("interface_plddt", clamp01(1.0 - m.interface_plddt / 100.0)),
        ("pae_interface", clamp01(m.pae_interface / 30.0)),
        (
            "interface_contacts",
            clamp01(1.0 - f64::from(m.interface_contacts) / 60.0),
        ),
        ("clashes", clamp01(f64::from(m.clashes) / 10.0)),
        ("hydrophobic_patch", clamp01(m.hydrophobic_patch)),
        ("net_charge", clamp01(m.net_charge.abs() / 10.0)),
        ("plddt_mean", clamp01(1.0 - m.plddt_mean / 100.0)),
    ]
}

fn weight_of(name: &str, w: &LossWeights) -> f64 {
    match name {
        "iptm" => w.iptm,
        "interface_plddt" => w.interface_plddt,
        "pae_interface" => w.pae_interface,
        "interface_contacts" => w.interface_contacts,
        "clashes" => w.clashes,
        "hydrophobic_patch" => w.hydrophobic_patch,
        "net_charge" => w.net_charge,
        "plddt_mean" => w.plddt_mean,
        _ => 0.0,
    }
}

/// Weighted loss, lower is better. Normalised by total weight so that changing
/// the weights does not change the scale of the number.
pub fn loss(m: &Metrics, w: &LossWeights) -> f64 {
    let total_weight = w.total_weight();
    if total_weight <= 0.0 {
        return f64::INFINITY;
    }
    let sum: f64 = penalties(m)
        .iter()
        .map(|(name, penalty)| penalty * weight_of(name, w))
        .sum();
    sum / total_weight
}

pub fn loss_breakdown(m: &Metrics, w: &LossWeights) -> LossBreakdown {
    let total_weight = w.total_weight().max(f64::EPSILON);
    let terms: Vec<LossTerm> = penalties(m)
        .iter()
        .map(|(name, penalty)| {
            let weight = weight_of(name, w);
            LossTerm {
                name,
                penalty: *penalty,
                weight,
                contribution: penalty * weight / total_weight,
            }
        })
        .collect();
    let total = terms.iter().map(|t| t.contribution).sum();
    LossBreakdown { terms, total }
}
