//! Hard gates and rejection reasons.
//!
//! Gates are deliberately separate from the loss: the loss ranks candidates
//! that are acceptable, gates decide what is acceptable at all. Keeping them
//! apart means a bad candidate cannot be weighted back into the population.

use serde::{Deserialize, Serialize};

use crate::ids::CandidateId;
use crate::metrics::Metrics;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gates {
    pub min_iptm: f64,
    pub min_interface_plddt: f64,
    pub max_pae_interface: f64,
    pub max_clashes: u32,
    pub min_interface_contacts: u32,
    pub max_hydrophobic_patch: f64,
}

impl Default for Gates {
    fn default() -> Self {
        Gates {
            min_iptm: 0.4,
            min_interface_plddt: 60.0,
            max_pae_interface: 15.0,
            max_clashes: 2,
            min_interface_contacts: 10,
            max_hydrophobic_patch: 0.75,
        }
    }
}

impl Gates {
    /// A gate set that admits anything. Useful for a first exploratory run.
    pub fn permissive() -> Self {
        Gates {
            min_iptm: 0.0,
            min_interface_plddt: 0.0,
            max_pae_interface: f64::INFINITY,
            max_clashes: u32::MAX,
            min_interface_contacts: 0,
            max_hydrophobic_patch: 1.0,
        }
    }

    pub fn check(&self, m: &Metrics) -> Result<(), RejectReason> {
        if m.iptm < self.min_iptm {
            return Err(RejectReason::gate("iptm", m.iptm, self.min_iptm, ">="));
        }
        if m.interface_plddt < self.min_interface_plddt {
            return Err(RejectReason::gate(
                "interface_plddt",
                m.interface_plddt,
                self.min_interface_plddt,
                ">=",
            ));
        }
        if m.pae_interface > self.max_pae_interface {
            return Err(RejectReason::gate(
                "pae_interface",
                m.pae_interface,
                self.max_pae_interface,
                "<=",
            ));
        }
        if m.clashes > self.max_clashes {
            return Err(RejectReason::gate(
                "clashes",
                f64::from(m.clashes),
                f64::from(self.max_clashes),
                "<=",
            ));
        }
        if m.interface_contacts < self.min_interface_contacts {
            return Err(RejectReason::gate(
                "interface_contacts",
                f64::from(m.interface_contacts),
                f64::from(self.min_interface_contacts),
                ">=",
            ));
        }
        if m.hydrophobic_patch > self.max_hydrophobic_patch {
            return Err(RejectReason::gate(
                "hydrophobic_patch",
                m.hydrophobic_patch,
                self.max_hydrophobic_patch,
                "<=",
            ));
        }
        Ok(())
    }
}

/// Why a candidate did not enter the population. Recorded in the trace so a
/// run can be audited after the fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum RejectReason {
    Gate {
        field: String,
        value: f64,
        bound: f64,
        comparison: String,
    },
    Duplicate {
        of: CandidateId,
    },
    TooSimilar {
        to: CandidateId,
        hamming: usize,
        required: usize,
    },
    WorseThanPopulation {
        worst: CandidateId,
        worst_loss: f64,
        loss: f64,
    },
    Unscored,
    Invalid {
        detail: String,
    },
    FoldFailed {
        detail: String,
    },
}

impl RejectReason {
    fn gate(field: &str, value: f64, bound: f64, comparison: &str) -> Self {
        RejectReason::Gate {
            field: field.to_string(),
            value,
            bound,
            comparison: comparison.to_string(),
        }
    }

    /// Short label for tables and counters.
    pub fn kind(&self) -> &'static str {
        match self {
            RejectReason::Gate { .. } => "gate",
            RejectReason::Duplicate { .. } => "duplicate",
            RejectReason::TooSimilar { .. } => "too_similar",
            RejectReason::WorseThanPopulation { .. } => "worse_than_population",
            RejectReason::Unscored => "unscored",
            RejectReason::Invalid { .. } => "invalid",
            RejectReason::FoldFailed { .. } => "fold_failed",
        }
    }
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::Gate {
                field,
                value,
                bound,
                comparison,
            } => write!(f, "gate {field}: {value:.3} fails {comparison} {bound:.3}"),
            RejectReason::Duplicate { of } => write!(f, "duplicate of {of}"),
            RejectReason::TooSimilar {
                to,
                hamming,
                required,
            } => write!(
                f,
                "within {hamming} substitutions of {to}, {required} required"
            ),
            RejectReason::WorseThanPopulation {
                worst,
                worst_loss,
                loss,
            } => write!(
                f,
                "loss {loss:.4} is worse than {worst} at {worst_loss:.4} and the population is full"
            ),
            RejectReason::Unscored => write!(f, "candidate has no loss"),
            RejectReason::Invalid { detail } => write!(f, "invalid: {detail}"),
            RejectReason::FoldFailed { detail } => write!(f, "fold failed: {detail}"),
        }
    }
}

impl std::error::Error for RejectReason {}
