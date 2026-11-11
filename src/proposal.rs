//! Design skills, and the validated proposal that is the only way to change a
//! sequence.

use serde::{Deserialize, Serialize};

use crate::ids::{CandidateId, ChainId};
use crate::scaffold::{is_standard_aa, BinderSequence, MutableScope};

/// A capability the compute service must advertise before a skill that needs it
/// is considered legal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Fold,
    InverseFolding,
    SequenceLikelihood,
}

impl Capability {
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::Fold => "fold",
            Capability::InverseFolding => "inverse_folding",
            Capability::SequenceLikelihood => "sequence_likelihood",
        }
    }
}

/// The design actions the harness knows about. The router decides which are
/// legal in a given cycle; the agent picks one from that set; the executor runs
/// it. Adding a variant here is deliberately a change to the core vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillId {
    /// Substitute a small number of positions in mutable regions.
    PointMutation,
    /// Replace one whole mutable region.
    RegionRedesign,
    /// Structure-conditioned proposals from an inverse-folding model.
    InverseFolding,
    /// Substitutions guided by a sequence-likelihood model.
    LikelihoodGuided,
}

impl SkillId {
    pub const ALL: [SkillId; 4] = [
        SkillId::PointMutation,
        SkillId::RegionRedesign,
        SkillId::InverseFolding,
        SkillId::LikelihoodGuided,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            SkillId::PointMutation => "point_mutation",
            SkillId::RegionRedesign => "region_redesign",
            SkillId::InverseFolding => "inverse_folding",
            SkillId::LikelihoodGuided => "likelihood_guided",
        }
    }

    /// The compute capability this skill requires, if any. A skill whose
    /// capability is absent is never offered to the agent.
    pub fn required_capability(&self) -> Option<Capability> {
        match self {
            SkillId::PointMutation | SkillId::RegionRedesign => None,
            SkillId::InverseFolding => Some(Capability::InverseFolding),
            SkillId::LikelihoodGuided => Some(Capability::SequenceLikelihood),
        }
    }
}

impl std::fmt::Display for SkillId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Substitution {
    pub chain: ChainId,
    pub position: usize,
    pub to: char,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ProposalError {
    #[error("{chain} position {position} is outside the mutable scope (framework or pinned)")]
    Immutable { chain: ChainId, position: usize },
    #[error("{chain} position {position} is beyond the chain (length {len})")]
    OutOfRange {
        chain: ChainId,
        position: usize,
        len: usize,
    },
    #[error("'{to}' is not a standard amino acid")]
    NonStandardResidue { to: char },
    #[error("{chain} position {position} is substituted more than once")]
    DuplicatePosition { chain: ChainId, position: usize },
    #[error("proposal has no substitutions")]
    Empty,
    #[error("parent sequence does not fit the scaffold: {detail}")]
    ParentMismatch { detail: String },
}

/// A validated set of substitutions.
///
/// Every field is private and there is **no `Deserialize` impl** — a proposal
/// cannot be constructed by parsing JSON or YAML, only by
/// [`MutableScope::propose`], which checks each substitution against the scope.
/// That is what makes "framework residues are never mutated" a property of the
/// type rather than a rule someone has to remember.
#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    parent: Option<CandidateId>,
    skill: SkillId,
    substitutions: Vec<Substitution>,
    rationale: String,
}

impl Proposal {
    pub fn parent(&self) -> Option<CandidateId> {
        self.parent
    }

    pub fn skill(&self) -> SkillId {
        self.skill
    }

    pub fn substitutions(&self) -> &[Substitution] {
        &self.substitutions
    }

    pub fn rationale(&self) -> &str {
        &self.rationale
    }

    pub fn len(&self) -> usize {
        self.substitutions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.substitutions.is_empty()
    }

    /// Apply to a parent sequence, yielding the child. Fails if the parent does
    /// not have the chains and lengths the substitutions assume.
    pub fn apply_to(&self, parent: &BinderSequence) -> Result<BinderSequence, ProposalError> {
        let mut child = parent.clone();
        for sub in &self.substitutions {
            if !child.replace(sub.chain, sub.position, sub.to) {
                return Err(ProposalError::ParentMismatch {
                    detail: format!(
                        "no residue at {} position {} in the parent sequence",
                        sub.chain, sub.position
                    ),
                });
            }
        }
        Ok(child)
    }

    /// Compact human-readable form, e.g. `chain0:52 S>Y, chain0:99 A>W`.
    pub fn describe(&self, parent: &BinderSequence) -> String {
        self.substitutions
            .iter()
            .map(|sub| {
                let from = parent
                    .chain(sub.chain)
                    .and_then(|s| s.chars().nth(sub.position))
                    .unwrap_or('?');
                format!("{}:{} {}>{}", sub.chain, sub.position + 1, from, sub.to)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl MutableScope {
    /// The single gate through which a sequence change must pass.
    ///
    /// Rejects substitutions that fall outside the scope, run past the end of a
    /// chain, introduce a non-standard residue, or touch the same position
    /// twice.
    pub fn propose(
        &self,
        parent: Option<CandidateId>,
        skill: SkillId,
        substitutions: Vec<Substitution>,
        rationale: impl Into<String>,
    ) -> Result<Proposal, ProposalError> {
        if substitutions.is_empty() {
            return Err(ProposalError::Empty);
        }

        let mut seen: Vec<(ChainId, usize)> = Vec::with_capacity(substitutions.len());
        for sub in &substitutions {
            let len = self.chain_len(sub.chain).ok_or(ProposalError::OutOfRange {
                chain: sub.chain,
                position: sub.position,
                len: 0,
            })?;
            if sub.position >= len {
                return Err(ProposalError::OutOfRange {
                    chain: sub.chain,
                    position: sub.position,
                    len,
                });
            }
            if !is_standard_aa(sub.to) {
                return Err(ProposalError::NonStandardResidue { to: sub.to });
            }
            if !self.is_mutable(sub.chain, sub.position) {
                return Err(ProposalError::Immutable {
                    chain: sub.chain,
                    position: sub.position,
                });
            }
            if seen.contains(&(sub.chain, sub.position)) {
                return Err(ProposalError::DuplicatePosition {
                    chain: sub.chain,
                    position: sub.position,
                });
            }
            seen.push((sub.chain, sub.position));
        }

        Ok(Proposal {
            parent,
            skill,
            substitutions,
            rationale: rationale.into(),
        })
    }
}
