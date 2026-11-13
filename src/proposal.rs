//! Design skills, and the validated proposal that is the only way to change a
//! sequence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

/// Reasons [`MutableScope::propose`] or [`Proposal::apply_to`] refuses a
/// change.
///
/// The variants that name a chain name it by its `label`, the string the task
/// YAML wrote, beside the internal `chain` id, so the rendered message names
/// something the reader can find in the task file. These messages reach the
/// trace, the dashboard and the run output, and whoever is debugging a
/// refused proposal is still the person who wrote `chain: H`. The id stays
/// available for callers that match on it.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ProposalError {
    #[error(
        "chain '{label}' position {position} is outside the mutable scope (framework or pinned)"
    )]
    Immutable {
        chain: ChainId,
        label: String,
        position: usize,
    },
    #[error("chain '{label}' position {position} is beyond the chain (length {len})")]
    OutOfRange {
        chain: ChainId,
        label: String,
        position: usize,
        len: usize,
    },
    #[error(
        "chain '{label}' is not a binder chain, so position {position} in it cannot be substituted; aim the proposal at a binder chain"
    )]
    NotABinderChain {
        chain: ChainId,
        label: String,
        position: usize,
    },
    #[error("'{to}' is not a standard amino acid")]
    NonStandardResidue { to: char },
    #[error("chain '{label}' position {position} is substituted more than once")]
    DuplicatePosition {
        chain: ChainId,
        label: String,
        position: usize,
    },
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
    /// The label the task YAML gave each chain, copied from the scope, so
    /// [`Proposal::apply_to`] can name a chain the way the reader's file does.
    /// Skipped by `Serialize` to keep the serialized shape of a proposal
    /// unchanged.
    #[serde(skip_serializing)]
    labels: BTreeMap<ChainId, String>,
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
                        "no residue at chain '{}' position {} in the parent sequence",
                        self.chain_label(sub.chain),
                        sub.position
                    ),
                });
            }
        }
        Ok(child)
    }

    /// The name an error message gives a chain: its label, which is what the
    /// task YAML wrote. A validated proposal only substitutes on chains the
    /// scope carries, so the id fallback serves hand-built internals alone.
    fn chain_label(&self, id: ChainId) -> String {
        self.labels
            .get(&id)
            .cloned()
            .unwrap_or_else(|| id.to_string())
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
    /// chain, aim at a chain the scope does not track as a binder, introduce a
    /// non-standard residue, or touch the same position twice.
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
            let len = self
                .chain_len(sub.chain)
                .ok_or_else(|| ProposalError::NotABinderChain {
                    chain: sub.chain,
                    label: self.chain_label(sub.chain),
                    position: sub.position,
                })?;
            if sub.position >= len {
                return Err(ProposalError::OutOfRange {
                    chain: sub.chain,
                    label: self.chain_label(sub.chain),
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
                    label: self.chain_label(sub.chain),
                    position: sub.position,
                });
            }
            if seen.contains(&(sub.chain, sub.position)) {
                return Err(ProposalError::DuplicatePosition {
                    chain: sub.chain,
                    label: self.chain_label(sub.chain),
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
            labels: self.chain_labels().clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TaskConfig;

    const VHH_YAML: &str = r#"
name: proposal-label-demo
binder:
  topology: vhh
  chains:
    - label: H
      sequence: ACDEFGHIKLMNPQRSTVWY
  regions:
    - name: loop1
      chain: H
      start: 5
      end: 8
target:
  chains:
    - label: A
      sequence: ACDEFGHIKLMNPQRSTVWY
design:
  cycles: 1
  batch_size: 1
"#;

    const PAIRED_YAML: &str = r#"
name: paired-proposal-label-demo
binder:
  topology: paired_chains
  chains:
    - label: H
      sequence: ACDEFGHIKLMNPQRSTVWY
    - label: L
      sequence: ACDEFGHIKLMNPQRSTVWY
  regions:
    - name: loopL
      chain: L
      start: 5
      end: 20
target:
  chains:
    - label: A
      sequence: ACDEFGHIKLMNPQRSTVWY
design:
  cycles: 1
  batch_size: 1
"#;

    /// A valid task whose binder chain `H` is only four residues long, so a
    /// proposal validated against `VHH_YAML` does not fit it.
    const SHORT_VHH_YAML: &str = r#"
name: short-parent-demo
binder:
  topology: vhh
  chains:
    - label: H
      sequence: ACDE
  regions:
    - name: loop1
      chain: H
      start: 1
      end: 4
target:
  chains:
    - label: A
      sequence: ACDE
design:
  cycles: 1
  batch_size: 1
"#;

    fn scope_for(yaml: &str) -> MutableScope {
        let config = TaskConfig::from_yaml(yaml).expect("fixture parses");
        let (_, scope) = config.resolve().expect("fixture resolves");
        scope
    }

    fn sub(chain: u8, position: usize, to: char) -> Substitution {
        Substitution {
            chain: ChainId(chain),
            position,
            to,
        }
    }

    fn refusal(scope: &MutableScope, substitutions: Vec<Substitution>) -> String {
        scope
            .propose(None, SkillId::PointMutation, substitutions, "test")
            .expect_err("must be refused")
            .to_string()
    }

    #[test]
    fn a_refused_framework_position_names_the_label_the_task_wrote() {
        // VHH_YAML stores loop1 as 4..8, so position 0 is framework.
        let msg = refusal(&scope_for(VHH_YAML), vec![sub(0, 0, 'W')]);
        assert_eq!(
            msg,
            "chain 'H' position 0 is outside the mutable scope (framework or pinned)"
        );
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn a_position_past_the_chain_end_names_the_label_the_task_wrote() {
        let msg = refusal(&scope_for(VHH_YAML), vec![sub(0, 50, 'W')]);
        assert_eq!(msg, "chain 'H' position 50 is beyond the chain (length 20)");
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn a_repeated_position_names_the_label_the_task_wrote() {
        // Position 4 is inside loop1, so the pair passes every check until
        // the repeat.
        let msg = refusal(&scope_for(VHH_YAML), vec![sub(0, 4, 'W'), sub(0, 4, 'Y')]);
        assert_eq!(msg, "chain 'H' position 4 is substituted more than once");
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn with_two_binder_chains_the_refusal_names_the_chain_that_refused_it() {
        // The region sits on L, so position 0 of L is framework and must be
        // refused naming L. H is chain0 and L is chain1, so an id rendering
        // would send the reader to the wrong chain.
        let msg = refusal(&scope_for(PAIRED_YAML), vec![sub(1, 0, 'W')]);
        assert!(msg.contains("chain 'L'"), "got: {msg}");
        assert!(!msg.contains("'H'"), "got: {msg}");
        assert!(!msg.contains("chain1"), "got: {msg}");
    }

    #[test]
    fn a_parent_that_lacks_the_residue_names_the_label_the_task_wrote() {
        let scope = scope_for(VHH_YAML);
        let proposal = scope
            .propose(None, SkillId::PointMutation, vec![sub(0, 4, 'W')], "test")
            .expect("position 4 is mutable in the scope");
        let short = TaskConfig::from_yaml(SHORT_VHH_YAML)
            .expect("fixture parses")
            .to_scaffold()
            .expect("fixture resolves");
        let parent = BinderSequence::from_scaffold(&short);
        let msg = proposal
            .apply_to(&parent)
            .expect_err("the short parent has no residue at position 4")
            .to_string();
        assert_eq!(
            msg,
            "parent sequence does not fit the scaffold: no residue at chain 'H' position 4 in the parent sequence"
        );
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn a_substitution_on_a_target_chain_names_the_label_the_task_wrote() {
        // The target chain A is chain1 in VHH_YAML. The scope does not track
        // its residues, but its labels map carries every chain the task
        // declared, so the refusal must name 'A' and not render the id. The
        // chain holds twenty residues, so a length claim of 0 would be false
        // and the message must make none.
        let msg = refusal(&scope_for(VHH_YAML), vec![sub(1, 0, 'W')]);
        assert_eq!(
            msg,
            "chain 'A' is not a binder chain, so position 0 in it cannot be substituted; aim the proposal at a binder chain"
        );
        assert!(!msg.contains("chain1"), "got: {msg}");
    }

    #[test]
    fn a_chain_the_task_lacks_falls_back_to_the_id() {
        // Only a hand-built substitution can name a chain id the task never
        // declared. The scope carries no label for it, so the message renders
        // the id.
        let msg = refusal(&scope_for(VHH_YAML), vec![sub(7, 0, 'W')]);
        assert_eq!(
            msg,
            "chain 'chain7' is not a binder chain, so position 0 in it cannot be substituted; aim the proposal at a binder chain"
        );
    }
}
