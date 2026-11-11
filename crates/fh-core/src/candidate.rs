//! A candidate: one binder sequence, where it came from, and how it scored.

use serde::{Deserialize, Serialize};

use crate::ids::CandidateId;
use crate::metrics::{loss, LossWeights, Metrics};
use crate::proposal::{Proposal, ProposalError, SkillId, Substitution};
use crate::scaffold::BinderSequence;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The sequence given in the task config, folded as cycle 0.
    Founder,
    /// Produced by a design skill.
    Skill(SkillId),
}

impl Origin {
    pub fn label(&self) -> &'static str {
        match self {
            Origin::Founder => "founder",
            Origin::Skill(skill) => skill.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub parent: Option<CandidateId>,
    pub cycle: u32,
    pub origin: Origin,
    pub sequence: BinderSequence,
    /// The substitutions that produced this candidate from its parent. Empty
    /// for the founder.
    pub substitutions: Vec<Substitution>,
    pub rationale: String,
    pub metrics: Option<Metrics>,
    pub loss: Option<f64>,
}

impl Candidate {
    pub fn founder(id: CandidateId, sequence: BinderSequence) -> Self {
        Candidate {
            id,
            parent: None,
            cycle: 0,
            origin: Origin::Founder,
            sequence,
            substitutions: Vec::new(),
            rationale: "input sequence from task config".to_string(),
            metrics: None,
            loss: None,
        }
    }

    /// The only way to derive a candidate from another: through a validated
    /// [`Proposal`], which already proved every substitution is in scope.
    pub fn from_proposal(
        id: CandidateId,
        cycle: u32,
        proposal: &Proposal,
        parent_sequence: &BinderSequence,
    ) -> Result<Self, ProposalError> {
        let sequence = proposal.apply_to(parent_sequence)?;
        Ok(Candidate {
            id,
            parent: proposal.parent(),
            cycle,
            origin: Origin::Skill(proposal.skill()),
            sequence,
            substitutions: proposal.substitutions().to_vec(),
            rationale: proposal.rationale().to_string(),
            metrics: None,
            loss: None,
        })
    }

    pub fn scored(mut self, metrics: Metrics, weights: &LossWeights) -> Self {
        self.loss = Some(loss(&metrics, weights));
        self.metrics = Some(metrics);
        self
    }

    pub fn loss_or_inf(&self) -> f64 {
        self.loss.unwrap_or(f64::INFINITY)
    }

    pub fn is_scored(&self) -> bool {
        self.loss.is_some() && self.metrics.is_some()
    }

    /// `c0012 point_mutation loss 0.2314` — one line for tables and logs.
    pub fn summary(&self) -> String {
        match self.loss {
            Some(l) => format!("{} {} loss {:.4}", self.id, self.origin.label(), l),
            None => format!("{} {} unscored", self.id, self.origin.label()),
        }
    }
}
