//! A design harness in Rust.
//!
//! The harness owns everything on this side of a compute boundary:
//! configuration, the mutable scope of a design, the search cycle, hard gates,
//! the retained population, provenance, and inspection. The folding model sits
//! behind [`contract`], an HTTP job protocol, so nothing here touches a tensor
//! or a GPU.
//!
//! The property worth knowing about: a [`Proposal`] can only be created by
//! [`MutableScope::propose`], which checks every substitution against the
//! scope, and a [`Candidate`] can only descend from another through a
//! `Proposal`. No public path edits a sequence directly, and `Proposal` has no
//! `Deserialize` impl, so one cannot arrive from a config file or an API
//! payload. "Positions outside the mutable scope never change" is therefore a
//! property of the types rather than a convention.
//!
//! Module layout, bottom up:
//!
//! * [`ids`], [`scaffold`], [`proposal`], [`candidate`], [`metrics`], [`gate`],
//!   [`population`], [`search`], [`config`] — the domain. No I/O.
//! * [`contract`] — the wire types for `fh.fold/v1`.
//! * [`fold`] — backends: the trait, a deterministic fixture, an HTTP client,
//!   and a stub service.
//! * [`skills`] — design actions and the router that decides which are legal.
//! * [`agent`] — the decision layer.
//! * [`trace`] — the event log, task directory layout, and run manifest.
//! * [`orchestrator`] — the cycle that ties them together.
//! * [`tui`] — the terminal dashboard.

pub mod agent;
pub mod candidate;
pub mod config;
pub mod contract;
pub mod fold;
pub mod gate;
pub mod ids;
pub mod metrics;
pub mod orchestrator;
pub mod population;
pub mod proposal;
pub mod scaffold;
pub mod search;
pub mod skills;
pub mod trace;
pub mod tui;

pub use candidate::{Candidate, Origin};
pub use config::{
    BinderSpec, ChainSpec, ConfigError, DesignConfig, ExecutionMode, FoldConfig, RegionSpec,
    TargetSpec, TaskConfig,
};
pub use gate::{Gates, RejectReason};
pub use ids::{CandidateId, ChainId, TaskId};
pub use metrics::{loss, loss_breakdown, LossBreakdown, LossTerm, LossWeights, Metrics};
pub use population::{AdmissionOutcome, Population, PopulationConfig, PopulationStats};
pub use proposal::{Capability, Proposal, ProposalError, SkillId, Substitution};
pub use scaffold::{
    is_standard_aa, BinderSequence, BinderTopology, Chain, ChainRole, FixedResidue, MutableScope,
    Region, Scaffold, ScaffoldError, AMINO_ACIDS,
};
pub use search::{CycleOutcome, CycleRecord, SearchHistory, SearchState, SkillStats};

/// The seeded generator used everywhere a decision is random.
///
/// ChaCha8 rather than `rand`'s `StdRng`, whose output the `rand` crate
/// documents as unstable across releases: a seed recorded in a run manifest
/// would stop reproducing that run after a dependency bump. ChaCha's stream is
/// specified, so the manifest keeps its meaning.
///
/// Named `RunRng` to avoid colliding with the `rand::Rng` trait.
pub type RunRng = rand_chacha::ChaCha8Rng;

/// Build the run generator from a task seed.
pub fn rng_from_seed(seed: u64) -> RunRng {
    use rand::SeedableRng;
    RunRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests;
