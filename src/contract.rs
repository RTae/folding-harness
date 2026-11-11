//! The `fh.fold/v1` contract between the harness client and a compute service.
//!
//! The split this crate describes is the reason the harness can be written in
//! Rust at all: the client never touches a tensor or a GPU. It submits a job,
//! polls for a result, and reads back confidence numbers plus a structure. Any
//! service that speaks these types — a Python process wrapping a folding
//! model, a cluster gateway, or the in-process fixture — is a valid backend.
//!
//! This crate depends on `serde` and nothing else, so implementing the service
//! side does not drag in the harness.
//!
//! Protocol shape:
//!
//! ```text
//! GET  /v1/health          -> Health
//! POST /v1/fold            -> JobAccepted      (returns immediately)
//! GET  /v1/fold/{job_id}   -> JobStatus        (poll until terminal)
//! ```
//!
//! Folding takes minutes, so submission is asynchronous by design: a client
//! that dies mid-run can reattach to a job id rather than losing the work.

use serde::{Deserialize, Serialize};

/// Bump the minor version for additive changes, the major for breaking ones.
/// The service echoes it in [`Health::protocol`]; the client refuses a major
/// mismatch.
pub const PROTOCOL: &str = "fh.fold/v1";

pub mod routes {
    pub const HEALTH: &str = "/v1/health";
    pub const FOLD: &str = "/v1/fold";
    /// Axum-style path pattern for one job.
    pub const JOB: &str = "/v1/fold/:job_id";

    pub fn job(job_id: &str) -> String {
        format!("/v1/fold/{job_id}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MoleculeKind {
    #[default]
    Protein,
    Dna,
    Rna,
    Ligand,
}

/// One chain of the complex to predict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainSpec {
    pub label: String,
    pub sequence: String,
    #[serde(default)]
    pub kind: MoleculeKind,
    /// True for chains the harness is designing. A service may use this to
    /// decide what to report per-residue confidence for; it has no effect on
    /// the prediction itself.
    #[serde(default)]
    pub designed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    /// Human-readable, for service-side logs. Not an identifier.
    pub name: String,
    /// Backend model name. The harness passes this through without
    /// interpreting it, so a service can route between several models.
    pub model: String,
    pub chains: Vec<ChainSpec>,
    pub seeds: Vec<u64>,
    pub diffusion_samples: u32,
    pub diffusion_steps: u32,
    pub use_msa: bool,
    /// The interface terms of the harness loss need per-atom confidence. A
    /// service that cannot provide it must fail the job rather than silently
    /// return complex-level numbers.
    pub need_atom_confidence: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoldRequest {
    pub protocol: String,
    pub job: JobSpec,
}

impl FoldRequest {
    pub fn new(job: JobSpec) -> Self {
        FoldRequest {
            protocol: PROTOCOL.to_string(),
            job,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobAccepted {
    pub job_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_position: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
}

impl JobState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, JobState::Succeeded | JobState::Failed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: String,
    pub state: JobState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<FoldResult>,
}

/// Scores for one predicted complex. Field names match
/// `crate::metrics::Metrics` so the adapter between them is mechanical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceReport {
    pub iptm: f64,
    pub ptm: f64,
    pub plddt_mean: f64,
    pub interface_plddt: f64,
    pub pae_interface: f64,
    pub interface_contacts: u32,
    pub clashes: u32,
    pub hydrophobic_patch: f64,
    pub net_charge: f64,
    /// Per-residue confidence for designed chains, in chain then residue order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plddt: Vec<f32>,
    /// Whether per-atom confidence was actually computed.
    pub atom_confidence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureFormat {
    Cif,
    Pdb,
    /// Alpha-carbon trace only. Enough for the terminal viewer and for
    /// coarse geometry, and small enough to keep inline.
    CaTrace,
}

/// One alpha carbon. The harness keeps these so the terminal viewer can draw a
/// backbone without parsing mmCIF; the full structure stays in `inline` or
/// behind `artifact_url`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CaResidue {
    pub chain_index: u8,
    pub index: u32,
    pub residue: char,
    pub xyz: [f32; 3],
    pub plddt: f32,
    pub designed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructurePayload {
    pub format: StructureFormat,
    /// Full structure text, when the service chose to inline it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
    /// Where the full artifact can be fetched, when it is too large to inline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_url: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ca_trace: Vec<CaResidue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoldResult {
    pub job_id: String,
    pub model: String,
    pub confidence: ConfidenceReport,
    pub structure: StructurePayload,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mb: u64,
    /// Job id currently holding this device, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leased_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub protocol: String,
    pub service: String,
    pub version: String,
    /// Capability strings the harness router checks before offering a skill,
    /// e.g. `fold`, `inverse_folding`, `sequence_likelihood`.
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<GpuInfo>,
    pub queue_depth: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Compare the major version of two protocol strings, e.g. `fh.fold/v1`.
pub fn major_version(protocol: &str) -> Option<&str> {
    protocol.rsplit('/').next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_major_version_is_comparable() {
        assert_eq!(major_version(PROTOCOL), Some("v1"));
        assert_eq!(major_version("fh.fold/v2"), Some("v2"));
        assert_ne!(major_version("fh.fold/v2"), major_version(PROTOCOL));
    }

    #[test]
    fn job_route_round_trips() {
        assert_eq!(routes::job("abc123"), "/v1/fold/abc123");
    }
}
