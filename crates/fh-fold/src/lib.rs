//! Fold backends.
//!
//! Everything above this crate talks to [`FoldBackend`], never to a model. The
//! three implementations shipped here are the deterministic [`fixture`] (no
//! GPU, no network), the [`http`] client for a real compute service, and —
//! behind the `server` feature — a [`server`] that puts any backend behind the
//! `fh.fold/v1` HTTP contract.

pub mod fixture;
pub mod http;
#[cfg(feature = "server")]
pub mod server;

use async_trait::async_trait;

use fh_contract::{ChainSpec, ConfidenceReport, FoldResult, Health, JobSpec, MoleculeKind};
use fh_core::{BinderSequence, ChainRole, FoldConfig, Metrics, Scaffold};

pub use fixture::FixtureBackend;
pub use http::HttpBackend;

#[derive(Debug, thiserror::Error)]
pub enum FoldError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("service returned HTTP {status}: {body}")]
    Service { status: u16, body: String },
    #[error("protocol mismatch: service speaks {service}, this harness speaks {harness}")]
    Protocol { service: String, harness: String },
    #[error("fold job failed: {0}")]
    Rejected(String),
    #[error("timed out after {secs}s waiting for job {job_id}")]
    Timeout { job_id: String, secs: u64 },
    #[error("malformed response: {0}")]
    Malformed(String),
    #[error("backend does not support this request: {0}")]
    Unsupported(String),
}

#[async_trait]
pub trait FoldBackend: Send + Sync {
    /// Short name for logs and the run header, e.g. `fixture` or
    /// `service(http://gpu-1:8080)`.
    fn name(&self) -> &str;

    async fn health(&self) -> Result<Health, FoldError>;

    async fn fold(&self, job: JobSpec) -> Result<FoldResult, FoldError>;
}

/// Wire confidence report to domain metrics. Mechanical by design: the two
/// structs are kept field-compatible so this never has to make a judgement.
pub fn metrics_from(report: &ConfidenceReport) -> Metrics {
    Metrics {
        iptm: report.iptm,
        ptm: report.ptm,
        plddt_mean: report.plddt_mean,
        interface_plddt: report.interface_plddt,
        pae_interface: report.pae_interface,
        interface_contacts: report.interface_contacts,
        clashes: report.clashes,
        hydrophobic_patch: report.hydrophobic_patch,
        net_charge: report.net_charge,
        plddt: report.plddt.clone(),
    }
}

/// Build the wire job for one candidate: the candidate's binder chains plus
/// the task's fixed target chains.
///
/// This is the only place the domain crosses into the wire format, so a change
/// to the contract shows up here rather than scattered across the orchestrator.
pub fn job_spec_for(
    name: &str,
    scaffold: &Scaffold,
    sequence: &BinderSequence,
    fold: &FoldConfig,
) -> JobSpec {
    let mut chains = Vec::with_capacity(scaffold.chains.len());
    for chain in &scaffold.chains {
        let (text, designed) = match chain.role {
            ChainRole::Binder => (
                sequence
                    .chain(chain.id)
                    .unwrap_or(chain.sequence.as_str())
                    .to_string(),
                true,
            ),
            ChainRole::Target => (chain.sequence.clone(), false),
        };
        chains.push(ChainSpec {
            label: chain.label.clone(),
            sequence: text,
            kind: MoleculeKind::Protein,
            designed,
        });
    }

    JobSpec {
        name: name.to_string(),
        model: fold.model.clone(),
        chains,
        seeds: if fold.seeds.is_empty() {
            vec![1]
        } else {
            fold.seeds.clone()
        },
        diffusion_samples: fold.diffusion_samples.max(1),
        diffusion_steps: fold.diffusion_steps.max(1),
        use_msa: fold.use_msa,
        need_atom_confidence: fold.need_atom_confidence,
    }
}
