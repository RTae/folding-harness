//! Fold backends.
//!
//! Everything above this crate talks to [`FoldBackend`], never to a model. The
//! three implementations shipped here are the deterministic [`fixture`] (no
//! GPU, no network), the [`http`] client for a real compute service, and —
//! behind the `server` feature — a [`server`] that puts any backend behind the
//! `fh.fold/v1` HTTP contract.

pub mod fixture;
pub mod http;

pub mod server;

use async_trait::async_trait;

use crate::contract::{
    CaResidue, ChainSpec, ConfidenceReport, FoldResult, Health, JobSpec, MoleculeKind,
};
use crate::gate::RejectReason;
use crate::{BinderSequence, ChainRole, FoldConfig, Metrics, Scaffold};

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

/// Outcome of checking a returned structure against the sequence the harness
/// sent. The three cases must stay distinguishable: a service that returned no
/// `ca_trace` at all has not passed the check, it has given nothing to check.
#[derive(Debug, Clone, PartialEq)]
pub enum Verification {
    /// Every chain in the trace carries the residue the harness sent.
    Matched,
    /// The service reported back a different sequence.
    Mismatch(RejectReason),
    /// The result holds no `ca_trace`, so no residue could be compared.
    /// Carries why. This is never a success.
    NothingToCheck { reason: String },
}

/// Check the `ca_trace` a service reported back against the sequence the
/// harness sent: the candidate's [`BinderSequence`] for binder chains, the
/// scaffold for target chains.
///
/// `CaResidue::chain_index` maps onto `scaffold.chains[i]`: [`job_spec_for`]
/// pushes chains in scaffold order, and the service reports each residue's
/// position in `job.chains`. `CaResidue::index` is contractual: zero-based
/// within the chain, so each chain's sorted indices must be exactly 0..n and
/// a renumbered trace is a mismatch even when every residue letter matches.
/// Pure function, no I/O, like its two neighbours.
pub fn verify_output(
    scaffold: &Scaffold,
    sequence: &BinderSequence,
    result: &FoldResult,
) -> Verification {
    let payload = &result.structure;

    if payload.ca_trace.is_empty() {
        let reason = match (&payload.inline, &payload.artifact_url) {
            (Some(_), _) => {
                "the service inlined the full structure and sent no ca_trace".to_string()
            }
            (_, Some(url)) => {
                format!("the structure sits behind artifact_url {url} and no ca_trace came with it")
            }
            (None, None) => "the service sent no ca_trace".to_string(),
        };
        return Verification::NothingToCheck { reason };
    }

    if let Some(residue) = payload
        .ca_trace
        .iter()
        .find(|residue| usize::from(residue.chain_index) >= scaffold.chains.len())
    {
        return Verification::Mismatch(RejectReason::SequenceMismatch {
            chain: format!("index {}", residue.chain_index),
            detail: "the ca_trace references a chain the job did not send".to_string(),
        });
    }

    for (chain_index, chain) in scaffold.chains.iter().enumerate() {
        let expected = match chain.role {
            ChainRole::Binder => {
                // `BinderSequence::from_scaffold` fills every binder chain,
                // so a chain missing here is a harness bug, not a service
                // state. Say so in debug builds rather than let the fallback
                // to the scaffold's starting sequence pass unnoticed.
                debug_assert!(
                    sequence.chain(chain.id).is_some(),
                    "binder chain {} is missing from the candidate sequence",
                    chain.label
                );
                sequence.chain(chain.id).unwrap_or(chain.sequence.as_str())
            }
            ChainRole::Target => chain.sequence.as_str(),
        };

        let mut returned: Vec<&CaResidue> = payload
            .ca_trace
            .iter()
            .filter(|residue| usize::from(residue.chain_index) == chain_index)
            .collect();
        returned.sort_by_key(|residue| residue.index);

        if returned.is_empty() {
            // A dropped binder is a mismatch: the harness is designing that
            // chain. A service may legitimately trace only what it designed,
            // so an absent target chain is simply not compared.
            if chain.role == ChainRole::Binder {
                return Verification::Mismatch(RejectReason::SequenceMismatch {
                    chain: chain.label.clone(),
                    detail: "the service dropped this designed chain from the ca_trace".to_string(),
                });
            }
            continue;
        }

        let sent: Vec<char> = expected.chars().collect();
        if returned.len() != sent.len() {
            return Verification::Mismatch(RejectReason::SequenceMismatch {
                chain: chain.label.clone(),
                detail: format!(
                    "the harness sent {} residues, the service returned {}",
                    sent.len(),
                    returned.len()
                ),
            });
        }

        // The contract pins `CaResidue::index` to the zero-based position
        // within the chain, so the sorted indices must be exactly 0..n. A
        // service that shifted or duplicated indices is caught here even when
        // every residue letter is correct.
        let indices: Vec<u32> = returned.iter().map(|residue| residue.index).collect();
        let positions: Vec<u32> = (0..returned.len() as u32).collect();
        if indices != positions {
            return Verification::Mismatch(RejectReason::SequenceMismatch {
                chain: chain.label.clone(),
                detail: format!(
                    "the service renumbered this chain: indices were {indices:?}, expected {positions:?}"
                ),
            });
        }

        for (position, (want, got)) in sent.iter().zip(&returned).enumerate() {
            if *want != got.residue {
                return Verification::Mismatch(RejectReason::SequenceMismatch {
                    chain: chain.label.clone(),
                    detail: format!(
                        "position {}: the harness sent {want}, the service returned {}",
                        position + 1,
                        got.residue
                    ),
                });
            }
        }
    }

    Verification::Matched
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{StructureFormat, StructurePayload};
    use crate::scaffold::{BinderTopology, Chain};
    use crate::ChainId;

    fn scaffold() -> Scaffold {
        Scaffold {
            topology: BinderTopology::Vhh,
            chains: vec![
                Chain {
                    id: ChainId(0),
                    label: "H".to_string(),
                    sequence: "QVQL".to_string(),
                    role: ChainRole::Binder,
                },
                Chain {
                    id: ChainId(1),
                    label: "A".to_string(),
                    sequence: "MKTV".to_string(),
                    role: ChainRole::Target,
                },
            ],
            regions: Vec::new(),
            fixed: Vec::new(),
        }
    }

    /// A ca_trace in the shape `FixtureBackend::ca_trace` emits: one residue
    /// per character, chain_index in chain order, index in residue order.
    fn trace(chains: &[&str]) -> Vec<CaResidue> {
        chains
            .iter()
            .enumerate()
            .flat_map(|(chain_index, sequence)| {
                sequence
                    .chars()
                    .enumerate()
                    .map(move |(index, residue)| CaResidue {
                        chain_index: chain_index as u8,
                        index: index as u32,
                        residue,
                        xyz: [0.0; 3],
                        plddt: 70.0,
                        designed: chain_index == 0,
                    })
            })
            .collect()
    }

    fn result(ca_trace: Vec<CaResidue>) -> FoldResult {
        FoldResult {
            job_id: "job".to_string(),
            model: "fixture".to_string(),
            confidence: ConfidenceReport {
                iptm: 0.5,
                ptm: 0.5,
                plddt_mean: 70.0,
                interface_plddt: 70.0,
                pae_interface: 10.0,
                interface_contacts: 20,
                clashes: 0,
                hydrophobic_patch: 0.3,
                net_charge: 0.0,
                plddt: Vec::new(),
                atom_confidence: true,
            },
            structure: StructurePayload {
                format: StructureFormat::CaTrace,
                inline: None,
                artifact_url: None,
                ca_trace,
            },
            elapsed_ms: 0,
        }
    }

    fn mismatch_reason(check: Verification) -> String {
        match check {
            Verification::Mismatch(reason) => reason.to_string(),
            other => panic!("expected a mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_correct_trace_passes() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let check = verify_output(&scaffold, &sequence, &result(trace(&["QVQL", "MKTV"])));
        assert_eq!(check, Verification::Matched);
    }

    #[test]
    fn a_substituted_residue_is_caught_and_named() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut ca_trace = trace(&["QVQL", "MKTV"]);
        ca_trace[1].residue = 'A';

        let reason = mismatch_reason(verify_output(&scaffold, &sequence, &result(ca_trace)));
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("position 2"), "{reason}");
        assert!(reason.contains("sent V"), "{reason}");
        assert!(reason.contains("returned A"), "{reason}");
    }

    #[test]
    fn a_truncated_trace_is_caught() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);

        let reason = mismatch_reason(verify_output(
            &scaffold,
            &sequence,
            &result(trace(&["QVQ", "MKTV"])),
        ));
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("sent 4 residues"), "{reason}");
        assert!(reason.contains("returned 3"), "{reason}");
    }

    #[test]
    fn a_missing_designed_chain_is_caught() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);

        let reason = mismatch_reason(verify_output(
            &scaffold,
            &sequence,
            &result(trace(&["", "MKTV"])),
        ));
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("dropped this designed chain"), "{reason}");
    }

    /// The lenient half of the asymmetry `a_missing_designed_chain_is_caught`
    /// pins: a service may trace only what it designed, so an absent target
    /// chain is not compared rather than rejected.
    #[test]
    fn an_absent_target_chain_is_not_a_mismatch() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let check = verify_output(&scaffold, &sequence, &result(trace(&["QVQL", ""])));
        assert_eq!(check, Verification::Matched);
    }

    /// Residues correct, order preserved, every index shifted by one. Without
    /// the index-coverage check this passes as a positional match.
    #[test]
    fn a_uniform_index_shift_is_caught() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut ca_trace = trace(&["QVQL", "MKTV"]);
        for residue in ca_trace.iter_mut().filter(|r| r.chain_index == 0) {
            residue.index += 1;
        }

        let reason = mismatch_reason(verify_output(&scaffold, &sequence, &result(ca_trace)));
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("renumbered"), "{reason}");
        assert!(reason.contains("[1, 2, 3, 4]"), "{reason}");
    }

    /// The reviewer's case: indices [0, 0, 1, 2] with residues Q, V, Q, L
    /// against a sent QVQL. Length and letters both agree, so only the index
    /// coverage catches it.
    #[test]
    fn a_duplicated_index_is_caught() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut ca_trace = trace(&["QVQL", "MKTV"]);
        ca_trace[1].index = 0;
        ca_trace[2].index = 1;
        ca_trace[3].index = 2;

        let reason = mismatch_reason(verify_output(&scaffold, &sequence, &result(ca_trace)));
        assert!(reason.contains("chain H"), "{reason}");
        assert!(reason.contains("[0, 0, 1, 2]"), "{reason}");
    }

    #[test]
    fn a_trace_referencing_a_chain_the_job_never_sent_is_caught() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut ca_trace = trace(&["QVQL", "MKTV"]);
        ca_trace.push(CaResidue {
            chain_index: 2,
            index: 0,
            residue: 'Q',
            xyz: [0.0; 3],
            plddt: 70.0,
            designed: false,
        });

        let reason = mismatch_reason(verify_output(&scaffold, &sequence, &result(ca_trace)));
        assert!(reason.contains("index 2"), "{reason}");
        assert!(reason.contains("did not send"), "{reason}");
    }

    #[test]
    fn an_empty_ca_trace_is_nothing_to_check_not_success() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let check = verify_output(&scaffold, &sequence, &result(Vec::new()));
        match check {
            Verification::NothingToCheck { reason } => {
                assert!(reason.contains("no ca_trace"), "{reason}");
            }
            other => panic!("expected nothing to check, got {other:?}"),
        }
    }

    #[test]
    fn an_inlined_structure_without_a_trace_is_nothing_to_check() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut fold = result(Vec::new());
        fold.structure.inline = Some("data_mmCIF".to_string());
        let check = verify_output(&scaffold, &sequence, &fold);
        match check {
            Verification::NothingToCheck { reason } => {
                assert!(reason.contains("inlined"), "{reason}");
            }
            other => panic!("expected nothing to check, got {other:?}"),
        }
    }

    #[test]
    fn a_structure_behind_an_artifact_url_without_a_trace_is_nothing_to_check() {
        let scaffold = scaffold();
        let sequence = BinderSequence::from_scaffold(&scaffold);
        let mut fold = result(Vec::new());
        fold.structure.artifact_url = Some("https://gpu-1:8080/artifacts/job-7.cif".to_string());
        let check = verify_output(&scaffold, &sequence, &fold);
        match check {
            Verification::NothingToCheck { reason } => {
                assert!(reason.contains("artifact_url"), "{reason}");
                assert!(reason.contains("job-7.cif"), "{reason}");
            }
            other => panic!("expected nothing to check, got {other:?}"),
        }
    }

    /// The check compares against what the harness sent, which for a binder is
    /// the candidate's sequence, not the scaffold's starting one.
    #[test]
    fn the_sent_sequence_decides_not_the_scaffold() {
        let scaffold = scaffold();
        let mut sequence = BinderSequence::from_scaffold(&scaffold);
        assert!(sequence.replace(ChainId(0), 1, 'W'));

        let stale = verify_output(&scaffold, &sequence, &result(trace(&["QVQL", "MKTV"])));
        assert!(matches!(stale, Verification::Mismatch(_)), "{stale:?}");

        let sent = verify_output(&scaffold, &sequence, &result(trace(&["QWQL", "MKTV"])));
        assert_eq!(sent, Verification::Matched);
    }
}
