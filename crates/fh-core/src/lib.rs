//! Domain types for the folding harness.
//!
//! This crate holds the vocabulary of a design run and nothing else: no I/O, no
//! HTTP, no model calls. Everything above it — skills, agent, orchestrator,
//! terminal UI — is written against these types.
//!
//! The property worth knowing about: a [`Proposal`] can only be created by
//! [`MutableScope::propose`], which checks every substitution against the
//! scope, and a [`Candidate`] can only descend from another through a
//! `Proposal`. There is no public path that edits a sequence directly, and
//! `Proposal` has no `Deserialize` impl, so one cannot be smuggled in through a
//! config file or an API payload. "Framework residues are never mutated" is
//! therefore a property of the types rather than a convention.

pub mod candidate;
pub mod config;
pub mod gate;
pub mod ids;
pub mod metrics;
pub mod population;
pub mod proposal;
pub mod scaffold;
pub mod search;

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
pub use search::{CycleRecord, SearchHistory, SearchState, SkillStats};

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = r#"
name: demo
binder:
  topology: vhh
  chains:
    - label: H
      sequence: QVQLVESGGGLVQAGGSLRLSCAASGYTFTSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGSYYDYWGQGTQVTVSS
  regions:
    - name: loop1
      chain: H
      start: 26
      end: 33
    - name: loop3
      chain: H
      start: 99
      end: 108
  fixed_residues: ["H:30"]
target:
  chains:
    - label: A
      sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
design:
  cycles: 3
  batch_size: 2
"#;

    fn config() -> TaskConfig {
        TaskConfig::from_yaml(YAML).expect("fixture config parses")
    }

    #[test]
    fn yaml_positions_are_one_based_inclusive() {
        let (_, scope) = config().resolve().expect("resolves");
        let loop1 = scope.region("loop1").expect("loop1 exists");
        // YAML said 26..33 inclusive, so internally 25..33 — eight positions.
        assert_eq!(loop1.start, 25);
        assert_eq!(loop1.end, 33);
        assert_eq!(loop1.len(), 8);
    }

    #[test]
    fn a_pinned_residue_overrides_its_region() {
        let (_, scope) = config().resolve().expect("resolves");
        let chain = ChainId(0);
        // H:30 one-based is index 29, inside loop1 (25..33) but pinned.
        assert!(scope.is_mutable(chain, 28));
        assert!(!scope.is_mutable(chain, 29));
        assert!(scope.is_mutable(chain, 30));
        assert_eq!(scope.region_positions("loop1").len(), 7);
    }

    #[test]
    fn framework_positions_cannot_be_proposed() {
        let (scaffold, scope) = config().resolve().expect("resolves");
        let _ = BinderSequence::from_scaffold(&scaffold);
        let chain = ChainId(0);

        // Position 0 is framework: outside every declared region.
        let err = scope
            .propose(
                None,
                SkillId::PointMutation,
                vec![Substitution {
                    chain,
                    position: 0,
                    to: 'W',
                }],
                "test",
            )
            .expect_err("framework position must be refused");
        assert_eq!(err, ProposalError::Immutable { chain, position: 0 });

        // The pinned position inside a region is refused the same way.
        let err = scope
            .propose(
                None,
                SkillId::PointMutation,
                vec![Substitution {
                    chain,
                    position: 29,
                    to: 'W',
                }],
                "test",
            )
            .expect_err("pinned position must be refused");
        assert_eq!(err, ProposalError::Immutable { chain, position: 29 });
    }

    #[test]
    fn in_scope_proposal_applies_and_changes_only_that_position() {
        let (scaffold, scope) = config().resolve().expect("resolves");
        let parent = BinderSequence::from_scaffold(&scaffold);
        let chain = ChainId(0);
        let proposal = scope
            .propose(
                None,
                SkillId::PointMutation,
                vec![Substitution {
                    chain,
                    position: 28,
                    to: 'W',
                }],
                "test",
            )
            .expect("in-scope proposal is accepted");

        let child = proposal.apply_to(&parent).expect("applies");
        assert_eq!(parent.hamming(&child), Some(1));
        assert_eq!(child.chain(chain).unwrap().chars().nth(28), Some('W'));
        assert_eq!(parent.total_len(), child.total_len());
    }

    #[test]
    fn non_standard_residues_and_repeats_are_refused() {
        let (_, scope) = config().resolve().expect("resolves");
        let chain = ChainId(0);
        let err = scope
            .propose(
                None,
                SkillId::PointMutation,
                vec![Substitution {
                    chain,
                    position: 28,
                    to: 'X',
                }],
                "test",
            )
            .expect_err("X is not a standard residue");
        assert_eq!(err, ProposalError::NonStandardResidue { to: 'X' });

        let err = scope
            .propose(
                None,
                SkillId::PointMutation,
                vec![
                    Substitution {
                        chain,
                        position: 28,
                        to: 'W',
                    },
                    Substitution {
                        chain,
                        position: 28,
                        to: 'Y',
                    },
                ],
                "test",
            )
            .expect_err("one position cannot be substituted twice");
        assert_eq!(
            err,
            ProposalError::DuplicatePosition {
                chain,
                position: 28
            }
        );
    }

    #[test]
    fn a_scaffold_with_no_mutable_positions_is_refused() {
        let mut cfg = config();
        cfg.binder.regions = vec![RegionSpec {
            name: "tiny".to_string(),
            chain: "H".to_string(),
            start: 30,
            end: 30,
        }];
        cfg.binder.fixed_residues = vec!["H:30".to_string()];
        let err = cfg.resolve().expect_err("nothing left to design");
        assert!(matches!(
            err,
            ConfigError::Scaffold(ScaffoldError::NothingMutable)
        ));
    }

    #[test]
    fn unknown_config_keys_fail_at_load() {
        let text = YAML.replace("batch_size: 2", "batch_sze: 2");
        let err = TaskConfig::from_yaml(&text).expect_err("a typo must not be silently defaulted");
        assert!(matches!(err, ConfigError::Yaml(_)), "got {err:?}");
    }

    #[test]
    fn loss_improves_as_confidence_rises() {
        let weights = LossWeights::default();
        let mut poor = Metrics::zeroed();
        poor.iptm = 0.30;
        poor.interface_plddt = 55.0;
        poor.pae_interface = 20.0;
        let mut good = poor.clone();
        good.iptm = 0.85;
        good.interface_plddt = 88.0;
        good.pae_interface = 6.0;
        assert!(loss(&good, &weights) < loss(&poor, &weights));

        let breakdown = loss_breakdown(&good, &weights);
        assert!((breakdown.total - loss(&good, &weights)).abs() < 1e-12);
        assert_eq!(breakdown.terms.len(), 8);
    }

    fn scored(id: u32, sequence: BinderSequence, iptm: f64) -> Candidate {
        let mut metrics = Metrics::zeroed();
        metrics.iptm = iptm;
        metrics.interface_plddt = 80.0;
        metrics.pae_interface = 8.0;
        metrics.interface_contacts = 40;
        let mut candidate = Candidate::founder(CandidateId(id), sequence);
        candidate = candidate.scored(metrics, &LossWeights::default());
        candidate
    }

    #[test]
    fn population_rejects_duplicates_and_evicts_the_worst() {
        let (scaffold, scope) = config().resolve().expect("resolves");
        let founder = BinderSequence::from_scaffold(&scaffold);
        let chain = ChainId(0);

        let mut population = Population::new(PopulationConfig {
            capacity: 2,
            min_hamming: 0,
            parent_temperature: 0.05,
        });

        let variant = |position: usize, to: char| {
            scope
                .propose(
                    None,
                    SkillId::PointMutation,
                    vec![Substitution {
                        chain,
                        position,
                        to,
                    }],
                    "test",
                )
                .expect("in scope")
                .apply_to(&founder)
                .expect("applies")
        };

        assert_eq!(
            population.admit(scored(1, founder.clone(), 0.50)),
            Ok(AdmissionOutcome::Admitted)
        );
        assert_eq!(
            population.admit(scored(1, founder.clone(), 0.90)),
            Err(RejectReason::Duplicate {
                of: CandidateId(1)
            })
        );

        assert_eq!(
            population.admit(scored(2, variant(26, 'W'), 0.60)),
            Ok(AdmissionOutcome::Admitted)
        );
        assert_eq!(population.len(), 2);

        // At capacity, a better candidate evicts the worst — here c0001.
        assert_eq!(
            population.admit(scored(3, variant(27, 'Y'), 0.95)),
            Ok(AdmissionOutcome::Evicted(CandidateId(1)))
        );
        assert_eq!(population.len(), 2);
        assert_eq!(population.best().map(|c| c.id), Some(CandidateId(3)));

        // A worse one is refused rather than displacing anything.
        assert!(matches!(
            population.admit(scored(4, variant(28, 'D'), 0.10)),
            Err(RejectReason::WorseThanPopulation { .. })
        ));
        assert_eq!(population.len(), 2);
    }

    #[test]
    fn gates_name_the_failing_field() {
        let gates = Gates::default();
        let mut metrics = Metrics::zeroed();
        metrics.iptm = 0.1;
        metrics.interface_plddt = 90.0;
        metrics.pae_interface = 5.0;
        metrics.interface_contacts = 30;
        match gates.check(&metrics) {
            Err(RejectReason::Gate { field, .. }) => assert_eq!(field, "iptm"),
            other => panic!("expected an iptm gate failure, got {other:?}"),
        }
    }
}
