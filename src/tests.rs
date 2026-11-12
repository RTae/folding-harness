//! Tests for the domain layer: the mutable-scope invariant, the one-based to
//! zero-based conversion, loss, gates and population admission.
//!
//! These pin behaviour the rest of the harness relies on. A change that makes
//! them pass by weakening an assertion is not a fix.

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
    assert_eq!(
        err,
        ProposalError::Immutable {
            chain,
            position: 29
        }
    );
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
    Candidate::founder(CandidateId(id), sequence).scored(metrics, &LossWeights::default())
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
        Err(RejectReason::Duplicate { of: CandidateId(1) })
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
fn a_restored_population_still_refuses_the_duplicates_it_recorded() {
    let (scaffold, scope) = config().resolve().expect("resolves");
    let founder = BinderSequence::from_scaffold(&scaffold);
    let chain = ChainId(0);
    let variant = scope
        .propose(
            None,
            SkillId::PointMutation,
            vec![Substitution {
                chain,
                position: 26,
                to: 'W',
            }],
            "test",
        )
        .expect("in scope")
        .apply_to(&founder)
        .expect("applies");

    let config = PopulationConfig {
        capacity: 8,
        min_hamming: 0,
        parent_temperature: 0.05,
    };
    let mut original = Population::new(config.clone());
    original
        .admit(scored(1, founder.clone(), 0.5))
        .expect("admitted");
    original
        .admit(scored(2, variant.clone(), 0.6))
        .expect("admitted");
    assert!(original.admit(scored(3, founder.clone(), 0.7)).is_err());

    let mut restored = Population::restore(
        config,
        original.members().to_vec(),
        original.admitted_total(),
        original.rejected_total(),
    );

    assert_eq!(restored.members(), original.members());
    assert_eq!(restored.admitted_total(), original.admitted_total());
    assert_eq!(restored.rejected_total(), original.rejected_total());
    // The rebuilt fingerprint map names the member the duplicate matches,
    // exactly as the population that recorded it would.
    assert_eq!(
        restored.admit(scored(4, founder.clone(), 0.9)),
        Err(RejectReason::Duplicate { of: CandidateId(1) })
    );
    assert_eq!(
        restored.admit(scored(5, variant.clone(), 0.9)),
        Err(RejectReason::Duplicate { of: CandidateId(2) })
    );
    // The variant scored higher ipTM, so it is the best; higher ipTM means
    // lower loss.
    assert_eq!(restored.best().map(|c| c.id), Some(CandidateId(2)));
}

#[test]
fn a_restored_history_rebuilds_the_skill_stats_and_the_stagnation_count() {
    let outcomes = [
        CycleOutcome {
            cycle: 1,
            skill: SkillId::PointMutation,
            proposed: 4,
            admitted: 2,
            rejected: 2,
            best_loss: Some(0.8),
            mean_loss: None,
        },
        CycleOutcome {
            cycle: 2,
            skill: SkillId::PointMutation,
            proposed: 4,
            admitted: 0,
            rejected: 4,
            best_loss: Some(0.8),
            mean_loss: None,
        },
        CycleOutcome {
            cycle: 3,
            skill: SkillId::RegionRedesign,
            proposed: 3,
            admitted: 1,
            rejected: 2,
            best_loss: Some(0.5),
            mean_loss: None,
        },
        CycleOutcome {
            cycle: 4,
            skill: SkillId::RegionRedesign,
            proposed: 3,
            admitted: 0,
            rejected: 3,
            best_loss: Some(0.6),
            mean_loss: None,
        },
    ];

    // Feed a history the way `run_cycle` does: one choice per cycle, then
    // the per-candidate notes, then the record.
    let mut history = SearchHistory::new();
    for outcome in outcomes {
        history.note_choice(outcome.skill);
        history.note_proposals(outcome.skill, outcome.proposed);
        for _ in 0..outcome.admitted {
            history.note_admitted(outcome.skill);
        }
        for _ in 0..outcome.rejected {
            history.note_rejected(outcome.skill);
        }
        history.record_cycle(outcome, 1e-3);
    }
    assert_eq!(history.cycles_without_improvement(), 1);

    let restored = SearchHistory::restore(history.records().to_vec());
    assert_eq!(restored.records(), history.records());
    assert_eq!(restored.stats(), history.stats());
    assert_eq!(restored.best_loss(), history.best_loss());
    assert_eq!(
        restored.cycles_without_improvement(),
        history.cycles_without_improvement()
    );
    assert_eq!(restored.loss_curve(), history.loss_curve());
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

#[test]
fn parent_selection_is_reproducible_from_a_seed() {
    let (scaffold, _) = config().resolve().expect("resolves");
    let founder = BinderSequence::from_scaffold(&scaffold);
    let mut population = Population::new(PopulationConfig {
        capacity: 8,
        min_hamming: 0,
        parent_temperature: 0.5,
    });
    for (index, iptm) in [0.4, 0.5, 0.6, 0.7].iter().enumerate() {
        let mut sequence = founder.clone();
        // Distinct sequences so none are rejected as duplicates.
        let mut metrics = Metrics::zeroed();
        metrics.iptm = *iptm;
        metrics.interface_contacts = 30;
        let candidate = Candidate::founder(CandidateId(index as u32 + 1), {
            sequence = sequence.clone();
            sequence.clone()
        })
        .scored(metrics, &LossWeights::default());
        // Only the first will be admitted (the rest are duplicates by
        // fingerprint), which is itself the behaviour we want from dedup.
        let _ = population.admit(candidate);
    }

    let draw = |seed: u64| {
        let mut rng = rng_from_seed(seed);
        (0..12)
            .map(|_| population.select_parent(&mut rng).map(|c| c.id))
            .collect::<Vec<_>>()
    };
    assert_eq!(draw(7), draw(7));
}
