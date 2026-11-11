//! Measuring whether a search policy works.
//!
//! The fixture backend plants an optimum derived from the run seed, so unlike
//! a real folding run this one has ground truth. A benchmark run can therefore
//! report **recovery**: the fraction of planted positions the best candidate
//! actually found. That is the number that says whether the router, the agent
//! and the population are doing anything, as opposed to producing a loss curve
//! that merely goes down.
//!
//! What this measures is the orchestration against a synthetic landscape. It
//! says nothing about whether a design would bind, and it cannot: the fixture
//! does not fold. Treat it the way you would a unit test for a search
//! algorithm, not as evidence about a molecule.
//!
//! The control is [`Policy::Random`]. A policy that cannot beat a uniform draw
//! over the legal skills is not earning its complexity.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::agent::{DesignAgent, HeuristicAgent, RandomAgent};
use crate::fold::FixtureBackend;
use crate::orchestrator::Orchestrator;
use crate::trace::TaskPaths;
use crate::{SkillId, TaskConfig, TaskId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Policy {
    /// The shipped policy: prefer the skill with the best admission rate, with
    /// a fixed exploration rate, and reach for region redesign when stalled.
    Heuristic,
    /// Uniform over the legal skills. The control.
    Random,
    /// Only point mutation, to isolate what region redesign contributes.
    PointOnly,
    /// No search at all: fold the founder and stop. The floor.
    FounderOnly,
}

impl Policy {
    pub const ALL: [Policy; 4] = [
        Policy::Heuristic,
        Policy::Random,
        Policy::PointOnly,
        Policy::FounderOnly,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Policy::Heuristic => "heuristic",
            Policy::Random => "random",
            Policy::PointOnly => "point-only",
            Policy::FounderOnly => "founder-only",
        }
    }

    pub fn parse(name: &str) -> Option<Policy> {
        Policy::ALL
            .iter()
            .copied()
            .find(|policy| policy.as_str() == name.trim().to_ascii_lowercase())
    }

    pub fn names() -> String {
        Policy::ALL
            .iter()
            .map(|policy| policy.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// One run of one policy at one seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub policy: String,
    pub seed: u64,
    pub best_loss: f64,
    /// Fraction of the planted optimum the best candidate recovered, 0..1.
    pub recovery: f64,
    pub folds: usize,
    pub cycles: u32,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub policy: String,
    pub runs: usize,
    pub mean_best_loss: f64,
    pub mean_recovery: f64,
    pub worst_recovery: f64,
    pub best_recovery: f64,
    pub mean_folds: f64,
    /// Planted positions recovered per thousand folds, so policies that spend
    /// different amounts of compute can be compared.
    pub recovery_per_1k_folds: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn summarise(policy: Policy, outcomes: &[Outcome]) -> Summary {
    let recoveries: Vec<f64> = outcomes.iter().map(|o| o.recovery).collect();
    let folds: Vec<f64> = outcomes.iter().map(|o| o.folds as f64).collect();
    let mean_folds = mean(&folds);
    let mean_recovery = mean(&recoveries);

    Summary {
        policy: policy.as_str().to_string(),
        runs: outcomes.len(),
        mean_best_loss: mean(&outcomes.iter().map(|o| o.best_loss).collect::<Vec<_>>()),
        mean_recovery,
        worst_recovery: recoveries.iter().copied().fold(f64::INFINITY, f64::min),
        best_recovery: recoveries.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean_folds,
        recovery_per_1k_folds: if mean_folds > 0.0 {
            mean_recovery * 1000.0 / mean_folds
        } else {
            0.0
        },
    }
}

/// Run one policy at one seed and score it against the planted optimum.
pub async fn run_once(
    base: &TaskConfig,
    policy: Policy,
    seed: u64,
    cycles: u32,
    root: &std::path::Path,
) -> Result<Outcome, crate::orchestrator::RunError> {
    let mut config = base.clone();
    config.design.seed = seed;
    config.design.cycles = if policy == Policy::FounderOnly {
        1
    } else {
        cycles
    };
    if policy == Policy::PointOnly {
        config.design.skills = vec![SkillId::PointMutation];
    }

    let (scaffold, scope) = config.resolve()?;
    let backend = Arc::new(FixtureBackend::planted(&scaffold, &scope, seed));

    let agent: Box<dyn DesignAgent> = match policy {
        Policy::Random => Box::new(RandomAgent::new(seed)),
        _ => Box::new(HeuristicAgent::new(seed)),
    };

    let task_id = TaskId(format!("{}-{seed}", policy.as_str()));
    let paths = TaskPaths::new(root.join(&task_id.0));

    let started = Instant::now();
    let mut orchestrator = Orchestrator::prepare(
        config.clone(),
        Arc::clone(&backend) as Arc<dyn crate::fold::FoldBackend>,
        agent,
        paths,
        task_id,
    )
    .await?;

    // The founder-only control folds the starting sequence and stops, so its
    // recovery is whatever the founder happened to have.
    let summary = if policy == Policy::FounderOnly {
        orchestrator.seed_founder_only().await?
    } else {
        orchestrator.run().await?
    };

    let best = summary.best.as_ref();
    let labelled: Vec<(String, String)> = scaffold
        .binder_chains()
        .map(|chain| {
            let sequence = best
                .and_then(|candidate| candidate.sequence.chain(chain.id))
                .unwrap_or(chain.sequence.as_str())
                .to_string();
            (chain.label.clone(), sequence)
        })
        .collect();

    Ok(Outcome {
        policy: policy.as_str().to_string(),
        seed,
        best_loss: best
            .and_then(|candidate| candidate.loss)
            .unwrap_or(f64::NAN),
        recovery: backend.recovery(&labelled).unwrap_or(0.0),
        folds: summary.folds,
        cycles: summary.cycles_completed,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// Run every policy across every seed.
pub async fn sweep(
    base: &TaskConfig,
    policies: &[Policy],
    seeds: &[u64],
    cycles: u32,
    root: &std::path::Path,
    mut progress: impl FnMut(&Outcome),
) -> Result<Vec<Outcome>, crate::orchestrator::RunError> {
    let mut outcomes = Vec::new();
    for policy in policies {
        for seed in seeds {
            let outcome = run_once(base, *policy, *seed, cycles, root).await?;
            progress(&outcome);
            outcomes.push(outcome);
        }
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = r#"
name: bench-test
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
target:
  chains:
    - label: A
      sequence: MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ
design:
  cycles: 6
  batch_size: 4
gates:
  min_iptm: 0.2
  min_interface_plddt: 40.0
  max_pae_interface: 30.0
  max_clashes: 2
  min_interface_contacts: 5
  max_hydrophobic_patch: 0.9
fold:
  model: fixture
  execution_mode: fixture
"#;

    fn temp_root(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fh-bench-{name}-{}", crate::trace::now_ms()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn policy_names_round_trip() {
        for policy in Policy::ALL {
            assert_eq!(Policy::parse(policy.as_str()), Some(policy));
        }
        assert_eq!(Policy::parse("HEURISTIC"), Some(Policy::Heuristic));
        assert_eq!(Policy::parse("nonsense"), None);
    }

    #[tokio::test]
    async fn a_search_recovers_more_of_the_planted_optimum_than_not_searching() {
        let config = TaskConfig::from_yaml(YAML).expect("parses");
        let root = temp_root("recovery");

        let searched = run_once(&config, Policy::Heuristic, 11, 10, &root)
            .await
            .expect("runs");
        let floor = run_once(&config, Policy::FounderOnly, 11, 10, &root)
            .await
            .expect("runs");

        // The founder is planted away from the optimum by construction, so it
        // recovers nothing and the search has to find the positions itself.
        assert_eq!(floor.recovery, 0.0, "the founder should start at zero");
        assert!(floor.folds < searched.folds);
        assert!(
            searched.recovery > floor.recovery,
            "search recovered {:.3}, founder {:.3}",
            searched.recovery,
            floor.recovery
        );
        assert!(searched.best_loss < floor.best_loss);

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_benchmark_is_reproducible_from_its_seed() {
        let config = TaskConfig::from_yaml(YAML).expect("parses");
        let root = temp_root("repro");

        let first = run_once(&config, Policy::Heuristic, 5, 6, &root)
            .await
            .expect("runs");
        let second = run_once(&config, Policy::Heuristic, 5, 6, &root)
            .await
            .expect("runs");

        assert_eq!(first.recovery, second.recovery);
        assert_eq!(first.best_loss, second.best_loss);
        assert_eq!(first.folds, second.folds);

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_sweep_reports_one_outcome_per_policy_and_seed() {
        let config = TaskConfig::from_yaml(YAML).expect("parses");
        let root = temp_root("sweep");
        let policies = [Policy::Heuristic, Policy::Random];
        let seeds = [1u64, 2];

        let mut seen = 0;
        let outcomes = sweep(&config, &policies, &seeds, 5, &root, |_| seen += 1)
            .await
            .expect("sweeps");

        assert_eq!(outcomes.len(), 4);
        assert_eq!(seen, 4);

        let heuristic: Vec<Outcome> = outcomes
            .iter()
            .filter(|o| o.policy == "heuristic")
            .cloned()
            .collect();
        let summary = summarise(Policy::Heuristic, &heuristic);
        assert_eq!(summary.runs, 2);
        assert!(summary.mean_recovery >= summary.worst_recovery);
        assert!(summary.best_recovery >= summary.mean_recovery);
        assert!(summary.recovery_per_1k_folds > 0.0);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn summarising_nothing_does_not_divide_by_zero() {
        let summary = summarise(Policy::Random, &[]);
        assert_eq!(summary.runs, 0);
        assert_eq!(summary.recovery_per_1k_folds, 0.0);
        assert_eq!(summary.mean_folds, 0.0);
    }
}
