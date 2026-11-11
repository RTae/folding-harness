//! A deterministic stand-in for a folding service.
//!
//! The fixture exists so the orchestration layer can be developed and tested
//! without a GPU, and so the terminal UI has something to draw. It is a test
//! double, not a model: it scores a sequence by how closely the mutable
//! positions match a **planted optimum** that the harness itself generated from
//! the run seed. That gives a search landscape with a known answer, which is
//! what makes an assertion like "the best loss improves over cycles" meaningful
//! in a test.
//!
//! Nothing here predicts anything about a real molecule, and the coordinates it
//! emits are a synthetic backbone, not a structure.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;

use crate::contract::{
    CaResidue, ConfidenceReport, FoldResult, Health, JobSpec, StructureFormat, StructurePayload,
    PROTOCOL,
};
use crate::{Capability, MutableScope, Scaffold, AMINO_ACIDS};

use super::{FoldBackend, FoldError};

/// Deterministic 64-bit mixer. Not for anything but reproducible test data.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn hash_text(text: &str, seed: u64) -> u64 {
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

const HYDROPHOBIC: [char; 8] = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'];

#[derive(Debug, Clone)]
struct Plant {
    /// The residue the fixture "wants" at each scored position.
    ideal: BTreeMap<usize, char>,
}

#[derive(Debug, Clone)]
pub struct FixtureBackend {
    /// Keyed by chain label, matching [`crate::contract::ChainSpec::label`].
    plants: BTreeMap<String, Plant>,
    seed: u64,
    latency: Duration,
    capabilities: Vec<Capability>,
    /// Fraction of jobs that fail outright, so recovery paths get exercised.
    failure_rate: f64,
    name: String,
}

impl Default for FixtureBackend {
    fn default() -> Self {
        FixtureBackend {
            plants: BTreeMap::new(),
            seed: 1,
            latency: Duration::ZERO,
            capabilities: vec![Capability::Fold],
            failure_rate: 0.0,
            name: "fixture".to_string(),
        }
    }
}

impl FixtureBackend {
    /// Plant an optimum derived from the scaffold's mutable scope.
    ///
    /// For every mutable position the fixture picks a target residue from the
    /// run seed. A candidate scores well as it matches more of them. Positions
    /// outside the scope are never scored, which is also a check on the rest of
    /// the harness: if a skill ever mutated a framework residue, the fixture
    /// would not reward it.
    pub fn planted(scaffold: &Scaffold, scope: &MutableScope, seed: u64) -> Self {
        let mut state = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
        let mut plants: BTreeMap<String, Plant> = BTreeMap::new();

        for chain in scaffold.binder_chains() {
            let mut ideal = BTreeMap::new();
            let current: Vec<char> = chain.sequence.chars().collect();
            for (chain_id, position) in scope.positions() {
                if chain_id != chain.id {
                    continue;
                }
                // Pick a residue that differs from the starting one, so the
                // founder is never already at the optimum.
                let mut pick = current.get(position).copied().unwrap_or('A');
                for _ in 0..8 {
                    let index = (splitmix64(&mut state) % AMINO_ACIDS.len() as u64) as usize;
                    let candidate = AMINO_ACIDS[index];
                    if Some(candidate) != current.get(position).copied() {
                        pick = candidate;
                        break;
                    }
                }
                ideal.insert(position, pick);
            }
            plants.insert(chain.label.clone(), Plant { ideal });
        }

        FixtureBackend {
            plants,
            seed,
            ..Default::default()
        }
    }

    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_capabilities(mut self, capabilities: Vec<Capability>) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Fraction of planted positions a job's designed chains match, 0..1.
    /// Returns 1.0 when nothing is planted, so an unplanted fixture still
    /// produces usable (if flat) scores.
    fn fitness(&self, job: &JobSpec) -> f64 {
        let mut matched = 0usize;
        let mut total = 0usize;
        for chain in job.chains.iter().filter(|c| c.designed) {
            let plant = match self.plants.get(&chain.label) {
                Some(plant) => plant,
                None => continue,
            };
            let residues: Vec<char> = chain.sequence.chars().collect();
            for (position, wanted) in &plant.ideal {
                total += 1;
                if residues.get(*position).copied() == Some(*wanted) {
                    matched += 1;
                }
            }
        }
        if total == 0 {
            return 1.0;
        }
        matched as f64 / total as f64
    }

    /// How much of the planted optimum a sequence has recovered, 0..1.
    ///
    /// The fixture knows the answer it planted, so this is the one score in
    /// the project that can be checked against ground truth. It is what makes
    /// `fh bench` able to say whether a search policy actually works, rather
    /// than only that it produced a number.
    ///
    /// Returns `None` when nothing was planted, since then there is nothing to
    /// recover.
    pub fn recovery(&self, chains: &[(String, String)]) -> Option<f64> {
        let mut matched = 0usize;
        let mut total = 0usize;
        for (label, sequence) in chains {
            let Some(plant) = self.plants.get(label) else {
                continue;
            };
            let residues: Vec<char> = sequence.chars().collect();
            for (position, wanted) in &plant.ideal {
                total += 1;
                if residues.get(*position).copied() == Some(*wanted) {
                    matched += 1;
                }
            }
        }
        if total == 0 {
            None
        } else {
            Some(matched as f64 / total as f64)
        }
    }

    /// Number of planted positions, which is the most a run could recover.
    pub fn planted_positions(&self) -> usize {
        self.plants.values().map(|plant| plant.ideal.len()).sum()
    }

    fn designed_text(job: &JobSpec) -> String {
        job.chains
            .iter()
            .filter(|c| c.designed)
            .map(|c| c.sequence.as_str())
            .collect::<Vec<_>>()
            .join("/")
    }

    fn confidence(&self, job: &JobSpec, fitness: f64, state: &mut u64) -> ConfidenceReport {
        let jitter = |state: &mut u64, span: f64| (unit(state) - 0.5) * span;

        let iptm = (0.30 + 0.60 * fitness + jitter(state, 0.04)).clamp(0.0, 1.0);
        let interface_plddt = (54.0 + 36.0 * fitness + jitter(state, 4.0)).clamp(0.0, 100.0);
        let plddt_mean = (62.0 + 30.0 * fitness + jitter(state, 3.0)).clamp(0.0, 100.0);
        let pae_interface = (22.0 - 16.0 * fitness + jitter(state, 1.5)).clamp(0.5, 32.0);
        let interface_contacts = (12.0 + 48.0 * fitness + jitter(state, 4.0)).max(0.0) as u32;

        // A small, deterministic share of candidates comes back with clashes so
        // the gate path is exercised on every run.
        let clashes = if splitmix64(state) % 19 == 0 {
            3 + (splitmix64(state) % 3) as u32
        } else {
            0
        };

        let designed: Vec<char> = job
            .chains
            .iter()
            .filter(|c| c.designed)
            .flat_map(|c| c.sequence.chars())
            .collect();
        let hydrophobic = designed.iter().filter(|c| HYDROPHOBIC.contains(c)).count() as f64
            / designed.len().max(1) as f64;
        let net_charge = designed
            .iter()
            .map(|c| match c {
                'K' | 'R' => 1.0,
                'D' | 'E' => -1.0,
                'H' => 0.1,
                _ => 0.0,
            })
            .sum::<f64>();

        let plddt: Vec<f32> = designed
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let wave = ((index as f64) * 0.35).sin() * 6.0;
                (plddt_mean + wave + jitter(state, 2.0)).clamp(20.0, 99.0) as f32
            })
            .collect();

        ConfidenceReport {
            iptm,
            ptm: (iptm * 0.9 + 0.06).clamp(0.0, 1.0),
            plddt_mean,
            interface_plddt,
            pae_interface,
            interface_contacts,
            clashes,
            hydrophobic_patch: (0.15 + 0.6 * hydrophobic).clamp(0.0, 1.0),
            net_charge,
            plddt,
            atom_confidence: job.need_atom_confidence,
        }
    }

    /// A synthetic backbone, shaped like a compact chain rather than a line.
    ///
    /// A confined random walk: each step is 3.8 Å, the real alpha-carbon
    /// spacing, with a restoring pull toward the chain's centre once the walk
    /// strays past a radius that grows with chain length the way a radius of
    /// gyration does. The result is a globular tangle the viewer can rotate,
    /// with the two chains placed side by side so there is an interface to
    /// look at.
    ///
    /// This is geometry, not prediction. Nothing here is folded, and the shape
    /// says nothing about the sequence beyond being deterministic in it.
    fn ca_trace(job: &JobSpec, plddt: &[f32], state: &mut u64) -> Vec<CaResidue> {
        /// Real Cα–Cα spacing, so the overall dimensions are plausible.
        const STEP: f32 = 3.8;

        let mut out = Vec::new();
        let mut designed_seen = 0usize;

        for (chain_index, chain) in job.chains.iter().enumerate() {
            let count = chain.sequence.chars().count().max(1);
            let radius = 2.4 * (count as f32).powf(0.38);
            // Chain 0 to the left of the origin, the rest to the right, so the
            // two sit adjacent instead of overlapping.
            let centre = [
                if chain_index == 0 {
                    -radius - 2.0
                } else {
                    radius + 2.0
                },
                0.0,
                0.0,
            ];

            let mut position = centre;
            let mut direction = [1.0f32, 0.0, 0.0];

            for (index, residue) in chain.sequence.chars().enumerate() {
                let mut jitter = |span: f32| (unit(state) as f32 - 0.5) * span;
                let mut next = [
                    direction[0] + jitter(1.6),
                    direction[1] + jitter(1.6),
                    direction[2] + jitter(1.6),
                ];

                // Pull back toward the centre once the walk wanders out, which
                // is what keeps it compact instead of escaping.
                let offset = [
                    position[0] - centre[0],
                    position[1] - centre[1],
                    position[2] - centre[2],
                ];
                let distance =
                    (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
                if distance > radius * 0.55 && distance > f32::EPSILON {
                    let pull = (distance / radius) * 1.4;
                    next[0] -= offset[0] / distance * pull;
                    next[1] -= offset[1] / distance * pull;
                    next[2] -= offset[2] / distance * pull;
                }

                let length = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                if length > f32::EPSILON {
                    direction = [next[0] / length, next[1] / length, next[2] / length];
                }
                position = [
                    position[0] + direction[0] * STEP,
                    position[1] + direction[1] * STEP,
                    position[2] + direction[2] * STEP,
                ];

                let confidence = if chain.designed {
                    let value = plddt.get(designed_seen).copied().unwrap_or(70.0);
                    designed_seen += 1;
                    value
                } else {
                    88.0
                };

                out.push(CaResidue {
                    chain_index: chain_index as u8,
                    index: index as u32,
                    residue,
                    xyz: position,
                    plddt: confidence,
                    designed: chain.designed,
                });
            }
        }
        out
    }
}

#[async_trait]
impl FoldBackend for FixtureBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn health(&self) -> Result<Health, FoldError> {
        Ok(Health {
            protocol: PROTOCOL.to_string(),
            service: "fh-fold fixture".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            capabilities: self
                .capabilities
                .iter()
                .map(|c| c.as_str().to_string())
                .collect(),
            models: vec!["fixture".to_string()],
            gpus: Vec::new(),
            queue_depth: 0,
        })
    }

    async fn fold(&self, job: JobSpec) -> Result<FoldResult, FoldError> {
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }

        // Everything below is a pure function of the sequence and the seed, so
        // a run replays identically.
        let text = Self::designed_text(&job);
        let mut state = hash_text(&text, self.seed ^ job.seeds.first().copied().unwrap_or(0));

        if self.failure_rate > 0.0 && unit(&mut state) < self.failure_rate {
            return Err(FoldError::Rejected(
                "fixture backend: synthetic failure".to_string(),
            ));
        }

        let fitness = self.fitness(&job);
        let confidence = self.confidence(&job, fitness, &mut state);
        let ca_trace = Self::ca_trace(&job, &confidence.plddt, &mut state);

        Ok(FoldResult {
            job_id: format!("fixture-{:016x}", hash_text(&text, self.seed)),
            model: job.model.clone(),
            confidence,
            structure: StructurePayload {
                format: StructureFormat::CaTrace,
                inline: None,
                artifact_url: None,
                ca_trace,
            },
            elapsed_ms: self.latency.as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ChainSpec, MoleculeKind};

    fn job(sequence: &str) -> JobSpec {
        JobSpec {
            name: "t".into(),
            model: "fixture".into(),
            chains: vec![
                ChainSpec {
                    label: "H".into(),
                    sequence: sequence.into(),
                    kind: MoleculeKind::Protein,
                    designed: true,
                },
                ChainSpec {
                    label: "A".into(),
                    sequence: "MKTVRQERLKSIVRILERSKEPVSGA".into(),
                    kind: MoleculeKind::Protein,
                    designed: false,
                },
            ],
            seeds: vec![1],
            diffusion_samples: 1,
            diffusion_steps: 8,
            use_msa: false,
            need_atom_confidence: true,
        }
    }

    fn planted(ideal: &[(usize, char)]) -> FixtureBackend {
        let mut plant = Plant {
            ideal: BTreeMap::new(),
        };
        for (position, residue) in ideal {
            plant.ideal.insert(*position, *residue);
        }
        let mut plants = BTreeMap::new();
        plants.insert("H".to_string(), plant);
        FixtureBackend {
            plants,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn matching_the_planted_optimum_scores_better() {
        let backend = planted(&[(1, 'W'), (2, 'Y')]);
        let poor = backend.fold(job("AAAAAAAAAA")).await.expect("folds");
        let good = backend.fold(job("AWYAAAAAAA")).await.expect("folds");
        assert!(
            good.confidence.iptm > poor.confidence.iptm,
            "{} should beat {}",
            good.confidence.iptm,
            poor.confidence.iptm
        );
    }

    #[tokio::test]
    async fn the_same_sequence_always_folds_to_the_same_numbers() {
        let backend = planted(&[(1, 'W')]);
        let first = backend.fold(job("AWAAAAAAAA")).await.expect("folds");
        let second = backend.fold(job("AWAAAAAAAA")).await.expect("folds");
        assert_eq!(first.confidence, second.confidence);
        assert_eq!(
            first.structure.ca_trace.len(),
            second.structure.ca_trace.len()
        );
    }

    #[tokio::test]
    async fn a_trace_is_emitted_for_every_residue_of_every_chain() {
        let backend = planted(&[]);
        let result = backend.fold(job("AAAAAAAAAA")).await.expect("folds");
        let expected = 10 + "MKTVRQERLKSIVRILERSKEPVSGA".len();
        assert_eq!(result.structure.ca_trace.len(), expected);
        assert!(result
            .structure
            .ca_trace
            .iter()
            .any(|residue| residue.designed));
    }
}
