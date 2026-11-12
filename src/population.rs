//! The retained population and its admission rules.
//!
//! Admission is the only place candidates accumulate, and it is deterministic:
//! duplicate check, diversity check, then capacity. Gates are applied by the
//! caller before admission so that a gate failure is recorded against the
//! candidate rather than silently dropped here.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::candidate::Candidate;
use crate::gate::RejectReason;
use crate::ids::CandidateId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PopulationConfig {
    /// Maximum retained candidates. The worst is evicted when a better one
    /// arrives at capacity.
    pub capacity: usize,
    /// Minimum number of differing positions between two retained candidates.
    /// `0` disables the diversity rule.
    pub min_hamming: usize,
    /// Boltzmann temperature for parent selection. Lower exploits the best
    /// candidate, higher spreads selection across the population.
    pub parent_temperature: f64,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        PopulationConfig {
            capacity: 24,
            min_hamming: 1,
            parent_temperature: 0.05,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct PopulationStats {
    pub size: usize,
    pub best_loss: Option<f64>,
    pub mean_loss: Option<f64>,
    pub worst_loss: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionOutcome {
    Admitted,
    /// Admitted, and the named candidate left the population to make room.
    Evicted(CandidateId),
    /// Admitted in place of a near-identical but worse candidate.
    Replaced(CandidateId),
}

#[derive(Debug, Clone)]
pub struct Population {
    members: Vec<Candidate>,
    fingerprints: HashMap<u64, CandidateId>,
    config: PopulationConfig,
    admitted: usize,
    rejected: usize,
}

impl Population {
    pub fn new(config: PopulationConfig) -> Self {
        Population {
            members: Vec::new(),
            fingerprints: HashMap::new(),
            config,
            admitted: 0,
            rejected: 0,
        }
    }

    /// Rebuild a population from members an earlier run recorded, so a
    /// resumed search refuses the same duplicates the recorded one would.
    /// The fingerprint map comes from the members; `admitted` and `rejected`
    /// are the counters the recording run reported. No I/O happens here: the
    /// caller parses the snapshot and passes the values in.
    pub fn restore(
        config: PopulationConfig,
        members: Vec<Candidate>,
        admitted: usize,
        rejected: usize,
    ) -> Self {
        let fingerprints = members
            .iter()
            .map(|member| (member.sequence.fingerprint(), member.id))
            .collect();
        Population {
            members,
            fingerprints,
            config,
            admitted,
            rejected,
        }
    }

    pub fn config(&self) -> &PopulationConfig {
        &self.config
    }

    pub fn members(&self) -> &[Candidate] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn admitted_total(&self) -> usize {
        self.admitted
    }

    pub fn rejected_total(&self) -> usize {
        self.rejected
    }

    pub fn get(&self, id: CandidateId) -> Option<&Candidate> {
        self.members.iter().find(|c| c.id == id)
    }

    pub fn best(&self) -> Option<&Candidate> {
        self.members
            .iter()
            .min_by(|a, b| a.loss_or_inf().total_cmp(&b.loss_or_inf()))
    }

    fn worst(&self) -> Option<&Candidate> {
        self.members
            .iter()
            .max_by(|a, b| a.loss_or_inf().total_cmp(&b.loss_or_inf()))
    }

    /// Members ordered best-first.
    pub fn ranked(&self) -> Vec<&Candidate> {
        let mut out: Vec<&Candidate> = self.members.iter().collect();
        out.sort_by(|a, b| a.loss_or_inf().total_cmp(&b.loss_or_inf()));
        out
    }

    pub fn stats(&self) -> PopulationStats {
        if self.members.is_empty() {
            return PopulationStats::default();
        }
        let losses: Vec<f64> = self.members.iter().map(|c| c.loss_or_inf()).collect();
        let sum: f64 = losses.iter().sum();
        PopulationStats {
            size: losses.len(),
            best_loss: losses.iter().copied().reduce(f64::min),
            mean_loss: Some(sum / losses.len() as f64),
            worst_loss: losses.iter().copied().reduce(f64::max),
        }
    }

    pub fn admit(&mut self, candidate: Candidate) -> Result<AdmissionOutcome, RejectReason> {
        if !candidate.is_scored() {
            self.rejected += 1;
            return Err(RejectReason::Unscored);
        }

        let fingerprint = candidate.sequence.fingerprint();
        if let Some(existing) = self.fingerprints.get(&fingerprint) {
            self.rejected += 1;
            return Err(RejectReason::Duplicate { of: *existing });
        }

        // Diversity: a candidate too close to a retained one replaces it if it
        // is better, and is rejected otherwise. This keeps the population from
        // collapsing onto one basin without discarding genuine improvements.
        if self.config.min_hamming > 0 {
            let mut replace: Option<(usize, CandidateId)> = None;
            for (index, member) in self.members.iter().enumerate() {
                let distance = match candidate.sequence.hamming(&member.sequence) {
                    Some(d) => d,
                    None => continue,
                };
                if distance < self.config.min_hamming {
                    if candidate.loss_or_inf() < member.loss_or_inf() {
                        replace = Some((index, member.id));
                        break;
                    }
                    self.rejected += 1;
                    return Err(RejectReason::TooSimilar {
                        to: member.id,
                        hamming: distance,
                        required: self.config.min_hamming,
                    });
                }
            }
            if let Some((index, displaced)) = replace {
                let old = self.members.swap_remove(index);
                self.fingerprints.remove(&old.sequence.fingerprint());
                self.fingerprints.insert(fingerprint, candidate.id);
                self.members.push(candidate);
                self.admitted += 1;
                return Ok(AdmissionOutcome::Replaced(displaced));
            }
        }

        if self.config.capacity > 0 && self.members.len() >= self.config.capacity {
            let worst = self.worst().map(|c| (c.id, c.loss_or_inf()));
            match worst {
                Some((worst_id, worst_loss)) if candidate.loss_or_inf() < worst_loss => {
                    if let Some(index) = self.members.iter().position(|c| c.id == worst_id) {
                        let old = self.members.swap_remove(index);
                        self.fingerprints.remove(&old.sequence.fingerprint());
                    }
                    self.fingerprints.insert(fingerprint, candidate.id);
                    self.members.push(candidate);
                    self.admitted += 1;
                    return Ok(AdmissionOutcome::Evicted(worst_id));
                }
                Some((worst_id, worst_loss)) => {
                    self.rejected += 1;
                    return Err(RejectReason::WorseThanPopulation {
                        worst: worst_id,
                        worst_loss,
                        loss: candidate.loss_or_inf(),
                    });
                }
                None => {}
            }
        }

        self.fingerprints.insert(fingerprint, candidate.id);
        self.members.push(candidate);
        self.admitted += 1;
        Ok(AdmissionOutcome::Admitted)
    }

    /// Boltzmann-weighted selection on loss. Deterministic for a given RNG
    /// state, so a run replays identically from its seed.
    pub fn select_parent<R: Rng>(&self, rng: &mut R) -> Option<&Candidate> {
        if self.members.is_empty() {
            return None;
        }
        if self.members.len() == 1 {
            return self.members.first();
        }

        let best = self.best().map(|c| c.loss_or_inf()).unwrap_or(0.0);
        let temperature = self.config.parent_temperature.max(1e-6);
        let weights: Vec<f64> = self
            .members
            .iter()
            .map(|c| {
                let delta = c.loss_or_inf() - best;
                (-delta / temperature).exp()
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return self.best();
        }

        let mut point = rng.gen_range(0.0..total);
        for (member, weight) in self.members.iter().zip(weights.iter()) {
            point -= weight;
            if point <= 0.0 {
                return Some(member);
            }
        }
        self.members.last()
    }
}
