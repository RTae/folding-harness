//! Search bookkeeping: what has been tried, what it bought, and whether the
//! run is still making progress.
//!
//! The router reads this to decide which skills are legal, the agent reads it
//! to choose between them, and the terminal UI reads it to show why. Keeping it
//! in one place means those three never disagree about the state of the run.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::proposal::SkillId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SkillStats {
    pub chosen: u32,
    pub proposed: u32,
    pub admitted: u32,
    pub rejected: u32,
}

impl SkillStats {
    /// Share of this skill's proposals that entered the population. `None`
    /// until it has proposed anything, so an untried skill is not treated as a
    /// failure.
    pub fn admission_rate(&self) -> Option<f64> {
        if self.proposed == 0 {
            None
        } else {
            Some(f64::from(self.admitted) / f64::from(self.proposed))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleRecord {
    pub cycle: u32,
    pub skill: SkillId,
    pub proposed: usize,
    pub admitted: usize,
    pub rejected: usize,
    pub best_loss: Option<f64>,
    pub mean_loss: Option<f64>,
    /// Whether this cycle improved on the best loss so far.
    pub improved: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SearchHistory {
    records: Vec<CycleRecord>,
    stats: BTreeMap<SkillId, SkillStats>,
    best_loss: Option<f64>,
    cycles_without_improvement: u32,
}

impl SearchHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn note_choice(&mut self, skill: SkillId) {
        self.stats.entry(skill).or_default().chosen += 1;
    }

    pub fn note_proposals(&mut self, skill: SkillId, count: usize) {
        self.stats.entry(skill).or_default().proposed += count as u32;
    }

    pub fn note_admitted(&mut self, skill: SkillId) {
        self.stats.entry(skill).or_default().admitted += 1;
    }

    pub fn note_rejected(&mut self, skill: SkillId) {
        self.stats.entry(skill).or_default().rejected += 1;
    }

    /// Close a cycle. `epsilon` is the smallest loss improvement that counts as
    /// progress, so numerical noise does not reset the stagnation counter.
    pub fn record_cycle(
        &mut self,
        cycle: u32,
        skill: SkillId,
        proposed: usize,
        admitted: usize,
        rejected: usize,
        best_loss: Option<f64>,
        mean_loss: Option<f64>,
        epsilon: f64,
    ) -> &CycleRecord {
        let improved = match (best_loss, self.best_loss) {
            (Some(new), Some(old)) => new < old - epsilon,
            (Some(_), None) => true,
            _ => false,
        };

        if improved {
            self.cycles_without_improvement = 0;
        } else {
            self.cycles_without_improvement = self.cycles_without_improvement.saturating_add(1);
        }

        if let Some(new) = best_loss {
            self.best_loss = Some(match self.best_loss {
                Some(old) => old.min(new),
                None => new,
            });
        }

        self.records.push(CycleRecord {
            cycle,
            skill,
            proposed,
            admitted,
            rejected,
            best_loss,
            mean_loss,
            improved,
        });
        self.records.last().expect("just pushed")
    }

    pub fn records(&self) -> &[CycleRecord] {
        &self.records
    }

    pub fn stats(&self) -> &BTreeMap<SkillId, SkillStats> {
        &self.stats
    }

    pub fn best_loss(&self) -> Option<f64> {
        self.best_loss
    }

    pub fn cycles_without_improvement(&self) -> u32 {
        self.cycles_without_improvement
    }

    /// Best loss per completed cycle, for the loss curve.
    pub fn loss_curve(&self) -> Vec<f64> {
        self.records.iter().filter_map(|r| r.best_loss).collect()
    }

    pub fn state(
        &self,
        cycle: u32,
        population_size: usize,
        stagnation_window: u32,
    ) -> SearchState {
        SearchState {
            cycle,
            population_size,
            best_loss: self.best_loss,
            cycles_without_improvement: self.cycles_without_improvement,
            stagnating: stagnation_window > 0
                && self.cycles_without_improvement >= stagnation_window,
            stats: self.stats.clone(),
        }
    }
}

/// A snapshot of the search, passed to the router and the agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchState {
    pub cycle: u32,
    pub population_size: usize,
    pub best_loss: Option<f64>,
    pub cycles_without_improvement: u32,
    /// True once `cycles_without_improvement` reaches the configured window.
    /// The router uses this to open up more disruptive skills.
    pub stagnating: bool,
    pub stats: BTreeMap<SkillId, SkillStats>,
}

impl SearchState {
    /// State for a run that has not completed a cycle yet.
    pub fn bootstrap() -> Self {
        SearchState {
            cycle: 0,
            population_size: 0,
            best_loss: None,
            cycles_without_improvement: 0,
            stagnating: false,
            stats: BTreeMap::new(),
        }
    }

    pub fn stats_for(&self, skill: SkillId) -> SkillStats {
        self.stats.get(&skill).copied().unwrap_or_default()
    }
}
