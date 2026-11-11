//! Legality, kept separate from choice.
//!
//! The router answers "which skills *may* be used right now", the agent answers
//! "which one *should* be". Splitting them means an agent — heuristic or
//! model-driven — can never reach a skill the compute service cannot serve, the
//! user disabled, or the search state does not warrant. It also means every
//! refusal has a reason that can be shown in the UI rather than a skill
//! silently never being picked.

use std::collections::BTreeSet;

use fh_core::{Capability, SearchState, SkillId};

/// Skills with an executor in this build. The router never offers a skill the
/// harness cannot actually run, even when the service advertises the
/// capability.
pub const IMPLEMENTED: [SkillId; 2] = [SkillId::PointMutation, SkillId::RegionRedesign];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Legality {
    Legal,
    /// Not in the user's `design.skills` list.
    NotEnabled,
    /// No executor in this build.
    NotImplemented,
    /// The compute service does not advertise the capability it needs.
    MissingCapability(Capability),
    /// Legal in principle, but not in this search state.
    NotNow(&'static str),
}

impl Legality {
    pub fn is_legal(&self) -> bool {
        matches!(self, Legality::Legal)
    }

    pub fn reason(&self) -> String {
        match self {
            Legality::Legal => "legal".to_string(),
            Legality::NotEnabled => "not enabled in design.skills".to_string(),
            Legality::NotImplemented => "no executor in this build".to_string(),
            Legality::MissingCapability(capability) => {
                format!("service lacks capability '{}'", capability.as_str())
            }
            Legality::NotNow(why) => why.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Router {
    enabled: BTreeSet<SkillId>,
    capabilities: BTreeSet<Capability>,
}

impl Router {
    pub fn new(
        enabled: impl IntoIterator<Item = SkillId>,
        capabilities: impl IntoIterator<Item = Capability>,
    ) -> Self {
        Router {
            enabled: enabled.into_iter().collect(),
            capabilities: capabilities.into_iter().collect(),
        }
    }

    /// Build from a service health report's capability strings. Unknown strings
    /// are ignored rather than rejected, so a newer service can advertise
    /// capabilities this client does not know about.
    pub fn from_capability_strings(
        enabled: impl IntoIterator<Item = SkillId>,
        advertised: &[String],
    ) -> Self {
        let capabilities = advertised
            .iter()
            .filter_map(|text| match text.as_str() {
                "fold" => Some(Capability::Fold),
                "inverse_folding" => Some(Capability::InverseFolding),
                "sequence_likelihood" => Some(Capability::SequenceLikelihood),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        Router {
            enabled: enabled.into_iter().collect(),
            capabilities,
        }
    }

    pub fn capabilities(&self) -> &BTreeSet<Capability> {
        &self.capabilities
    }

    pub fn enabled(&self) -> &BTreeSet<SkillId> {
        &self.enabled
    }

    pub fn assess(&self, skill: SkillId, state: &SearchState) -> Legality {
        if !self.enabled.contains(&skill) {
            return Legality::NotEnabled;
        }
        if !IMPLEMENTED.contains(&skill) {
            return Legality::NotImplemented;
        }
        if let Some(capability) = skill.required_capability() {
            if !self.capabilities.contains(&capability) {
                return Legality::MissingCapability(capability);
            }
        }

        match skill {
            // Always available: the conservative local move.
            SkillId::PointMutation => Legality::Legal,

            // Replacing a whole region is disruptive. It is the right move
            // while bootstrapping, when there is nothing to refine yet, and
            // again once refinement has stopped paying off — but not in
            // between, where it mostly discards progress.
            SkillId::RegionRedesign => {
                if state.population_size <= 1 || state.stagnating {
                    Legality::Legal
                } else {
                    Legality::NotNow(
                        "region redesign is reserved for bootstrap and stagnation",
                    )
                }
            }

            SkillId::InverseFolding | SkillId::LikelihoodGuided => {
                if state.population_size == 0 {
                    Legality::NotNow("needs a scored parent structure")
                } else {
                    Legality::Legal
                }
            }
        }
    }

    pub fn legal(&self, state: &SearchState) -> Vec<SkillId> {
        SkillId::ALL
            .iter()
            .copied()
            .filter(|skill| self.assess(*skill, state).is_legal())
            .collect()
    }

    /// Every skill with its verdict, for the UI and the trace.
    pub fn assess_all(&self, state: &SearchState) -> Vec<(SkillId, Legality)> {
        SkillId::ALL
            .iter()
            .copied()
            .map(|skill| (skill, self.assess(skill, state)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(population: usize, stagnating: bool) -> SearchState {
        let mut state = SearchState::bootstrap();
        state.population_size = population;
        state.stagnating = stagnating;
        state
    }

    #[test]
    fn point_mutation_is_always_legal_when_enabled() {
        let router = Router::new([SkillId::PointMutation], [Capability::Fold]);
        assert!(router.assess(SkillId::PointMutation, &state(5, false)).is_legal());
        assert!(router.assess(SkillId::PointMutation, &state(0, true)).is_legal());
    }

    #[test]
    fn region_redesign_opens_only_at_bootstrap_and_stagnation() {
        let router = Router::new(
            [SkillId::PointMutation, SkillId::RegionRedesign],
            [Capability::Fold],
        );
        assert!(router
            .assess(SkillId::RegionRedesign, &state(1, false))
            .is_legal());
        assert!(router
            .assess(SkillId::RegionRedesign, &state(8, true))
            .is_legal());
        assert!(matches!(
            router.assess(SkillId::RegionRedesign, &state(8, false)),
            Legality::NotNow(_)
        ));
    }

    #[test]
    fn a_skill_the_service_cannot_serve_is_never_offered() {
        let router = Router::new(
            [SkillId::PointMutation, SkillId::InverseFolding],
            [Capability::Fold],
        );
        assert_eq!(
            router.assess(SkillId::InverseFolding, &state(4, false)),
            Legality::MissingCapability(Capability::InverseFolding)
        );
        assert!(!router.legal(&state(4, false)).contains(&SkillId::InverseFolding));
    }

    #[test]
    fn capability_alone_does_not_make_an_unimplemented_skill_legal() {
        let router = Router::new(
            [SkillId::InverseFolding],
            [Capability::Fold, Capability::InverseFolding],
        );
        assert_eq!(
            router.assess(SkillId::InverseFolding, &state(4, false)),
            Legality::NotImplemented
        );
        assert!(router.legal(&state(4, false)).is_empty());
    }

    #[test]
    fn a_disabled_skill_stays_disabled() {
        let router = Router::new([SkillId::PointMutation], [Capability::Fold]);
        assert_eq!(
            router.assess(SkillId::RegionRedesign, &state(1, true)),
            Legality::NotEnabled
        );
    }
}
