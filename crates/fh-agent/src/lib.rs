//! The decision layer.
//!
//! An agent answers one question per cycle: of the skills the router says are
//! legal, which one now? It cannot widen that set, so the agent is the place
//! where judgement lives and nowhere else. Two implementations are here: a
//! deterministic [`HeuristicAgent`] that needs no model and makes offline runs
//! reproducible, and [`LlmAgent`], which asks a [`ModelProvider`] and falls
//! back to the heuristic when the answer cannot be interpreted.

use async_trait::async_trait;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use fh_core::{SearchState, SkillId};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("the router offered no legal skills")]
    NoLegalSkills,
    #[error("model provider failed: {0}")]
    Provider(String),
    #[error("could not interpret the model's choice: {0}")]
    Undecidable(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub skill: SkillId,
    pub rationale: String,
    /// Which mechanism actually decided, so a run can be audited when a model
    /// answer was discarded in favour of the fallback.
    pub decided_by: String,
}

#[async_trait]
pub trait DesignAgent: Send {
    fn name(&self) -> &str;

    async fn choose(
        &mut self,
        legal: &[SkillId],
        state: &SearchState,
    ) -> Result<Decision, AgentError>;

    /// Optional periodic summary, written into the trace. Default: nothing.
    async fn reflect(&mut self, _state: &SearchState) -> Result<Option<String>, AgentError> {
        Ok(None)
    }
}

/// Deterministic policy over the legal set.
///
/// Prefers whichever skill has actually been getting candidates admitted, with
/// a fixed exploration rate so a skill that started badly is retried, and a
/// hard override that reaches for a disruptive skill once the search stalls.
#[derive(Debug, Clone)]
pub struct HeuristicAgent {
    rng: StdRng,
    exploration: f64,
    name: String,
}

impl HeuristicAgent {
    pub fn new(seed: u64) -> Self {
        HeuristicAgent {
            rng: StdRng::seed_from_u64(seed),
            exploration: 0.15,
            name: "heuristic".to_string(),
        }
    }

    pub fn with_exploration(mut self, probability: f64) -> Self {
        self.exploration = probability.clamp(0.0, 1.0);
        self
    }

    /// Untried skills score 1.0 so each gets at least one cycle before being
    /// judged.
    fn score(state: &SearchState, skill: SkillId) -> f64 {
        state.stats_for(skill).admission_rate().unwrap_or(1.0)
    }
}

#[async_trait]
impl DesignAgent for HeuristicAgent {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose(
        &mut self,
        legal: &[SkillId],
        state: &SearchState,
    ) -> Result<Decision, AgentError> {
        let first = *legal.first().ok_or(AgentError::NoLegalSkills)?;

        if state.stagnating && legal.contains(&SkillId::RegionRedesign) {
            return Ok(Decision {
                skill: SkillId::RegionRedesign,
                rationale: format!(
                    "no improvement for {} cycles; replacing a whole region instead of refining",
                    state.cycles_without_improvement
                ),
                decided_by: self.name.clone(),
            });
        }

        if legal.len() > 1 && self.rng.gen_bool(self.exploration) {
            let skill = legal.choose(&mut self.rng).copied().unwrap_or(first);
            return Ok(Decision {
                skill,
                rationale: "exploration draw".to_string(),
                decided_by: self.name.clone(),
            });
        }

        let mut best = first;
        let mut best_score = f64::NEG_INFINITY;
        for skill in legal {
            let score = Self::score(state, *skill);
            if score > best_score {
                best_score = score;
                best = *skill;
            }
        }

        let stats = state.stats_for(best);
        let rationale = match stats.admission_rate() {
            Some(rate) => format!(
                "best admission rate among legal skills: {:.0}% over {} proposals",
                rate * 100.0,
                stats.proposed
            ),
            None => "not tried yet in this run".to_string(),
        };

        Ok(Decision {
            skill: best,
            rationale,
            decided_by: self.name.clone(),
        })
    }

    async fn reflect(&mut self, state: &SearchState) -> Result<Option<String>, AgentError> {
        if state.stats.is_empty() {
            return Ok(None);
        }
        let mut parts: Vec<String> = state
            .stats
            .iter()
            .map(|(skill, stats)| {
                let rate = stats
                    .admission_rate()
                    .map(|r| format!("{:.0}%", r * 100.0))
                    .unwrap_or_else(|| "untried".to_string());
                format!("{skill}: {} proposed, {rate} admitted", stats.proposed)
            })
            .collect();
        parts.sort();
        Ok(Some(format!(
            "cycle {} | best loss {} | {}",
            state.cycle,
            state
                .best_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "none".to_string()),
            parts.join("; ")
        )))
    }
}

/// A text-completion model. Implement this over whichever API you use; the
/// harness never assumes a vendor, a transport, or a tool-calling format.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn model(&self) -> &str;

    async fn complete(&self, system: &str, user: &str) -> Result<String, AgentError>;
}

/// System and user prompt for one choice. Exposed so it can be inspected and
/// tested rather than being buried in the call.
pub fn choice_prompt(legal: &[SkillId], state: &SearchState) -> (String, String) {
    let system = "You choose the next design action in a protein design search. \
Answer with JSON only: {\"skill\": \"<one of the listed skills>\", \"rationale\": \"<one sentence>\"}. \
Choose only from the listed skills."
        .to_string();

    let mut lines = vec![
        format!("cycle: {}", state.cycle),
        format!("population size: {}", state.population_size),
        format!(
            "best loss: {}",
            state
                .best_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "none yet".to_string())
        ),
        format!(
            "cycles without improvement: {}{}",
            state.cycles_without_improvement,
            if state.stagnating { " (stagnating)" } else { "" }
        ),
        String::new(),
        "legal skills:".to_string(),
    ];

    for skill in legal {
        let stats = state.stats_for(*skill);
        let rate = stats
            .admission_rate()
            .map(|r| format!("{:.0}% admitted", r * 100.0))
            .unwrap_or_else(|| "untried".to_string());
        lines.push(format!(
            "- {skill}: chosen {} times, {} proposals, {rate}",
            stats.chosen, stats.proposed
        ));
    }

    (system, lines.join("\n"))
}

/// Parse a model answer into one of the legal skills.
///
/// Accepts the requested JSON, and also a bare mention of a skill name, since
/// models often add prose around it. Returns `None` when nothing legal is
/// named — the caller then falls back rather than guessing.
pub fn parse_choice(answer: &str, legal: &[SkillId]) -> Option<(SkillId, Option<String>)> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(answer.trim()) {
        if let Some(name) = value.get("skill").and_then(|v| v.as_str()) {
            if let Some(skill) = legal.iter().find(|s| s.as_str() == name.trim()) {
                let rationale = value
                    .get("rationale")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                return Some((*skill, rationale));
            }
        }
    }

    let lowered = answer.to_ascii_lowercase();
    legal
        .iter()
        .filter_map(|skill| lowered.find(skill.as_str()).map(|at| (at, *skill)))
        .min_by_key(|(at, _)| *at)
        .map(|(_, skill)| (skill, None))
}

/// Asks a model, and falls back to the heuristic when the answer is unusable.
///
/// The fallback is deliberate: a design run should not stop because a provider
/// returned prose, and the trace records which mechanism decided.
pub struct LlmAgent<P: ModelProvider> {
    provider: P,
    fallback: HeuristicAgent,
    name: String,
}

impl<P: ModelProvider> LlmAgent<P> {
    pub fn new(provider: P, seed: u64) -> Self {
        let name = format!("model({})", provider.model());
        LlmAgent {
            provider,
            fallback: HeuristicAgent::new(seed),
            name,
        }
    }
}

#[async_trait]
impl<P: ModelProvider> DesignAgent for LlmAgent<P> {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose(
        &mut self,
        legal: &[SkillId],
        state: &SearchState,
    ) -> Result<Decision, AgentError> {
        if legal.is_empty() {
            return Err(AgentError::NoLegalSkills);
        }
        if legal.len() == 1 {
            return Ok(Decision {
                skill: legal[0],
                rationale: "only one legal skill".to_string(),
                decided_by: self.name.clone(),
            });
        }

        let (system, user) = choice_prompt(legal, state);
        match self.provider.complete(&system, &user).await {
            Ok(answer) => match parse_choice(&answer, legal) {
                Some((skill, rationale)) => Ok(Decision {
                    skill,
                    rationale: rationale
                        .unwrap_or_else(|| "chosen by model, no rationale given".to_string()),
                    decided_by: self.name.clone(),
                }),
                None => {
                    let mut decision = self.fallback.choose(legal, state).await?;
                    decision.decided_by = format!("{} (model answer unusable)", self.fallback.name());
                    Ok(decision)
                }
            },
            Err(error) => {
                let mut decision = self.fallback.choose(legal, state).await?;
                decision.decided_by = format!("{} (provider error: {error})", self.fallback.name());
                Ok(decision)
            }
        }
    }

    async fn reflect(&mut self, state: &SearchState) -> Result<Option<String>, AgentError> {
        self.fallback.reflect(state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(stagnating: bool, cycles_without_improvement: u32) -> SearchState {
        let mut state = SearchState::bootstrap();
        state.population_size = 6;
        state.stagnating = stagnating;
        state.cycles_without_improvement = cycles_without_improvement;
        state
    }

    #[tokio::test]
    async fn an_empty_legal_set_is_an_error_not_a_guess() {
        let mut agent = HeuristicAgent::new(1);
        assert!(matches!(
            agent.choose(&[], &state(false, 0)).await,
            Err(AgentError::NoLegalSkills)
        ));
    }

    #[tokio::test]
    async fn stagnation_reaches_for_the_disruptive_skill() {
        let mut agent = HeuristicAgent::new(1);
        let decision = agent
            .choose(
                &[SkillId::PointMutation, SkillId::RegionRedesign],
                &state(true, 4),
            )
            .await
            .expect("decides");
        assert_eq!(decision.skill, SkillId::RegionRedesign);
        assert!(decision.rationale.contains('4'), "{}", decision.rationale);
    }

    #[tokio::test]
    async fn the_heuristic_agent_is_reproducible() {
        let run = || async {
            let mut agent = HeuristicAgent::new(99);
            let mut picks = Vec::new();
            for _ in 0..20 {
                picks.push(
                    agent
                        .choose(
                            &[SkillId::PointMutation, SkillId::RegionRedesign],
                            &state(false, 0),
                        )
                        .await
                        .expect("decides")
                        .skill,
                );
            }
            picks
        };
        assert_eq!(run().await, run().await);
    }

    #[test]
    fn a_choice_is_parsed_from_json_or_from_prose() {
        let legal = [SkillId::PointMutation, SkillId::RegionRedesign];
        let (skill, rationale) = parse_choice(
            r#"{"skill": "region_redesign", "rationale": "stalled"}"#,
            &legal,
        )
        .expect("parses json");
        assert_eq!(skill, SkillId::RegionRedesign);
        assert_eq!(rationale.as_deref(), Some("stalled"));

        let (skill, _) =
            parse_choice("I would use point_mutation here.", &legal).expect("parses prose");
        assert_eq!(skill, SkillId::PointMutation);
    }

    #[test]
    fn an_illegal_or_empty_answer_does_not_parse() {
        let legal = [SkillId::PointMutation];
        assert!(parse_choice(r#"{"skill": "inverse_folding"}"#, &legal).is_none());
        assert!(parse_choice("no idea", &legal).is_none());
    }

    struct StubProvider {
        answer: String,
    }

    #[async_trait]
    impl ModelProvider for StubProvider {
        fn model(&self) -> &str {
            "stub"
        }
        async fn complete(&self, _system: &str, _user: &str) -> Result<String, AgentError> {
            Ok(self.answer.clone())
        }
    }

    struct BrokenProvider;

    #[async_trait]
    impl ModelProvider for BrokenProvider {
        fn model(&self) -> &str {
            "broken"
        }
        async fn complete(&self, _system: &str, _user: &str) -> Result<String, AgentError> {
            Err(AgentError::Provider("connection reset".to_string()))
        }
    }

    #[tokio::test]
    async fn a_model_choice_is_honoured_when_it_is_legal() {
        let mut agent = LlmAgent::new(
            StubProvider {
                answer: r#"{"skill":"region_redesign","rationale":"open a loop"}"#.to_string(),
            },
            1,
        );
        let decision = agent
            .choose(
                &[SkillId::PointMutation, SkillId::RegionRedesign],
                &state(false, 0),
            )
            .await
            .expect("decides");
        assert_eq!(decision.skill, SkillId::RegionRedesign);
        assert_eq!(decision.rationale, "open a loop");
    }

    #[tokio::test]
    async fn a_failing_provider_falls_back_and_says_so() {
        let mut agent = LlmAgent::new(BrokenProvider, 1);
        let decision = agent
            .choose(
                &[SkillId::PointMutation, SkillId::RegionRedesign],
                &state(false, 0),
            )
            .await
            .expect("falls back rather than failing the run");
        assert!(
            decision.decided_by.contains("provider error"),
            "{}",
            decision.decided_by
        );
    }

    #[tokio::test]
    async fn a_model_naming_an_illegal_skill_cannot_widen_the_set() {
        let mut agent = LlmAgent::new(
            StubProvider {
                answer: r#"{"skill":"inverse_folding"}"#.to_string(),
            },
            1,
        );
        let decision = agent
            .choose(
                &[SkillId::PointMutation, SkillId::RegionRedesign],
                &state(false, 0),
            )
            .await
            .expect("decides");
        assert_ne!(decision.skill, SkillId::InverseFolding);
        assert!(decision.decided_by.contains("unusable"));
    }
}
