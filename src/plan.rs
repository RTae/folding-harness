//! Turning a request in prose into a validated task config.
//!
//! The model chooses *parameters* — which spans are mutable, which residues are
//! pinned, how many cycles, where the gates sit. It is never asked for a
//! sequence: sequences come from the user, as files or arguments. A model that
//! guessed a target sequence and had it folded would produce a confident,
//! meaningless result, which is worse than an error.
//!
//! Whatever comes back is parsed into [`PlanDraft`], merged with the user's
//! sequences, and put through the same [`TaskConfig`] validation as a
//! hand-written file. A hallucinated field fails `deny_unknown_fields`; a
//! region past the end of a chain fails the scaffold check. The types are the
//! guardrail, and [`draft`] feeds the error back for one correction attempt.

use serde::{Deserialize, Serialize};

use crate::agent::{AgentError, ModelProvider};
use crate::config::{BinderSpec, ChainSpec, DesignConfig, FoldConfig, RegionSpec, TargetSpec};
use crate::{
    BinderTopology, ConfigError, Gates, LossWeights, MutableScope, PopulationConfig, Scaffold,
    SkillId, TaskConfig,
};

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error(transparent)]
    Provider(#[from] AgentError),
    #[error("the reply contained no JSON object")]
    NoJson,
    #[error("the JSON did not match the plan schema: {source}\n\nreply was:\n{raw}")]
    Schema {
        #[source]
        source: serde_json::Error,
        raw: String,
    },
    #[error("the plan did not validate: {source}")]
    Invalid {
        #[source]
        source: ConfigError,
    },
}

/// A labelled sequence supplied by the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub label: String,
    pub sequence: String,
}

impl Chain {
    pub fn new(label: impl Into<String>, sequence: impl Into<String>) -> Self {
        Chain {
            label: label.into(),
            sequence: sequence.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.sequence.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// What the model is allowed to decide. No sequence fields, by construction.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanDraft {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    pub topology: BinderTopology,
    pub regions: Vec<RegionSpec>,
    #[serde(default)]
    pub fixed_residues: Vec<String>,
    #[serde(default)]
    pub cycles: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub max_substitutions: Option<usize>,
    #[serde(default)]
    pub skills: Option<Vec<SkillId>>,
    #[serde(default)]
    pub gates: Option<Gates>,
    #[serde(default)]
    pub loss: Option<LossWeights>,
    #[serde(default)]
    pub population: Option<PopulationConfig>,
}

impl PlanDraft {
    /// Merge with the user's sequences into a full task config.
    pub fn into_config(self, binder: &[Chain], target: &[Chain], seed: u64) -> TaskConfig {
        let mut design = DesignConfig {
            seed,
            ..DesignConfig::default()
        };
        if let Some(cycles) = self.cycles {
            design.cycles = cycles.max(1);
        }
        if let Some(batch) = self.batch_size {
            design.batch_size = batch.max(1);
        }
        if let Some(max) = self.max_substitutions {
            design.max_substitutions = max.max(1);
        }
        if let Some(skills) = self.skills {
            if !skills.is_empty() {
                design.skills = skills;
            }
        }

        TaskConfig {
            name: self.name.unwrap_or_else(|| "chat-task".to_string()),
            notes: self.notes,
            binder: BinderSpec {
                topology: self.topology,
                chains: binder
                    .iter()
                    .map(|chain| ChainSpec {
                        label: chain.label.clone(),
                        sequence: chain.sequence.clone(),
                    })
                    .collect(),
                regions: self.regions,
                fixed_residues: self.fixed_residues,
            },
            target: TargetSpec {
                chains: target
                    .iter()
                    .map(|chain| ChainSpec {
                        label: chain.label.clone(),
                        sequence: chain.sequence.clone(),
                    })
                    .collect(),
            },
            design,
            population: self.population.unwrap_or_default(),
            gates: self.gates.unwrap_or_default(),
            loss: self.loss.unwrap_or_default(),
            fold: FoldConfig::default(),
        }
    }
}

/// A validated plan: the config, and what it resolves to.
#[derive(Debug)]
pub struct Plan {
    pub config: TaskConfig,
    pub scaffold: Scaffold,
    pub scope: MutableScope,
}

impl Plan {
    /// The lines shown to a user before they confirm. Deliberately concrete:
    /// what may change, what may not, and what will happen.
    pub fn summary(&self) -> Vec<String> {
        let mut lines = vec![
            format!("task        {}", self.config.name),
            format!("topology    {}", self.scaffold.topology.as_str()),
        ];
        for chain in &self.scaffold.chains {
            lines.push(format!(
                "chain {:<4} {:>4} residues  {}",
                chain.label,
                chain.len(),
                match chain.role {
                    crate::ChainRole::Binder => "binder",
                    crate::ChainRole::Target => "target (never changes)",
                }
            ));
        }
        lines.push(String::new());
        lines.push(format!("mutable     {}", self.scope.summary()));
        for region in self.scope.regions() {
            let mutable = self.scope.region_positions(&region.name).len();
            let pinned = region.len() - mutable;
            lines.push(format!(
                "  {:<10} residues {}..{}  {} mutable{}",
                region.name,
                region.start + 1,
                region.end,
                mutable,
                if pinned > 0 {
                    format!(", {pinned} pinned")
                } else {
                    String::new()
                }
            ));
        }
        let immutable = self.scope.total_binder_len() - self.scope.mutable_count();
        lines.push(format!(
            "  {:<10} {} binder residues cannot change",
            "fixed", immutable
        ));
        lines.push(String::new());
        lines.push(format!(
            "search      {} cycles x {} proposals, seed {}",
            self.config.design.cycles, self.config.design.batch_size, self.config.design.seed
        ));
        lines.push(format!(
            "skills      {}",
            self.config
                .design
                .skills
                .iter()
                .map(|skill| skill.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        lines.push(format!(
            "gates       ipTM >= {:.2}, interface pLDDT >= {:.0}, PAE <= {:.0}, clashes <= {}",
            self.config.gates.min_iptm,
            self.config.gates.min_interface_plddt,
            self.config.gates.max_pae_interface,
            self.config.gates.max_clashes
        ));
        lines.push(format!(
            "backend     {} ({:?})",
            self.config.fold.model, self.config.fold.execution_mode
        ));
        lines
    }
}

pub fn system_prompt() -> String {
    "You help set up runs for a protein design harness. The user describes what they want; \
you reply with a plan as a single JSON object.

The harness runs a search loop: pick a parent sequence, pick a design skill, propose candidate \
sequences, predict each structure with a folding backend, score them against a weighted loss, \
apply hard gates, and keep the survivors in a population.

Hard rules:
- NEVER invent, guess, complete or modify a sequence. The user supplies the sequences; you only \
choose parameters. There is no sequence field in the schema and you must not add one.
- Residue positions are ONE-BASED and INCLUSIVE, and must lie inside the binder chain whose \
label you name. You are told each chain's length; a region beyond it is rejected.
- Regions must not overlap. Only binder chains can have regions.
- `fixed_residues` pins positions that fall INSIDE a region, written \"LABEL:POSITION\". Use it \
for residues the user says to keep.
- Set gates PERMISSIVELY for a first run. Gates are absolute: a candidate failing any of them is \
discarded whatever its loss. If a gate sits above what the starting sequence already scores, \
nothing is ever admitted and the search cannot move. Suggested first-run values: min_iptm 0.25, \
min_interface_plddt 48, max_pae_interface 25, max_clashes 2, min_interface_contacts 8, \
max_hydrophobic_patch 0.75. Say in `notes` that they should be tightened once a run clears them.

Reply with the JSON object and a short sentence of explanation. Do not wrap the JSON in prose \
that changes its meaning. Schema:

{
  \"name\": \"short-kebab-name\",
  \"notes\": \"one or two sentences on the choices made\",
  \"topology\": \"vhh\" | \"scfv\" | \"paired_chains\",
  \"regions\": [{\"name\": \"loop1\", \"chain\": \"<binder chain label>\", \"start\": 26, \"end\": 33}],
  \"fixed_residues\": [\"H:52\"],
  \"cycles\": 14,
  \"batch_size\": 6,
  \"max_substitutions\": 3,
  \"skills\": [\"point_mutation\", \"region_redesign\"],
  \"gates\": {\"min_iptm\": 0.25, \"min_interface_plddt\": 48.0, \"max_pae_interface\": 25.0, \
\"max_clashes\": 2, \"min_interface_contacts\": 8, \"max_hydrophobic_patch\": 0.75},
  \"population\": {\"capacity\": 16, \"min_hamming\": 1, \"parent_temperature\": 0.05}
}

Only the listed keys are accepted; any other key is an error. `topology` must match the number \
of binder chains: vhh and scfv take one, paired_chains takes two.

Two design skills exist: `point_mutation` substitutes a few positions anywhere in the mutable \
scope, and `region_redesign` replaces one whole region. The harness offers region_redesign only \
while bootstrapping or when the search has stalled.

Be honest about limits: the harness predicts structures, and a prediction is not a measurement. \
Results need experimental validation. Do not claim a design will bind.".to_string()
}

pub fn user_prompt(prose: &str, binder: &[Chain], target: &[Chain]) -> String {
    let describe = |chains: &[Chain]| -> String {
        if chains.is_empty() {
            return "  (none supplied)".to_string();
        }
        chains
            .iter()
            .map(|chain| format!("  label {} — {} residues", chain.label, chain.len()))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Binder chains the user supplied:\n{}\n\nTarget chains the user supplied:\n{}\n\n\
Their request:\n{}\n\nReply with the plan JSON.",
        describe(binder),
        describe(target),
        prose.trim()
    )
}

/// Pull the outermost JSON object out of a reply, tolerating fences and prose.
pub fn extract_json(text: &str) -> Option<&str> {
    // Prefer a fenced block when there is one: a reply may legitimately
    // mention braces in prose.
    let fenced = text
        .find("```json")
        .map(|at| at + "```json".len())
        .or_else(|| text.find("```").map(|at| at + 3));
    if let Some(start) = fenced {
        if let Some(end) = text[start..].find("```") {
            let inner = text[start..start + end].trim();
            if inner.starts_with('{') {
                return Some(inner);
            }
        }
    }

    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(text[start..=end].trim())
}

/// Parse a reply into a validated plan.
pub fn plan_from_reply(
    reply: &str,
    binder: &[Chain],
    target: &[Chain],
    seed: u64,
) -> Result<Plan, PlanError> {
    let json = extract_json(reply).ok_or(PlanError::NoJson)?;
    let draft: PlanDraft = serde_json::from_str(json).map_err(|source| PlanError::Schema {
        source,
        raw: json.chars().take(1200).collect(),
    })?;
    let config = draft.into_config(binder, target, seed);
    let (scaffold, scope) = config
        .resolve()
        .map_err(|source| PlanError::Invalid { source })?;
    Ok(Plan {
        config,
        scaffold,
        scope,
    })
}

/// Ask for a plan, and give one correction attempt if the first does not
/// validate. Feeding the typed error back is usually enough: the errors name
/// the field and the bound.
pub async fn draft(
    provider: &dyn ModelProvider,
    prose: &str,
    binder: &[Chain],
    target: &[Chain],
    seed: u64,
) -> Result<Plan, PlanError> {
    let system = system_prompt();
    let first = provider
        .complete(&system, &user_prompt(prose, binder, target))
        .await?;

    match plan_from_reply(&first, binder, target, seed) {
        Ok(plan) => Ok(plan),
        Err(error) => {
            let retry = format!(
                "{}\n\nYour previous reply did not work:\n\n{}\n\n{}\n\nReply with corrected \
plan JSON only.",
                user_prompt(prose, binder, target),
                first.chars().take(1500).collect::<String>(),
                error
            );
            let second = provider.complete(&system, &retry).await?;
            plan_from_reply(&second, binder, target, seed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binder() -> Vec<Chain> {
        vec![Chain::new(
            "H",
            "QVQLVESGGGLVQAGGSLRLSCAASGYTFTSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGSYYDYWGQGTQVTVSS",
        )]
    }

    fn target() -> Vec<Chain> {
        vec![Chain::new(
            "A",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY",
        )]
    }

    const GOOD: &str = r#"Here is a plan.
```json
{
  "name": "demo",
  "topology": "vhh",
  "regions": [{"name": "loop1", "chain": "H", "start": 26, "end": 33}],
  "fixed_residues": ["H:30"],
  "cycles": 6
}
```
Gates left at defaults."#;

    #[test]
    fn json_is_extracted_from_a_fenced_block_amid_prose() {
        let json = extract_json(GOOD).expect("finds the block");
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"topology\""));
    }

    #[test]
    fn json_is_extracted_without_a_fence() {
        let json = extract_json("sure: {\"a\": 1} done").expect("finds braces");
        assert_eq!(json, "{\"a\": 1}");
        assert!(extract_json("no braces here").is_none());
    }

    #[test]
    fn a_good_reply_becomes_a_validated_plan() {
        let plan = plan_from_reply(GOOD, &binder(), &target(), 7).expect("validates");
        assert_eq!(plan.config.name, "demo");
        assert_eq!(plan.config.design.cycles, 6);
        assert_eq!(plan.config.design.seed, 7);
        // 26..33 one-based inclusive is 8 positions, one of them pinned.
        assert_eq!(plan.scope.mutable_count(), 7);
        let summary = plan.summary();
        assert!(summary.iter().any(|line| line.contains("loop1")));
        assert!(summary.iter().any(|line| line.contains("cannot change")));
    }

    #[test]
    fn an_invented_sequence_field_is_refused() {
        let reply = r#"{"topology":"vhh","regions":[],"sequence":"QVQL"}"#;
        let error = plan_from_reply(reply, &binder(), &target(), 1)
            .expect_err("a sequence field must not be accepted");
        assert!(matches!(error, PlanError::Schema { .. }), "{error}");
    }

    #[test]
    fn a_region_past_the_end_of_the_chain_is_refused() {
        let reply =
            r#"{"topology":"vhh","regions":[{"name":"x","chain":"H","start":400,"end":420}]}"#;
        let error = plan_from_reply(reply, &binder(), &target(), 1)
            .expect_err("out-of-bounds region must be refused");
        assert!(matches!(error, PlanError::Invalid { .. }), "{error}");
    }

    #[test]
    fn a_topology_that_contradicts_the_chain_count_is_refused() {
        let reply = r#"{"topology":"paired_chains","regions":[{"name":"x","chain":"H","start":26,"end":33}]}"#;
        let error = plan_from_reply(reply, &binder(), &target(), 1)
            .expect_err("one binder chain is not paired_chains");
        assert!(matches!(error, PlanError::Invalid { .. }), "{error}");
    }

    #[test]
    fn a_plan_with_no_mutable_positions_is_refused() {
        let reply = r#"{"topology":"vhh","regions":[{"name":"x","chain":"H","start":30,"end":30}],"fixed_residues":["H:30"]}"#;
        let error =
            plan_from_reply(reply, &binder(), &target(), 1).expect_err("nothing left to design");
        assert!(matches!(error, PlanError::Invalid { .. }), "{error}");
    }

    #[test]
    fn the_system_prompt_states_the_sequence_rule_and_the_gate_trap() {
        let prompt = system_prompt();
        assert!(prompt.contains("NEVER invent"));
        assert!(prompt.contains("ONE-BASED"));
        assert!(prompt.contains("nothing is ever admitted"));
    }

    #[test]
    fn the_schema_in_the_prompt_has_literal_single_braces() {
        // This prompt was once built with `format!`, where braces are escaped
        // as `{{`. If it is ever rewritten, the schema must still read as JSON
        // to the model rather than as doubled braces.
        let prompt = system_prompt();
        assert!(
            !prompt.contains("{{"),
            "doubled opening braces in the schema"
        );
        assert!(
            !prompt.contains("}}"),
            "doubled closing braces in the schema"
        );
        assert!(prompt.contains("\"name\": \"short-kebab-name\""));
        assert!(prompt.contains("\"start\": 26"));
    }

    #[test]
    fn the_user_prompt_states_lengths_but_not_sequences() {
        let prompt = user_prompt("design the loops", &binder(), &target());
        assert!(prompt.contains("115 residues"));
        assert!(prompt.contains("label H"));
        assert!(
            !prompt.contains("QVQLVESGGG"),
            "the sequence itself does not need to be in the prompt"
        );
    }
}
