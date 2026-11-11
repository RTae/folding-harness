//! Task configuration: the YAML a user writes, and its conversion into the
//! internal domain types.
//!
//! Two deliberate choices live here. First, `deny_unknown_fields`: a typo in a
//! long design config should fail at load, not silently run with a default.
//! Second, residue numbering is **one-based and inclusive in YAML** because
//! that is how positions are written in the literature, and zero-based
//! half-open everywhere inside the harness. The conversion happens once, in
//! [`TaskConfig::to_scaffold`].

use serde::{Deserialize, Serialize};

use crate::gate::Gates;
use crate::ids::ChainId;
use crate::metrics::LossWeights;
use crate::population::PopulationConfig;
use crate::proposal::SkillId;
use crate::scaffold::{
    BinderTopology, Chain, ChainRole, FixedResidue, MutableScope, Region, Scaffold, ScaffoldError,
};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("chain label '{label}' is used more than once")]
    DuplicateLabel { label: String },
    #[error("region '{name}' references unknown chain '{chain}'")]
    UnknownRegionChain { name: String, chain: String },
    #[error("region '{name}' has start {start}; positions are one-based, so start must be >= 1")]
    ZeroStart { name: String, start: usize },
    #[error("region '{name}' spans {start}..{end}; end must be >= start")]
    InvertedRegion {
        name: String,
        start: usize,
        end: usize,
    },
    #[error("fixed residue '{raw}' is not in CHAIN:POSITION form, e.g. 'H:52'")]
    BadFixedResidue { raw: String },
    #[error("fixed residue '{raw}' references unknown chain '{chain}'")]
    UnknownFixedChain { raw: String, chain: String },
    #[error("fixed residue '{raw}' has position 0; positions are one-based")]
    ZeroFixedPosition { raw: String },
    #[error("binder has no chains")]
    NoBinderChains,
    #[error("target has no chains")]
    NoTargetChains,
    #[error("no design skills enabled")]
    NoSkills,
    #[error("cycles must be at least 1")]
    NoCycles,
    #[error("batch_size must be at least 1")]
    EmptyBatch,
    #[error("fold.execution_mode is 'service' but fold.url is not set")]
    MissingServiceUrl,
    #[error(transparent)]
    Scaffold(#[from] ScaffoldError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainSpec {
    /// Short label used in config, logs and the structure viewer, e.g. "H".
    pub label: String,
    pub sequence: String,
}

/// One-based, inclusive region bounds, as written by a user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionSpec {
    pub name: String,
    pub chain: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinderSpec {
    pub topology: BinderTopology,
    pub chains: Vec<ChainSpec>,
    /// Mutable regions. Without any, there is nothing to design.
    #[serde(default)]
    pub regions: Vec<RegionSpec>,
    /// Positions to pin even though they fall inside a mutable region, written
    /// as `CHAIN:POSITION` with one-based positions, e.g. `H:52`.
    #[serde(default)]
    pub fixed_residues: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpec {
    pub chains: Vec<ChainSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesignConfig {
    pub cycles: u32,
    /// Candidates proposed per cycle.
    pub batch_size: usize,
    /// Seeds the whole run: parent selection, skill choice and proposals.
    pub seed: u64,
    /// Upper bound on substitutions in one point-mutation proposal.
    pub max_substitutions: usize,
    /// Cycles without improvement before the run is treated as stagnating.
    pub stagnation_window: u32,
    /// Improvement smaller than this does not count as progress.
    pub stagnation_epsilon: f64,
    /// Skills the user permits. The router narrows this further.
    pub skills: Vec<SkillId>,
    /// Folds in flight at once.
    pub concurrency: usize,
}

impl Default for DesignConfig {
    fn default() -> Self {
        DesignConfig {
            cycles: 12,
            batch_size: 6,
            seed: 7,
            max_substitutions: 3,
            stagnation_window: 3,
            stagnation_epsilon: 1e-3,
            skills: vec![SkillId::PointMutation, SkillId::RegionRedesign],
            concurrency: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Deterministic in-process stand-in. No GPU, no network.
    #[default]
    Fixture,
    /// A compute service speaking the `fh.fold/v1` contract.
    Service,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FoldConfig {
    /// Backend model name, passed through to the service and recorded in the
    /// trace. The harness does not interpret it.
    pub model: String,
    pub execution_mode: ExecutionMode,
    pub url: Option<String>,
    pub token: Option<String>,
    pub seeds: Vec<u64>,
    pub diffusion_samples: u32,
    pub diffusion_steps: u32,
    pub use_msa: bool,
    /// Per-atom confidence is required by the interface terms of the loss.
    pub need_atom_confidence: bool,
    pub poll_interval_secs: f64,
    pub timeout_secs: u64,
}

impl Default for FoldConfig {
    fn default() -> Self {
        FoldConfig {
            model: "fixture".to_string(),
            execution_mode: ExecutionMode::Fixture,
            url: None,
            token: None,
            seeds: vec![1],
            diffusion_samples: 1,
            diffusion_steps: 200,
            use_msa: true,
            need_atom_confidence: true,
            poll_interval_secs: 2.0,
            timeout_secs: 1800,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskConfig {
    pub name: String,
    #[serde(default)]
    pub notes: Option<String>,
    pub binder: BinderSpec,
    pub target: TargetSpec,
    #[serde(default)]
    pub design: DesignConfig,
    #[serde(default)]
    pub population: PopulationConfig,
    #[serde(default)]
    pub gates: Gates,
    #[serde(default)]
    pub loss: LossWeights,
    #[serde(default)]
    pub fold: FoldConfig,
}

impl TaskConfig {
    pub fn from_yaml(text: &str) -> Result<Self, ConfigError> {
        let config: TaskConfig = serde_yaml::from_str(text)?;
        config.check()?;
        Ok(config)
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml(&text)
    }

    pub fn to_yaml(&self) -> Result<String, ConfigError> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Config-level checks that do not need the scaffold built.
    fn check(&self) -> Result<(), ConfigError> {
        if self.binder.chains.is_empty() {
            return Err(ConfigError::NoBinderChains);
        }
        if self.target.chains.is_empty() {
            return Err(ConfigError::NoTargetChains);
        }
        if self.design.skills.is_empty() {
            return Err(ConfigError::NoSkills);
        }
        if self.design.cycles == 0 {
            return Err(ConfigError::NoCycles);
        }
        if self.design.batch_size == 0 {
            return Err(ConfigError::EmptyBatch);
        }
        if self.fold.execution_mode == ExecutionMode::Service && self.fold.url.is_none() {
            return Err(ConfigError::MissingServiceUrl);
        }
        Ok(())
    }

    /// Binder chains first, then target chains, both in config order.
    fn chain_ids(&self) -> Result<Vec<(String, ChainId, ChainRole)>, ConfigError> {
        let mut out = Vec::new();
        let mut next: u8 = 0;
        for (role, specs) in [
            (ChainRole::Binder, &self.binder.chains),
            (ChainRole::Target, &self.target.chains),
        ] {
            for spec in specs {
                if out.iter().any(|(label, _, _)| label == &spec.label) {
                    return Err(ConfigError::DuplicateLabel {
                        label: spec.label.clone(),
                    });
                }
                out.push((spec.label.clone(), ChainId(next), role));
                next = next.saturating_add(1);
            }
        }
        Ok(out)
    }

    pub fn to_scaffold(&self) -> Result<Scaffold, ConfigError> {
        self.check()?;
        let ids = self.chain_ids()?;
        let id_of = |label: &str| -> Option<ChainId> {
            ids.iter()
                .find(|(l, _, _)| l == label)
                .map(|(_, id, _)| *id)
        };

        let mut chains = Vec::new();
        for (spec, role) in self
            .binder
            .chains
            .iter()
            .map(|s| (s, ChainRole::Binder))
            .chain(self.target.chains.iter().map(|s| (s, ChainRole::Target)))
        {
            let id = id_of(&spec.label).expect("label was just registered");
            chains.push(Chain {
                id,
                label: spec.label.clone(),
                sequence: spec.sequence.trim().to_uppercase(),
                role,
            });
        }

        let mut regions = Vec::new();
        for spec in &self.binder.regions {
            let chain = id_of(&spec.chain).ok_or_else(|| ConfigError::UnknownRegionChain {
                name: spec.name.clone(),
                chain: spec.chain.clone(),
            })?;
            if spec.start == 0 {
                return Err(ConfigError::ZeroStart {
                    name: spec.name.clone(),
                    start: spec.start,
                });
            }
            if spec.end < spec.start {
                return Err(ConfigError::InvertedRegion {
                    name: spec.name.clone(),
                    start: spec.start,
                    end: spec.end,
                });
            }
            // one-based inclusive -> zero-based half-open
            regions.push(Region {
                name: spec.name.clone(),
                chain,
                start: spec.start - 1,
                end: spec.end,
            });
        }

        let mut fixed = Vec::new();
        for raw in &self.binder.fixed_residues {
            let (label, position) = raw
                .split_once(':')
                .ok_or_else(|| ConfigError::BadFixedResidue { raw: raw.clone() })?;
            let position: usize = position
                .trim()
                .parse()
                .map_err(|_| ConfigError::BadFixedResidue { raw: raw.clone() })?;
            if position == 0 {
                return Err(ConfigError::ZeroFixedPosition { raw: raw.clone() });
            }
            let chain = id_of(label.trim()).ok_or_else(|| ConfigError::UnknownFixedChain {
                raw: raw.clone(),
                chain: label.trim().to_string(),
            })?;
            fixed.push(FixedResidue {
                chain,
                position: position - 1,
            });
        }

        let scaffold = Scaffold {
            topology: self.binder.topology,
            chains,
            regions,
            fixed,
        };
        scaffold.validate()?;
        Ok(scaffold)
    }

    /// Scaffold plus derived mutable scope, the pair every run starts from.
    pub fn resolve(&self) -> Result<(Scaffold, MutableScope), ConfigError> {
        let scaffold = self.to_scaffold()?;
        let scope = scaffold.mutable_scope()?;
        Ok((scaffold, scope))
    }
}
