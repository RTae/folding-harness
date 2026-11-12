//! The design scaffold and the mutable scope derived from it.
//!
//! This module carries the harness's central safety property: **positions
//! outside the mutable scope cannot be changed by any design skill.** The
//! property is enforced by construction, not by convention — see
//! [`MutableScope::propose`], which is the only way to obtain a
//! [`crate::Proposal`].

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use crate::ids::ChainId;

pub const AMINO_ACIDS: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];

pub fn is_standard_aa(c: char) -> bool {
    AMINO_ACIDS.contains(&c)
}

/// Binder topologies this harness accepts. Anything else is rejected before a
/// task is allowed to allocate compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinderTopology {
    /// Single-domain binder: one binder chain.
    Vhh,
    /// Single-chain variable fragment: one binder chain with a linker.
    ScFv,
    /// Two binder chains, paired.
    PairedChains,
}

impl BinderTopology {
    pub fn expected_binder_chains(&self) -> usize {
        match self {
            BinderTopology::Vhh | BinderTopology::ScFv => 1,
            BinderTopology::PairedChains => 2,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            BinderTopology::Vhh => "vhh",
            BinderTopology::ScFv => "scfv",
            BinderTopology::PairedChains => "paired_chains",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainRole {
    /// Part of the designed binder. May contain mutable regions.
    Binder,
    /// Part of the fixed target. Never mutable.
    Target,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    pub id: ChainId,
    pub label: String,
    pub sequence: String,
    pub role: ChainRole,
}

impl Chain {
    pub fn len(&self) -> usize {
        self.sequence.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// A half-open, zero-based region of one chain: `[start, end)`.
///
/// Task YAML uses one-based inclusive bounds because that is how residues are
/// numbered in the literature; the conversion happens once, in
/// [`crate::config`], and everything downstream of it is zero-based half-open.
/// Display converts back through [`Region::display_span`]. A pinned residue
/// converts back through [`FixedResidue::display_position`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub name: String,
    pub chain: ChainId,
    pub start: usize,
    pub end: usize,
}

/// A region's bounds as every display writes them: one-based inclusive.
///
/// [`fmt::Display`] renders the pair as `start..end`, the form a task YAML
/// uses, so a user can read a printed region and pin a residue from it
/// without converting anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionSpan {
    pub start: usize,
    pub end: usize,
}

impl fmt::Display for RegionSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A pinned residue's position as every display writes it: one-based.
///
/// [`fmt::Display`] renders the number a task YAML writes, so a user can
/// search their file for a printed position without converting anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayPosition(pub usize);

impl fmt::Display for DisplayPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Region {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, position: usize) -> bool {
        position >= self.start && position < self.end
    }

    pub fn positions(&self) -> Range<usize> {
        self.start..self.end
    }

    /// The bounds of this region for display, one-based inclusive. A
    /// zero-based half-open end already reads as the inclusive one-based
    /// position, so only the start moves. This is the one place that
    /// converts a stored region back out. The conversion into it lives in
    /// [`crate::config::TaskConfig::to_scaffold`].
    pub fn display_span(&self) -> RegionSpan {
        RegionSpan {
            start: self.start + 1,
            end: self.end,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedResidue {
    pub chain: ChainId,
    pub position: usize,
}

impl FixedResidue {
    /// The position of this residue for display, one-based. This is the one
    /// place that converts a stored pinned position back out. The conversion
    /// into it lives in [`crate::config::TaskConfig::to_scaffold`].
    pub fn display_position(&self) -> DisplayPosition {
        DisplayPosition(self.position + 1)
    }
}

/// Structural faults in a scaffold.
///
/// The variants a task YAML can cause name the chain by its `label`, the
/// string the user wrote, beside the internal `chain` id, so the rendered
/// message names something the user can find in their own file. The id stays
/// available for callers that match on it.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ScaffoldError {
    #[error("scaffold has no chains")]
    NoChains,
    #[error("duplicate chain label '{label}'")]
    DuplicateLabel { label: String },
    #[error("duplicate chain id {id}")]
    DuplicateChainId { id: ChainId },
    #[error("chain '{label}' has an empty sequence")]
    EmptyChain { label: String },
    #[error("chain '{label}' contains '{residue}', which is not a standard amino acid")]
    NonStandardResidue { label: String, residue: char },
    #[error("topology {topology} expects {expected} binder chain(s), found {found}")]
    TopologyMismatch {
        topology: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("scaffold has no target chain")]
    NoTarget,
    #[error("region '{name}' references chain '{label}', which is not a binder chain")]
    RegionNotOnBinder {
        name: String,
        chain: ChainId,
        label: String,
    },
    #[error(
        "region '{name}' spans {span} but chain '{label}' has length {len}; {}",
        region_out_of_bounds_fix(*span, *len)
    )]
    RegionOutOfBounds {
        name: String,
        chain: ChainId,
        label: String,
        /// One-based inclusive, the bounds a task YAML writes. Converted from
        /// the stored region by [`Region::display_span`].
        span: RegionSpan,
        len: usize,
    },
    #[error("region '{name}' is empty; give it at least one residue")]
    EmptyRegion { name: String },
    #[error("regions '{a}' and '{b}' overlap")]
    OverlappingRegions { a: String, b: String },
    #[error("duplicate region name '{name}'")]
    DuplicateRegionName { name: String },
    #[error(
        "fixed residue {label}:{position} is outside that chain (length {len}); {}",
        fixed_out_of_bounds_fix(*len)
    )]
    FixedOutOfBounds {
        chain: ChainId,
        label: String,
        /// One-based, the position a task YAML writes. Converted from the
        /// stored pinned residue by [`FixedResidue::display_position`].
        position: DisplayPosition,
        len: usize,
    },
    #[error("every mutable position is also pinned as a fixed residue: nothing left to design")]
    NothingMutable,
}

/// The fix for a region that ran past its chain, quoted in the one-based
/// bounds the message shows. A `span.start` above `len` puts the whole region
/// past the last residue, so lowering `end` alone cannot repair it.
fn region_out_of_bounds_fix(span: RegionSpan, len: usize) -> String {
    if span.start > len {
        format!("lower start and end to {len} or less")
    } else {
        format!("lower end to {len} or less")
    }
}

/// The fix for a pinned residue that ran past its chain. A chain of length 0
/// is one this scaffold does not carry, so there is no position to lower to.
fn fixed_out_of_bounds_fix(len: usize) -> String {
    if len == 0 {
        "that chain has no residues, so remove the entry".to_string()
    } else {
        format!("lower the position to {len} or less")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scaffold {
    pub topology: BinderTopology,
    pub chains: Vec<Chain>,
    /// Mutable regions of the binder. Naming them after CDRs is conventional
    /// but not required; the harness treats them as opaque named spans.
    pub regions: Vec<Region>,
    /// Explicitly pinned positions. These take precedence over `regions`: a
    /// position inside a mutable region that is also listed here is immutable.
    pub fixed: Vec<FixedResidue>,
}

impl Scaffold {
    pub fn chain(&self, id: ChainId) -> Option<&Chain> {
        self.chains.iter().find(|c| c.id == id)
    }

    pub fn chain_by_label(&self, label: &str) -> Option<&Chain> {
        self.chains.iter().find(|c| c.label == label)
    }

    pub fn binder_chains(&self) -> impl Iterator<Item = &Chain> {
        self.chains.iter().filter(|c| c.role == ChainRole::Binder)
    }

    pub fn target_chains(&self) -> impl Iterator<Item = &Chain> {
        self.chains.iter().filter(|c| c.role == ChainRole::Target)
    }

    /// The name an error message gives a chain: its label, which is what a
    /// task YAML writes. The id fallback only serves a hand-built scaffold
    /// whose region or pin references a chain id that does not exist, which
    /// [`crate::config::TaskConfig::to_scaffold`] rejects before validating.
    fn chain_label(&self, id: ChainId) -> String {
        self.chain(id)
            .map(|c| c.label.clone())
            .unwrap_or_else(|| id.to_string())
    }

    pub fn validate(&self) -> Result<(), ScaffoldError> {
        if self.chains.is_empty() {
            return Err(ScaffoldError::NoChains);
        }

        let mut labels = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for chain in &self.chains {
            if !labels.insert(chain.label.as_str()) {
                return Err(ScaffoldError::DuplicateLabel {
                    label: chain.label.clone(),
                });
            }
            if !ids.insert(chain.id) {
                return Err(ScaffoldError::DuplicateChainId { id: chain.id });
            }
            if chain.sequence.is_empty() {
                return Err(ScaffoldError::EmptyChain {
                    label: chain.label.clone(),
                });
            }
            if let Some(bad) = chain.sequence.chars().find(|c| !is_standard_aa(*c)) {
                return Err(ScaffoldError::NonStandardResidue {
                    label: chain.label.clone(),
                    residue: bad,
                });
            }
        }

        let binders = self.binder_chains().count();
        let expected = self.topology.expected_binder_chains();
        if binders != expected {
            return Err(ScaffoldError::TopologyMismatch {
                topology: self.topology.as_str(),
                expected,
                found: binders,
            });
        }
        if self.target_chains().count() == 0 {
            return Err(ScaffoldError::NoTarget);
        }

        let mut region_names = BTreeSet::new();
        for region in &self.regions {
            if !region_names.insert(region.name.as_str()) {
                return Err(ScaffoldError::DuplicateRegionName {
                    name: region.name.clone(),
                });
            }
            let chain = self
                .chain(region.chain)
                .filter(|c| c.role == ChainRole::Binder)
                .ok_or_else(|| ScaffoldError::RegionNotOnBinder {
                    name: region.name.clone(),
                    chain: region.chain,
                    label: self.chain_label(region.chain),
                })?;
            if region.is_empty() {
                return Err(ScaffoldError::EmptyRegion {
                    name: region.name.clone(),
                });
            }
            if region.end > chain.len() {
                return Err(ScaffoldError::RegionOutOfBounds {
                    name: region.name.clone(),
                    chain: region.chain,
                    label: chain.label.clone(),
                    span: region.display_span(),
                    len: chain.len(),
                });
            }
        }

        for (i, a) in self.regions.iter().enumerate() {
            for b in self.regions.iter().skip(i + 1) {
                if a.chain == b.chain && a.start < b.end && b.start < a.end {
                    return Err(ScaffoldError::OverlappingRegions {
                        a: a.name.clone(),
                        b: b.name.clone(),
                    });
                }
            }
        }

        for fixed in &self.fixed {
            let len = self.chain(fixed.chain).map(|c| c.len()).unwrap_or(0);
            if fixed.position >= len {
                return Err(ScaffoldError::FixedOutOfBounds {
                    chain: fixed.chain,
                    label: self.chain_label(fixed.chain),
                    position: fixed.display_position(),
                    len,
                });
            }
        }

        Ok(())
    }

    /// Derive the mutable scope. Validates first, so a `MutableScope` can only
    /// exist for a scaffold that passed every structural check.
    pub fn mutable_scope(&self) -> Result<MutableScope, ScaffoldError> {
        self.validate()?;

        let mut mask: BTreeMap<ChainId, Vec<bool>> = BTreeMap::new();
        let mut lengths: BTreeMap<ChainId, usize> = BTreeMap::new();
        for chain in self.binder_chains() {
            mask.insert(chain.id, vec![false; chain.len()]);
            lengths.insert(chain.id, chain.len());
        }

        for region in &self.regions {
            if let Some(flags) = mask.get_mut(&region.chain) {
                for pos in region.positions() {
                    flags[pos] = true;
                }
            }
        }

        // Explicit pins win over region membership.
        for fixed in &self.fixed {
            if let Some(flags) = mask.get_mut(&fixed.chain) {
                if let Some(slot) = flags.get_mut(fixed.position) {
                    *slot = false;
                }
            }
        }

        let mutable = mask.values().flatten().filter(|m| **m).count();
        if mutable == 0 {
            return Err(ScaffoldError::NothingMutable);
        }

        Ok(MutableScope {
            mask,
            regions: self.regions.clone(),
            lengths,
        })
    }
}

/// The set of positions a design skill is permitted to change.
///
/// Fields are private and there is no `Deserialize` impl: a scope can only be
/// produced by [`Scaffold::mutable_scope`], so it cannot be forged from a
/// config file or a network payload.
#[derive(Debug, Clone)]
pub struct MutableScope {
    mask: BTreeMap<ChainId, Vec<bool>>,
    regions: Vec<Region>,
    lengths: BTreeMap<ChainId, usize>,
}

impl MutableScope {
    pub fn is_mutable(&self, chain: ChainId, position: usize) -> bool {
        self.mask
            .get(&chain)
            .and_then(|flags| flags.get(position))
            .copied()
            .unwrap_or(false)
    }

    pub fn chain_len(&self, chain: ChainId) -> Option<usize> {
        self.lengths.get(&chain).copied()
    }

    pub fn chains(&self) -> impl Iterator<Item = ChainId> + '_ {
        self.mask.keys().copied()
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub fn region(&self, name: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.name == name)
    }

    /// Every mutable position, in chain then position order.
    pub fn positions(&self) -> Vec<(ChainId, usize)> {
        let mut out = Vec::new();
        for (chain, flags) in &self.mask {
            for (pos, mutable) in flags.iter().enumerate() {
                if *mutable {
                    out.push((*chain, pos));
                }
            }
        }
        out
    }

    /// Mutable positions within one named region. A region may be partly
    /// pinned, so this is not simply the region's span.
    pub fn region_positions(&self, name: &str) -> Vec<(ChainId, usize)> {
        match self.region(name) {
            None => Vec::new(),
            Some(region) => region
                .positions()
                .filter(|pos| self.is_mutable(region.chain, *pos))
                .map(|pos| (region.chain, pos))
                .collect(),
        }
    }

    pub fn mutable_count(&self) -> usize {
        self.mask.values().flatten().filter(|m| **m).count()
    }

    pub fn total_binder_len(&self) -> usize {
        self.lengths.values().sum()
    }

    /// One-line summary for `validate` output and run headers.
    pub fn summary(&self) -> String {
        let regions: Vec<String> = self
            .regions
            .iter()
            .map(|r| format!("{}({})", r.name, self.region_positions(&r.name).len()))
            .collect();
        format!(
            "{} mutable of {} binder residues across {} region(s): {}",
            self.mutable_count(),
            self.total_binder_len(),
            self.regions.len(),
            regions.join(", ")
        )
    }
}

/// The mutable part of a candidate: binder chains only. The target never
/// varies within a task, so it is not carried per candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinderSequence {
    chains: BTreeMap<ChainId, String>,
}

impl BinderSequence {
    pub fn from_scaffold(scaffold: &Scaffold) -> Self {
        BinderSequence {
            chains: scaffold
                .binder_chains()
                .map(|c| (c.id, c.sequence.clone()))
                .collect(),
        }
    }

    pub fn chain(&self, id: ChainId) -> Option<&str> {
        self.chains.get(&id).map(|s| s.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (ChainId, &str)> {
        self.chains.iter().map(|(id, seq)| (*id, seq.as_str()))
    }

    pub fn total_len(&self) -> usize {
        self.chains.values().map(|s| s.chars().count()).sum()
    }

    /// FNV-1a over chain ids and residues. Used for exact-duplicate detection,
    /// not for cryptographic purposes.
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (id, seq) in &self.chains {
            for byte in std::iter::once(id.0).chain(seq.bytes()) {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
        hash
    }

    /// Number of differing positions, or `None` if the two sequences do not
    /// describe the same chain set and lengths.
    pub fn hamming(&self, other: &BinderSequence) -> Option<usize> {
        if self.chains.len() != other.chains.len() {
            return None;
        }
        let mut total = 0;
        for (id, mine) in &self.chains {
            let theirs = other.chains.get(id)?;
            if mine.len() != theirs.len() {
                return None;
            }
            total += mine
                .chars()
                .zip(theirs.chars())
                .filter(|(a, b)| a != b)
                .count();
        }
        Some(total)
    }

    pub(crate) fn replace(&mut self, chain: ChainId, position: usize, residue: char) -> bool {
        match self.chains.get_mut(&chain) {
            None => false,
            Some(seq) => {
                let mut chars: Vec<char> = seq.chars().collect();
                match chars.get_mut(position) {
                    None => false,
                    Some(slot) => {
                        *slot = residue;
                        *seq = chars.into_iter().collect();
                        true
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TaskConfig;

    const VHH_YAML: &str = r#"
name: label-demo
binder:
  topology: vhh
  chains:
    - label: H
      sequence: ACDEFGHIKLMNPQRSTVWY
  regions:
    - name: loop1
      chain: H
      start: 5
      end: 8
  fixed_residues: ["H:10"]
target:
  chains:
    - label: A
      sequence: ACDEFGHIKLMNPQRSTVWY
design:
  cycles: 1
  batch_size: 1
"#;

    const PAIRED_YAML: &str = r#"
name: paired-label-demo
binder:
  topology: paired_chains
  chains:
    - label: H
      sequence: ACDEFGHIKLMNPQRSTVWY
    - label: L
      sequence: ACDEFGHIKLMNPQRSTVWY
  regions:
    - name: loopL
      chain: L
      start: 5
      end: 25
target:
  chains:
    - label: A
      sequence: ACDEFGHIKLMNPQRSTVWY
design:
  cycles: 1
  batch_size: 1
"#;

    fn scaffold_error(yaml: &str) -> String {
        let config = TaskConfig::from_yaml(yaml).expect("fixture parses");
        config
            .to_scaffold()
            .expect_err("fixture must fail")
            .to_string()
    }

    #[test]
    fn a_region_past_its_chain_names_the_label_the_yaml_used() {
        let msg = scaffold_error(&VHH_YAML.replace("end: 8", "end: 25"));
        assert!(msg.contains("chain 'H'"), "got: {msg}");
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn a_pinned_residue_past_its_chain_names_the_label_the_yaml_used() {
        let msg = scaffold_error(&VHH_YAML.replace("H:10", "H:25"));
        assert!(msg.contains("fixed residue H:"), "got: {msg}");
        assert!(!msg.contains("chain0"), "got: {msg}");
    }

    #[test]
    fn a_region_on_a_target_chain_names_that_chains_label() {
        let msg = scaffold_error(&VHH_YAML.replace("chain: H", "chain: A"));
        assert!(msg.contains("chain 'A'"), "got: {msg}");
        assert!(!msg.contains("chain1"), "got: {msg}");
    }

    #[test]
    fn with_two_binder_chains_the_error_names_the_one_that_rejected_the_region() {
        let msg = scaffold_error(PAIRED_YAML);
        assert!(msg.contains("chain 'L'"), "got: {msg}");
        assert!(!msg.contains("'H'"), "got: {msg}");
        assert!(!msg.contains("chain1"), "got: {msg}");
    }

    /// Twelve residues, so a bound of 13 is visibly past the end.
    const BINDER_12: &str = "QVQLVESGGGLV";
    const TARGET: &str = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG";

    fn region(start: usize, end: usize) -> Region {
        Region {
            name: "loop1".to_string(),
            chain: ChainId(0),
            start,
            end,
        }
    }

    /// The twelve-residue binder chain `H` and the target chain `A`, for
    /// scaffolds built by hand because no YAML can produce them.
    fn chains() -> Vec<Chain> {
        vec![
            Chain {
                id: ChainId(0),
                label: "H".to_string(),
                sequence: BINDER_12.to_string(),
                role: ChainRole::Binder,
            },
            Chain {
                id: ChainId(1),
                label: "A".to_string(),
                sequence: TARGET.to_string(),
                role: ChainRole::Target,
            },
        ]
    }

    /// A task whose binder chain `H` is twelve residues long. `binder_extra`
    /// is the block under test, written the way a user writes it, and must end
    /// in a newline.
    fn task_yaml(binder_extra: &str) -> String {
        format!(
            r#"name: numbering
binder:
  topology: vhh
  chains:
    - label: H
      sequence: {BINDER_12}
{binder_extra}target:
  chains:
    - label: A
      sequence: {TARGET}
design:
  cycles: 1
  batch_size: 1
"#
        )
    }

    /// The message a user sees when this config is turned into a scaffold.
    fn rejection(yaml: &str) -> String {
        let config = TaskConfig::from_yaml(yaml).expect("config parses");
        config
            .to_scaffold()
            .expect_err("scaffold must be rejected")
            .to_string()
    }

    #[test]
    fn a_stored_region_displays_with_the_start_incremented_and_the_end_unchanged() {
        // examples/vhh_demo.yaml writes loop1 as 26..33, and
        // `TaskConfig::to_scaffold` stores that as the half-open 25..33.
        // The half-open end already reads as the inclusive one-based
        // position, so only the start moves on the way out.
        let span = region(25, 33).display_span();
        assert_eq!((span.start, span.end), (26, 33));
        assert_eq!(span.to_string(), "26..33");
    }

    #[test]
    fn a_single_residue_region_displays_as_one_position() {
        // A YAML region of start 30, end 30 is stored as 29..30. It must
        // read back as 30..30, the case an off-by-one shows up in first.
        assert_eq!(region(29, 30).display_span().to_string(), "30..30");
    }

    #[test]
    fn a_stored_pinned_position_displays_incremented_by_one() {
        // A YAML `H:14` is stored as position 13. The display must show the
        // number the user wrote, so a dropped increment (13) and a doubled
        // one (15) both fail here.
        let fixed = FixedResidue {
            chain: ChainId(0),
            position: 13,
        };
        assert_eq!(fixed.display_position().to_string(), "14");
        assert_ne!(fixed.display_position().to_string(), "13");
        assert_ne!(fixed.display_position().to_string(), "15");
        // A chain's first residue is stored as 0 and must read as 1.
        let first = FixedResidue {
            chain: ChainId(0),
            position: 0,
        };
        assert_eq!(first.display_position().to_string(), "1");
    }

    #[test]
    fn a_region_past_the_end_of_its_chain_reports_the_bounds_the_config_wrote() {
        // The config writes 10..16 and `to_scaffold` stores 9..16. A user
        // searching their file for the reported span must find 10..16, and
        // must be told which bound to move.
        let message = rejection(&task_yaml(
            "  regions:\n    - name: loop1\n      chain: H\n      start: 10\n      end: 16\n",
        ));
        assert!(message.contains("spans 10..16"), "{message}");
        assert!(!message.contains("9..16"), "{message}");
        assert!(message.contains("has length 12"), "{message}");
        assert!(message.contains("lower end to 12 or less"), "{message}");
    }

    #[test]
    fn a_region_wholly_past_its_chain_tells_the_user_to_move_both_bounds() {
        // Start 20 is past residue 12, so lowering `end` alone cannot fix
        // this region and the advice has to say so.
        let message = rejection(&task_yaml(
            "  regions:\n    - name: loop1\n      chain: H\n      start: 20\n      end: 24\n",
        ));
        assert!(message.contains("spans 20..24"), "{message}");
        assert!(
            message.contains("lower start and end to 12 or less"),
            "{message}"
        );
    }

    #[test]
    fn a_pinned_residue_past_its_chain_reports_the_position_the_config_wrote() {
        // The config pins H:14 and `to_scaffold` stores 13. The message
        // quotes the YAML form, the way it does for a region span, and names
        // the chain by the label the config wrote rather than by its id.
        let message = rejection(&task_yaml("  fixed_residues: [\"H:14\"]\n"));
        assert!(message.contains("fixed residue H:14"), "{message}");
        assert!(!message.contains("H:13"), "{message}");
        assert!(!message.contains("chain0"), "{message}");
        assert!(
            message.contains("lower the position to 12 or less"),
            "{message}"
        );
    }

    #[test]
    fn a_pinned_residue_on_a_chain_the_scaffold_lacks_says_to_remove_the_entry() {
        // A chain length of 0 means the scaffold does not carry that chain
        // id, so no position would be in range and lowering one is not the
        // fix.
        let scaffold = Scaffold {
            topology: BinderTopology::Vhh,
            chains: chains(),
            regions: Vec::new(),
            fixed: vec![FixedResidue {
                chain: ChainId(7),
                position: 4,
            }],
        };
        let message = scaffold
            .validate()
            .expect_err("a pinned residue on an absent chain must be rejected")
            .to_string();
        assert!(
            message.contains("that chain has no residues, so remove the entry"),
            "{message}"
        );
        assert!(!message.contains("lower the position"), "{message}");
    }

    #[test]
    fn an_empty_region_tells_the_user_to_give_it_a_residue() {
        // Only a hand-built scaffold can carry this. `to_scaffold` rejects
        // `end < start` and stores `end == start` as a region of length one.
        let scaffold = Scaffold {
            topology: BinderTopology::Vhh,
            chains: chains(),
            regions: vec![region(5, 5)],
            fixed: Vec::new(),
        };
        let message = scaffold
            .validate()
            .expect_err("an empty region must be rejected")
            .to_string();
        assert_eq!(
            message,
            "region 'loop1' is empty; give it at least one residue"
        );
    }
}
