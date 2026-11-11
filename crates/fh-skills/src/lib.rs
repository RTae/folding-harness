//! Design skills: the actions that turn a parent candidate into proposals.
//!
//! A skill never edits a sequence. It assembles substitutions and hands them to
//! [`fh_core::MutableScope::propose`], which validates them or refuses. A buggy
//! or adversarial skill therefore cannot mutate a framework residue — the worst
//! it can do is get an error back.

pub mod router;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;

use fh_core::{
    Candidate, MutableScope, Proposal, ProposalError, SearchState, SkillId, Substitution,
    AMINO_ACIDS,
};

pub use router::{Legality, Router, IMPLEMENTED};

pub struct SkillContext<'a> {
    pub scope: &'a MutableScope,
    pub parent: &'a Candidate,
    pub cycle: u32,
    /// How many proposals to return.
    pub batch_size: usize,
    /// Cap on substitutions in a single proposal, for skills that sample.
    pub max_substitutions: usize,
    pub state: &'a SearchState,
}

pub trait DesignSkill: Send + Sync {
    fn id(&self) -> SkillId;

    /// One line for `fh skills` and the UI.
    fn describe(&self) -> &'static str;

    /// The RNG is passed in rather than held so a run is reproducible from the
    /// task seed alone.
    fn propose(
        &self,
        ctx: &SkillContext<'_>,
        rng: &mut StdRng,
    ) -> Result<Vec<Proposal>, ProposalError>;
}

/// Pick a standard residue that differs from the current one, so a proposal is
/// never a no-op.
fn different_residue(current: Option<char>, rng: &mut StdRng) -> char {
    loop {
        let pick = AMINO_ACIDS[rng.gen_range(0..AMINO_ACIDS.len())];
        if Some(pick) != current {
            return pick;
        }
    }
}

fn residue_at(ctx: &SkillContext<'_>, chain: fh_core::ChainId, position: usize) -> Option<char> {
    ctx.parent
        .sequence
        .chain(chain)
        .and_then(|seq| seq.chars().nth(position))
}

/// Substitute a few positions anywhere in the mutable scope. The conservative
/// move: small steps that keep most of a working candidate.
pub struct PointMutation;

impl DesignSkill for PointMutation {
    fn id(&self) -> SkillId {
        SkillId::PointMutation
    }

    fn describe(&self) -> &'static str {
        "substitute up to `max_substitutions` positions sampled from the mutable scope"
    }

    fn propose(
        &self,
        ctx: &SkillContext<'_>,
        rng: &mut StdRng,
    ) -> Result<Vec<Proposal>, ProposalError> {
        let positions = ctx.scope.positions();
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        let cap = ctx.max_substitutions.clamp(1, positions.len());
        let mut out = Vec::with_capacity(ctx.batch_size);

        for _ in 0..ctx.batch_size {
            let count = rng.gen_range(1..=cap);
            let chosen: Vec<(fh_core::ChainId, usize)> =
                positions.choose_multiple(rng, count).copied().collect();

            let substitutions: Vec<Substitution> = chosen
                .iter()
                .map(|(chain, position)| Substitution {
                    chain: *chain,
                    position: *position,
                    to: different_residue(residue_at(ctx, *chain, *position), rng),
                })
                .collect();

            let rationale = format!(
                "{count} substitution(s) sampled from {} mutable positions",
                positions.len()
            );
            out.push(ctx.scope.propose(
                Some(ctx.parent.id),
                SkillId::PointMutation,
                substitutions,
                rationale,
            )?);
        }

        Ok(out)
    }
}

/// Replace every mutable position of one region at once. Disruptive, so the
/// router only offers it while bootstrapping or when refinement has stalled.
pub struct RegionRedesign;

impl DesignSkill for RegionRedesign {
    fn id(&self) -> SkillId {
        SkillId::RegionRedesign
    }

    fn describe(&self) -> &'static str {
        "replace all mutable positions of one region, leaving other regions alone"
    }

    fn propose(
        &self,
        ctx: &SkillContext<'_>,
        rng: &mut StdRng,
    ) -> Result<Vec<Proposal>, ProposalError> {
        let names: Vec<String> = ctx
            .scope
            .regions()
            .iter()
            .map(|region| region.name.clone())
            .filter(|name| !ctx.scope.region_positions(name).is_empty())
            .collect();

        if names.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(ctx.batch_size);
        for _ in 0..ctx.batch_size {
            let name = match names.choose(rng) {
                Some(name) => name.clone(),
                None => continue,
            };
            let positions = ctx.scope.region_positions(&name);
            let substitutions: Vec<Substitution> = positions
                .iter()
                .map(|(chain, position)| Substitution {
                    chain: *chain,
                    position: *position,
                    to: different_residue(residue_at(ctx, *chain, *position), rng),
                })
                .collect();

            let rationale = format!(
                "full redesign of region '{name}' ({} mutable positions)",
                positions.len()
            );
            out.push(ctx.scope.propose(
                Some(ctx.parent.id),
                SkillId::RegionRedesign,
                substitutions,
                rationale,
            )?);
        }

        Ok(out)
    }
}

/// Executor for a skill id, or `None` when this build cannot run it.
/// [`Router`] checks the same list, so the agent is never offered a skill that
/// would land here as `None`.
pub fn executor(id: SkillId) -> Option<Box<dyn DesignSkill>> {
    match id {
        SkillId::PointMutation => Some(Box::new(PointMutation)),
        SkillId::RegionRedesign => Some(Box::new(RegionRedesign)),
        // These need a model on the compute side. The contract and the router
        // already account for them; the executors are not written yet.
        SkillId::InverseFolding | SkillId::LikelihoodGuided => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    use fh_core::{BinderSequence, CandidateId, TaskConfig};

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
"#;

    struct Fixture {
        scope: MutableScope,
        parent: Candidate,
        state: SearchState,
    }

    fn fixture() -> Fixture {
        let config = TaskConfig::from_yaml(YAML).expect("parses");
        let (scaffold, scope) = config.resolve().expect("resolves");
        let parent = Candidate::founder(CandidateId(1), BinderSequence::from_scaffold(&scaffold));
        Fixture {
            scope,
            parent,
            state: SearchState::bootstrap(),
        }
    }

    fn context<'a>(fixture: &'a Fixture, batch_size: usize) -> SkillContext<'a> {
        SkillContext {
            scope: &fixture.scope,
            parent: &fixture.parent,
            cycle: 1,
            batch_size,
            max_substitutions: 3,
            state: &fixture.state,
        }
    }

    #[test]
    fn point_mutation_stays_inside_the_scope() {
        let fixture = fixture();
        let ctx = context(&fixture, 25);
        let mut rng = StdRng::seed_from_u64(42);
        let proposals = PointMutation.propose(&ctx, &mut rng).expect("proposes");

        assert_eq!(proposals.len(), 25);
        for proposal in &proposals {
            assert!(!proposal.is_empty());
            assert!(proposal.len() <= 3);
            for substitution in proposal.substitutions() {
                assert!(
                    fixture
                        .scope
                        .is_mutable(substitution.chain, substitution.position),
                    "proposed {:?} outside the mutable scope",
                    substitution
                );
                // H:30 one-based is index 29 and is pinned.
                assert_ne!(substitution.position, 29);
            }
        }
    }

    #[test]
    fn point_mutation_never_proposes_a_no_op() {
        let fixture = fixture();
        let ctx = context(&fixture, 40);
        let mut rng = StdRng::seed_from_u64(7);
        for proposal in PointMutation.propose(&ctx, &mut rng).expect("proposes") {
            let child = proposal.apply_to(&fixture.parent.sequence).expect("applies");
            assert_eq!(
                fixture.parent.sequence.hamming(&child),
                Some(proposal.len())
            );
        }
    }

    #[test]
    fn region_redesign_touches_exactly_one_region() {
        let fixture = fixture();
        let ctx = context(&fixture, 12);
        let mut rng = StdRng::seed_from_u64(99);
        let proposals = RegionRedesign.propose(&ctx, &mut rng).expect("proposes");

        for proposal in &proposals {
            let regions: Vec<&str> = fixture
                .scope
                .regions()
                .iter()
                .filter(|region| {
                    proposal
                        .substitutions()
                        .iter()
                        .any(|s| s.chain == region.chain && region.contains(s.position))
                })
                .map(|region| region.name.as_str())
                .collect();
            assert_eq!(regions.len(), 1, "expected one region, got {regions:?}");

            let expected = fixture.scope.region_positions(regions[0]).len();
            assert_eq!(proposal.len(), expected);
        }
    }

    #[test]
    fn the_same_seed_produces_the_same_proposals() {
        let fixture = fixture();
        let ctx = context(&fixture, 6);
        let run = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            PointMutation
                .propose(&ctx, &mut rng)
                .expect("proposes")
                .iter()
                .map(|p| p.substitutions().to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn unimplemented_skills_have_no_executor() {
        assert!(executor(SkillId::PointMutation).is_some());
        assert!(executor(SkillId::RegionRedesign).is_some());
        assert!(executor(SkillId::InverseFolding).is_none());
        assert!(executor(SkillId::LikelihoodGuided).is_none());
    }
}
