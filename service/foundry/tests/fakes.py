"""Hand-written fakes. No mocking library, per `AGENTS.md`.

`FakeRf3Output` mimics foundry's `RF3Output` (a dataclass with an
`atom_array` carrying biotite-style annotations plus two confidence dicts),
which is what `rf3.convert_sample` duck-types against. The shapes come from
`models/rf3/src/rf3/inference_engines/rf3.py` and the regression fixture
`models/rf3/tests/data/inference_regression_tests/8vkf_from_file/
8vkf_from_file_summary_confidences.json` at the pinned commit.
"""

from __future__ import annotations

from foundry_service.contract import ChainSpec, FoldResult, JobSpec, MoleculeKind
from foundry_service.rf3_output import Atom, Rf3Sample


def chain(label: str, sequence: str, designed: bool) -> ChainSpec:
    return ChainSpec(label=label, sequence=sequence, kind=MoleculeKind.PROTEIN, designed=designed)


def job_spec(
    binder: str = "QVQL",
    target: str = "MKTV",
    *,
    model: str = "rf3",
    seeds: tuple[int, ...] = (1,),
    diffusion_samples: int = 2,
    diffusion_steps: int = 8,
    use_msa: bool = False,
    need_atom_confidence: bool = True,
    name: str = "fake-job",
) -> JobSpec:
    chains = []
    if binder is not None:
        chains.append(chain("H", binder, designed=True))
    if target is not None:
        chains.append(chain("A", target, designed=False))
    return JobSpec(
        name=name,
        model=model,
        chains=chains,
        seeds=list(seeds),
        diffusion_samples=diffusion_samples,
        diffusion_steps=diffusion_steps,
        use_msa=use_msa,
        need_atom_confidence=need_atom_confidence,
    )


def atom(
    chain_index: int,
    res_index: int,
    atom_name: str,
    xyz: tuple[float, float, float],
    plddt: float,
    res_name: str = "ALA",
) -> Atom:
    return Atom(
        chain_index=chain_index,
        res_index=res_index,
        res_name=res_name,
        atom_name=atom_name,
        xyz=xyz,
        plddt=plddt,
    )


def sample(
    atoms: tuple[Atom, ...] | list[Atom] = (),
    *,
    ptm: float = 0.9,
    iptm: float | None = 0.8,
    has_clash: bool = False,
    ranking_score: float | None = None,
    overall_plddt: float = 80.0,
    overall_pae: float = 4.0,
    chain_pair_pae: tuple[tuple[float | None, ...], ...] = ((None, 4.0), (None, None)),
    example_id: str = "fake-job",
    sample_idx: int = 0,
    seed: int = 1,
) -> Rf3Sample:
    if ranking_score is None:
        ranking_score = 0.8 * (ptm if iptm is None else iptm) + 0.2 * ptm - 100 * int(has_clash)
    return Rf3Sample(
        example_id=example_id,
        sample_idx=sample_idx,
        seed=seed,
        ranking_score=ranking_score,
        ptm=ptm,
        iptm=iptm,
        has_clash=has_clash,
        overall_plddt=overall_plddt,
        overall_pae=overall_pae,
        chain_pair_pae=chain_pair_pae,
        atoms=tuple(atoms),
    )


class FakeAtomArray:
    """Stands in for foundry's predicted AtomArray (biotite-style annotations)."""

    def __init__(
        self,
        chain_id: list[str],
        res_id: list[int],
        res_name: list[str],
        atom_name: list[str],
        coord: list[tuple[float, float, float]],
    ) -> None:
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        self.atom_name = atom_name
        self.coord = coord


class FakeRf3Output:
    """Stands in for foundry's RF3Output dataclass."""

    def __init__(
        self,
        atom_array: FakeAtomArray,
        summary_confidences: dict,
        confidences: dict | None,
        example_id: str = "fake-job",
        sample_idx: int = 0,
        seed: int = 1,
    ) -> None:
        self.example_id = example_id
        self.atom_array = atom_array
        self.summary_confidences = summary_confidences
        self.confidences = confidences
        self.sample_idx = sample_idx
        self.seed = seed


def fake_output(
    *,
    chain_ids: list[str],
    res_ids: list[int],
    res_names: list[str],
    atom_names: list[str],
    coords: list[tuple[float, float, float]],
    atom_plddts: list[float] | None,
    ptm: float = 0.9,
    iptm: float | None = 0.8,
    has_clash: bool = False,
    overall_plddt: float = 0.8,
    overall_pae: float = 4.0,
    chain_pair_pae: list[list[float | None]] | None = None,
    ranking_score: float | None = 0.76,
    token_res_ids: list[int] | None = None,
    sample_idx: int = 0,
) -> FakeRf3Output:
    """A two-chain-shaped foundry output built from parallel lists.

    `atom_plddts` are on foundry's 0..1 scale, and `overall_plddt` likewise,
    matching the regression fixture at the pinned commit.
    """
    if chain_pair_pae is None:
        chain_pair_pae = [[None, 3.0], [None, None]]
    summary: dict = {
        "chain_ptm": [overall_plddt, overall_plddt],
        "chain_pair_pae": chain_pair_pae,
        "overall_plddt": overall_plddt,
        "overall_pae": overall_pae,
        "ptm": ptm,
        "iptm": iptm,
        "has_clash": has_clash,
    }
    if ranking_score is not None:
        summary["ranking_score"] = ranking_score
    confidences: dict | None = None
    if atom_plddts is not None:
        confidences = {
            "atom_chain_ids": list(chain_ids),
            "atom_plddts": list(atom_plddts),
            "token_chain_ids": list(chain_ids),
            # Sequential token indexing, the trap this service must ignore.
            "token_res_ids": token_res_ids
            if token_res_ids is not None
            else list(range(len(chain_ids))),
        }
    return FakeRf3Output(
        atom_array=FakeAtomArray(chain_ids, res_ids, res_names, atom_names, coords),
        summary_confidences=summary,
        confidences=confidences,
        sample_idx=sample_idx,
    )


def fold_result(job_id: str = "job-000001", model: str = "rf3") -> FoldResult:
    """A minimal valid FoldResult for server and queue tests."""
    from foundry_service.contract import (
        CaResidue,
        ConfidenceReport,
        StructureFormat,
        StructurePayload,
    )

    return FoldResult(
        job_id=job_id,
        model=model,
        confidence=ConfidenceReport(
            iptm=0.8,
            ptm=0.9,
            plddt_mean=80.0,
            interface_plddt=72.5,
            pae_interface=3.0,
            interface_contacts=1,
            clashes=0,
            hydrophobic_patch=0.5,
            net_charge=-1.0,
            plddt=[80.0, 80.0],
            atom_confidence=True,
        ),
        structure=StructurePayload(
            format=StructureFormat.CA_TRACE,
            inline=None,
            artifact_url=None,
            ca_trace=[
                CaResidue(
                    chain_index=0,
                    index=index,
                    residue=residue,
                    xyz=(float(index), 0.0, 0.0),
                    plddt=80.0,
                    designed=True,
                )
                for index, residue in enumerate("QV")
            ],
        ),
        elapsed_ms=7,
    )
