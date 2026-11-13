"""The internal shape one RF3 sample takes inside this service.

`rf3.py` converts foundry's `RF3Output` objects into these plain dataclasses,
and everything downstream (`derive.py`, `trace.py`, `adapter.py`) reads only
this shape. That is what makes the rest of the service testable without a GPU:
a test builds an `Rf3Sample` by hand instead of running a model.

Field provenance, at the pinned foundry commit recorded in `rf3.py`:

- `ptm`, `iptm`, `has_clash`, `ranking_score`, `overall_plddt`, `overall_pae`,
  `chain_pair_pae` come from `RF3Output.summary_confidences`, built by
  `compile_af3_style_confidence_outputs` in `models/rf3/src/rf3/utils/
  predicted_error.py` plus the metric merge in `models/rf3/src/rf3/
  inference_engines/rf3.py`.
- `atoms` comes from `RF3Output.atom_array` (coordinates, chain, residue and
  atom names) zipped with `RF3Output.confidences["atom_plddts"]`.

Two conversions already happened by the time data lands here, both in
`rf3.py`, and both are stated there: pLDDT is rescaled from RF3's 0..1 range
to the harness's 0..100 range, and residues are re-indexed from the input
rather than trusted from `confidences["token_res_ids"]`, which the foundry
source itself documents as sequential indexing rather than residue numbering.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    """One predicted heavy atom.

    `chain_index` is the position of the chain in the harness's `job.chains`.
    `res_index` is the zero-based position of the residue within that chain,
    derived from the sorted `res_id` values foundry reported for the chain.
    `plddt` is on the harness's 0..100 scale.
    """

    chain_index: int
    res_index: int
    res_name: str
    atom_name: str
    xyz: tuple[float, float, float]
    plddt: float


@dataclass(frozen=True)
class Rf3Sample:
    """One diffusion sample of one job, in the shape this service consumes.

    `iptm` is None when foundry reported no interface TM-score, which happens
    for single-chain inputs. `chain_pair_pae[i][j]` is the mean predicted
    aligned error in angstroms between chains i and j, or None where foundry
    left the matrix entry empty (the diagonal and the lower triangle).
    """

    example_id: str
    sample_idx: int
    seed: int
    ranking_score: float
    ptm: float
    iptm: float | None
    has_clash: bool
    overall_plddt: float
    overall_pae: float
    chain_pair_pae: tuple[tuple[float | None, ...], ...]
    atoms: tuple[Atom, ...]
