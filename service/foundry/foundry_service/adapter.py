"""From an RF3 sample to the harness's `FoldResult`.

Which numbers are model-reported and which are adapter-derived
(`AGENTS.md` requires this to be stated where a reader of the numbers finds
it):

- Model-reported by RF3: `ptm`, `iptm`, `plddt_mean` (from `overall_plddt`,
  rescaled 0..1 to 0..100), and every per-atom pLDDT behind `plddt` and the
  trace.
- Adapter-derived, NOT model-reported: `interface_plddt`,
  `interface_contacts`, `clashes`, `hydrophobic_patch`, and `pae_interface`.
  RF3 reports none of the five as a harness-ready number. `derive.py`
  computes the first three from predicted coordinates, the fourth as a
  sequence-composition proxy, and `pae_interface` as the adapter's mean over
  the chain-pair PAE matrix RF3 reports. The mean covers the entries across
  the designed/target boundary, falls back to the mean over every entry of
  the matrix when no boundary entry exists, and falls back to RF3's
  `overall_pae` when the matrix is empty. Both fallbacks are whole-complex
  numbers reported under an interface-specific field name. The exact list is
  `ADAPTER_DERIVED_FIELDS`.
- `clashes` counts inter-chain heavy-atom pairs closer than 1.1 A only.
  RF3's own `has_clash` is a per-sample flag of 0 or 1, not a count, and
  whether it also covers intra-chain clashes is unverified, because no real
  fold has been run. The derived count may therefore sit below what foundry
  would flag, and until one real run settles the question the harness's
  `max_clashes` gate is weaker than its name suggests. See the `derive.py`
  module docstring.
- `net_charge` is computed exactly from the designed sequence.

Every rejection names what failed and why, so a failed job reaches the
harness with a reason instead of a malformed candidate.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import derive
from .contract import (
    CaResidue,
    ConfidenceReport,
    FoldResult,
    JobSpec,
    StructureFormat,
    StructurePayload,
)
from .rf3_output import Rf3Sample
from .trace import build_ca_trace

ADAPTER_DERIVED_FIELDS = (
    "interface_plddt",
    "interface_contacts",
    "clashes",
    "hydrophobic_patch",
    "pae_interface",
)


class AdapterError(Exception):
    """The model ran, but its output cannot honestly become a FoldResult."""


def pick_best_sample(samples: Sequence[Rf3Sample]) -> Rf3Sample:
    """The sample with the highest foundry ranking score.

    foundry's ranking score is 0.8*ipTM + 0.2*pTM - 100*has_clash, computed
    at the pinned commit in `models/rf3/src/rf3/inference_engines/rf3.py`.
    """
    if not samples:
        raise AdapterError("RF3 returned no samples, so there is no structure to report")
    return max(samples, key=lambda sample: sample.ranking_score)


def check_residues(job: JobSpec, sample: Rf3Sample) -> None:
    """Fail the job when the model's residues disagree with the sent sequence.

    Missing residues are allowed: `trace.build_ca_trace` fills them in and
    marks them with zero confidence. Extra residues or a different residue
    name at a position are a real disagreement, and the harness would reject
    the candidate as `sequence_mismatch` anyway. Failing here records the
    reason on this side instead.

    Non-standard residue names (ligands, modified residues) are not checked
    against the sequence, because this service only sends standard protein
    residues and anything else back means foundry changed the molecule, which
    the count check still catches.
    """
    per_chain_returned: dict[int, dict[int, str]] = {}
    for atom in sample.atoms:
        per_chain_returned.setdefault(atom.chain_index, {})[atom.res_index] = atom.res_name

    for chain_index, chain in enumerate(job.chains):
        returned = per_chain_returned.get(chain_index, {})
        if not returned:
            if chain.designed:
                raise AdapterError(
                    f"chain {chain.label}: RF3 returned no atoms for this designed chain, "
                    f"so the trace would be zeros for all {len(chain.sequence)} residues"
                )
            continue
        extra = [index for index in returned if index >= len(chain.sequence)]
        if extra:
            raise AdapterError(
                f"chain {chain.label}: RF3 returned {len(returned)} residues but the harness "
                f"sent {len(chain.sequence)}; extra residue indices {sorted(extra)}"
            )
        for index, res_name in sorted(returned.items()):
            one_letter = derive.THREE_TO_ONE.get(res_name)
            if one_letter is None:
                continue
            sent = chain.sequence[index]
            if one_letter != sent:
                raise AdapterError(
                    f"chain {chain.label} position {index + 1}: the harness sent {sent}, "
                    f"RF3 returned {res_name} ({one_letter})"
                )


def confidence_for(job: JobSpec, sample: Rf3Sample) -> ConfidenceReport:
    """Assemble the wire report. Derived fields are computed, never invented."""
    if len(job.chains) > 1 and sample.iptm is None:
        raise AdapterError(
            "RF3 reported no ipTM for a multi-chain job; refusing to pass a "
            "complex-level number off as an interface one"
        )
    if job.need_atom_confidence and not sample.atoms:
        raise AdapterError(
            "the job set need_atom_confidence but RF3 returned no per-atom pLDDT; "
            "the contract requires failing rather than returning complex-level numbers only"
        )

    designed = [chain for chain in job.chains if chain.designed]
    designed_indices = [index for index, chain in enumerate(job.chains) if chain.designed]
    designed_sequence = "".join(chain.sequence for chain in designed)

    # Derived, not model-reported: the adapter's mean over RF3's chain-pair
    # PAE matrix. The overall_pae fallback fires only when the matrix is
    # empty, and reports a whole-complex number under an interface-specific
    # field name; the module docstring discloses both fallbacks.
    pae = derive.pae_interface(sample.chain_pair_pae, designed_indices)
    if pae is None:
        pae = sample.overall_pae

    mask, contacts = derive.interface(sample.atoms)
    per_residue = derive.residue_plddts(sample.atoms)
    interface_plddt = sum(per_residue[key] for key in mask) / len(mask) if mask else 0.0

    return ConfidenceReport(
        iptm=sample.iptm if sample.iptm is not None else sample.ptm,
        ptm=sample.ptm,
        plddt_mean=sample.overall_plddt,
        interface_plddt=interface_plddt,
        pae_interface=pae,
        interface_contacts=contacts,
        clashes=derive.clash_count(sample.atoms),
        hydrophobic_patch=derive.hydrophobic_fraction(designed_sequence),
        net_charge=derive.net_charge(designed_sequence),
        plddt=designed_plddt(job, sample, per_residue),
        atom_confidence=bool(sample.atoms),
    )


def designed_plddt(
    job: JobSpec,
    sample: Rf3Sample,
    per_residue: dict[tuple[int, int], float] | None = None,
) -> list[float]:
    """Per-residue pLDDT for the designed chains, in chain then residue order.

    One entry per sent residue, so the vector lines up with the sequence the
    harness sent. A residue the model did not resolve gets 0.0.
    """
    if per_residue is None:
        per_residue = derive.residue_plddts(sample.atoms)
    values: list[float] = []
    for chain_index, chain in enumerate(job.chains):
        if not chain.designed:
            continue
        for index in range(len(chain.sequence)):
            values.append(per_residue.get((chain_index, index), 0.0))
    return values


def fold_result(
    job_id: str,
    job: JobSpec,
    samples: Sequence[Rf3Sample],
    elapsed_ms: int,
) -> FoldResult:
    """The complete result for one job: pick, check, derive, build the trace."""
    sample = pick_best_sample(samples)
    check_residues(job, sample)
    trace: list[CaResidue] = build_ca_trace(job.chains, sample)
    return FoldResult(
        job_id=job_id,
        model=job.model,
        confidence=confidence_for(job, sample),
        structure=StructurePayload(
            format=StructureFormat.CA_TRACE,
            inline=None,
            artifact_url=None,
            ca_trace=trace,
        ),
        elapsed_ms=elapsed_ms,
    )
