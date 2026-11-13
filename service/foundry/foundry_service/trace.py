"""The alpha-carbon trace builder.

The harness verifies the returned trace before it scores anything:
`verify_output` in `src/fold/mod.rs` rejects the candidate with a recorded
`sequence_mismatch` unless the trace carries exactly one entry per residue
sent, each chain's sorted indices are exactly 0..n, and the residues match
the sequence it sent.

So `build_ca_trace` emits one entry per input residue, unconditionally. It
walks the sequence the harness sent, not the atoms the model returned. A
residue the model resolved contributes its CA coordinate and its mean atom
pLDDT. A residue with no resolved atoms contributes xyz (0, 0, 0) and pLDDT
0.0 rather than being dropped: whether RF3 can omit residues is unverified,
and a dropped residue would fail `verify_output` for the whole candidate. The
zero coordinate and zero confidence say plainly that the model placed nothing
there.

The zeros carry a second cost, accepted with both costs in view. The TUI
projection (`project` in `src/tui/structure.rs`) and the web view
(`src/web/app.js`) both centre the picture on the centroid of every trace
residue and scale it by the largest extent, so a residue sitting at the
origin far from the model's placement drags the centroid and the scale for
the whole picture. Neither view filters unresolved residues out before
projecting the trace.

The zeros also pass the adapter's residue check. `check_residues` in
`adapter.py` fails a job only when a designed chain returns no atoms at all,
so a designed chain with one unresolved residue reaches the harness as zeros
without a recorded failure.

Residue letters come from the sent sequence by construction, so the letter
check in `verify_output` cannot fail on a trace built here. `adapter.py`
separately checks the model's own residue names against the sent sequence and
fails the job on a real mismatch, rather than papering over it with the trace.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contract import CaResidue, ChainSpec
from .derive import group_residues
from .rf3_output import Rf3Sample

UNRESOLVED_XYZ = (0.0, 0.0, 0.0)


def build_ca_trace(chains: Sequence[ChainSpec], sample: Rf3Sample) -> list[CaResidue]:
    """One `CaResidue` per residue of every chain, in chain then residue order.

    `chain_index` is the position of the chain in `chains`, which is the
    position in the harness's `job.chains`. That is the mapping
    `verify_output` expects: it compares chain i of the trace against chain i
    of the scaffold.
    """
    residues = group_residues(sample.atoms)

    trace: list[CaResidue] = []
    for chain_index, chain in enumerate(chains):
        for index, residue in enumerate(chain.sequence):
            group = residues.get((chain_index, index))
            if group:
                xyz = _representative_xyz(group)
                plddt = sum(atom.plddt for atom in group) / len(group)
            else:
                xyz = UNRESOLVED_XYZ
                plddt = 0.0
            trace.append(
                CaResidue(
                    chain_index=chain_index,
                    index=index,
                    residue=residue,
                    xyz=xyz,
                    plddt=plddt,
                    designed=chain.designed,
                )
            )
    return trace


def _representative_xyz(group: Sequence) -> tuple[float, float, float]:
    """The CA coordinate when present, else the centroid of the residue's atoms."""
    for atom in group:
        if atom.atom_name == "CA":
            return atom.xyz
    count = len(group)
    return (
        sum(atom.xyz[0] for atom in group) / count,
        sum(atom.xyz[1] for atom in group) / count,
        sum(atom.xyz[2] for atom in group) / count,
    )
