"""Metrics this adapter derives, because RF3 does not report them.

Honesty note required by `AGENTS.md`: two numbers the harness reads are
**adapter-derived, not model-reported**.

- `interface_plddt`: RF3 reports per-atom pLDDT (`atom_plddts`) but no
  interface pLDDT. This module averages per-residue pLDDT over the residues
  it identifies as interface residues from the predicted coordinates.
- `interface_contacts`: RF3 reports no contact count at all. This module
  counts inter-chain residue pairs whose closest heavy atoms are within
  `INTERFACE_CUTOFF_ANGSTROM`.

Three more numbers are derived here and flagged the same way in `adapter.py`:

- `clashes`: at the pinned commit, RF3's clash metric is a per-sample
  `has_clash` flag of 0 or 1 (`models/rf3/src/rf3/metrics/clashing_chains.py`),
  not a count. The harness field is a count and its gate default is
  `max_clashes: 2`, so mapping the flag to 0/1 would let a clashing structure
  pass. This module instead counts heavy-atom pairs closer than
  `CLASH_CUTOFF_ANGSTROM` (1.1 A, the distance foundry's `MIN_CLASH_DISTANCE`
  sets), and it counts inter-chain pairs only. Two limits follow. The
  1.1 A distance matches foundry's metric, but the chain scope is not shown
  to match. Whether foundry's `has_clash` flag also counts clashes within
  one chain is unverified, because no real fold has been run against a
  checkpoint.
  The derived count may therefore miss clashes foundry itself would flag, and
  a structure RF3 flags as clashing can report `clashes: 0` here and pass a
  `max_clashes: 2` gate. Until one real run settles the question, treat the
  gate as weaker than its name suggests.
- `hydrophobic_patch`: RF3 reports nothing about solvent exposure, and this
  service computes no SASA. `hydrophobic_fraction` is a sequence-composition
  proxy, documented as such wherever the number surfaces.
- `pae_interface`: RF3 reports the chain-pair PAE matrix but no single
  interface PAE. `pae_interface` below averages the matrix entries across the
  designed/target boundary, with the documented fallbacks. The number is a
  mean the adapter chose to take over a model-reported matrix, not a
  model-reported metric.

`net_charge` is the one field computed exactly: the formal side-chain charge
of the designed sequence at neutral pH.

All functions are pure. They read `Rf3Sample.atoms` and sequences, and they
use no network, no filesystem and no torch.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from .rf3_output import Atom

# A residue pair counts as an interface contact when its closest heavy-atom
# pair is within this distance. 5 A between heavy atoms is the standard
# structural-biology definition of a residue-residue contact, and the same
# cutoff defines the interface mask for `interface_plddt`, so the two derived
# numbers describe one consistent interface.
INTERFACE_CUTOFF_ANGSTROM = 5.0

# The distance foundry's own clash metric uses (MIN_CLASH_DISTANCE in
# models/rf3/src/rf3/metrics/clashing_chains.py at the pinned commit). The
# derived count applies it to inter-chain pairs only; whether foundry's
# `has_clash` flag also covers intra-chain pairs is unverified, because no
# real fold has been run. See the module docstring.
CLASH_CUTOFF_ANGSTROM = 1.1

# Formal side-chain charge at neutral pH. Histidine is treated as neutral.
CHARGED_RESIDUES = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}

# Kyte-Doolitch hydropathy-positive set. This defines the composition proxy
# for `hydrophobic_patch`; it is not a measurement of exposed surface.
HYDROPHOBIC_RESIDUES = frozenset("AVLIMFWY")

# Three-letter to one-letter names for the standard residues, used to check
# the model's residues against the sequence the harness sent.
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

ResKey = tuple[int, int]


def group_residues(atoms: Iterable[Atom]) -> dict[ResKey, list[Atom]]:
    """Atoms grouped by (chain_index, res_index), insertion order preserved."""
    residues: dict[ResKey, list[Atom]] = defaultdict(list)
    for atom in atoms:
        residues[(atom.chain_index, atom.res_index)].append(atom)
    return dict(residues)


def residue_plddts(atoms: Iterable[Atom]) -> dict[ResKey, float]:
    """Mean pLDDT (0..100) per residue, the mean over that residue's atoms."""
    grouped = group_residues(atoms)
    return {key: sum(atom.plddt for atom in group) / len(group) for key, group in grouped.items()}


def _residue_bounds(
    residues: dict[ResKey, list[Atom]],
) -> dict[ResKey, tuple[float, float, float, float]]:
    """Per residue, the centroid (x, y, z) and the max atom distance from it.

    The centroid plus radius bounds every atom of the residue, so two residues
    whose bounds are further apart than the cutoff cannot hold a contact. That
    prunes the quadratic atom-pair search to spatially plausible pairs.
    """
    bounds: dict[ResKey, tuple[float, float, float, float]] = {}
    for key, group in residues.items():
        count = len(group)
        cx = sum(atom.xyz[0] for atom in group) / count
        cy = sum(atom.xyz[1] for atom in group) / count
        cz = sum(atom.xyz[2] for atom in group) / count
        radius = max(math.dist((cx, cy, cz), atom.xyz) for atom in group)
        bounds[key] = (cx, cy, cz, radius)
    return bounds


def _min_pair_distance_sq(group_a: Sequence[Atom], group_b: Sequence[Atom]) -> float:
    best = math.inf
    for a in group_a:
        for b in group_b:
            dx = a.xyz[0] - b.xyz[0]
            dy = a.xyz[1] - b.xyz[1]
            dz = a.xyz[2] - b.xyz[2]
            squared = dx * dx + dy * dy + dz * dz
            if squared < best:
                best = squared
    return best


def interface(
    atoms: Iterable[Atom], cutoff: float = INTERFACE_CUTOFF_ANGSTROM
) -> tuple[set[ResKey], int]:
    """The interface mask and the contact count in one pass.

    Returns `(interface_residues, contacts)` where `interface_residues` holds
    every residue with at least one atom within `cutoff` angstroms of an atom
    in a different chain, and `contacts` is the number of such inter-chain
    residue pairs. A pair is counted once, so `interface_contacts` is a pair
    count rather than a per-chain sum.
    """
    residues = group_residues(atoms)
    bounds = _residue_bounds(residues)
    keys = list(residues)

    interface_residues: set[ResKey] = set()
    contacts = 0
    cutoff_sq = cutoff * cutoff

    for i, key_a in enumerate(keys):
        chain_a, _ = key_a
        ax, ay, az, radius_a = bounds[key_a]
        for key_b in keys[i + 1 :]:
            chain_b, _ = key_b
            if chain_a == chain_b:
                continue
            bx, by, bz, radius_b = bounds[key_b]
            reach = radius_a + radius_b + cutoff
            dx, dy, dz = ax - bx, ay - by, az - bz
            if dx * dx + dy * dy + dz * dz > reach * reach:
                continue
            if _min_pair_distance_sq(residues[key_a], residues[key_b]) <= cutoff_sq:
                contacts += 1
                interface_residues.add(key_a)
                interface_residues.add(key_b)
    return interface_residues, contacts


def interface_plddt(atoms: Iterable[Atom], cutoff: float = INTERFACE_CUTOFF_ANGSTROM) -> float:
    """Mean per-residue pLDDT over interface residues, 0..100.

    Adapter-derived, not model-reported. Returns 0.0 when no residue is
    within the cutoff of another chain, which is honest: there is no
    interface to be confident about, and the harness gate on
    `min_interface_plddt` rejects the candidate.
    """
    atom_list = list(atoms)
    mask, _ = interface(atom_list, cutoff)
    if not mask:
        return 0.0
    per_residue = residue_plddts(atom_list)
    return sum(per_residue[key] for key in mask) / len(mask)


def interface_contacts(atoms: Iterable[Atom], cutoff: float = INTERFACE_CUTOFF_ANGSTROM) -> int:
    """Count of inter-chain residue pairs under the contact cutoff.

    Adapter-derived, not model-reported.
    """
    _, contacts = interface(atoms, cutoff)
    return contacts


def clash_count(atoms: Iterable[Atom], cutoff: float = CLASH_CUTOFF_ANGSTROM) -> int:
    """Count of inter-chain heavy-atom pairs closer than `cutoff`.

    Adapter-derived, not model-reported. RF3 reports a per-sample `has_clash`
    flag of 0 or 1, not a count. The 1.1 A cutoff is the distance foundry's
    `MIN_CLASH_DISTANCE` sets, but this count covers inter-chain pairs only,
    and whether foundry's flag also counts intra-chain clashes is unverified
    because no real fold has been run. The count may therefore sit below what
    foundry itself would flag. See the module docstring.
    """
    atom_list = list(atoms)
    residues = group_residues(atom_list)
    bounds = _residue_bounds(residues)
    keys = list(residues)

    total = 0
    cutoff_sq = cutoff * cutoff
    for i, key_a in enumerate(keys):
        chain_a, _ = key_a
        ax, ay, az, radius_a = bounds[key_a]
        for key_b in keys[i + 1 :]:
            chain_b, _ = key_b
            if chain_a == chain_b:
                continue
            bx, by, bz, radius_b = bounds[key_b]
            reach = radius_a + radius_b + cutoff
            dx, dy, dz = ax - bx, ay - by, az - bz
            if dx * dx + dy * dy + dz * dz > reach * reach:
                continue
            for a in residues[key_a]:
                for b in residues[key_b]:
                    ddx = a.xyz[0] - b.xyz[0]
                    ddy = a.xyz[1] - b.xyz[1]
                    ddz = a.xyz[2] - b.xyz[2]
                    if ddx * ddx + ddy * ddy + ddz * ddz < cutoff_sq:
                        total += 1
    return total


def net_charge(sequence: str) -> float:
    """Formal side-chain charge at neutral pH. Exact from the sequence."""
    return sum(CHARGED_RESIDUES.get(residue, 0.0) for residue in sequence)


def hydrophobic_fraction(sequence: str) -> float:
    """Fraction of residues in the Kyte-Doolitch hydropathy-positive set.

    Sequence-composition proxy for the harness's `hydrophobic_patch`, which
    asks for the largest exposed hydrophobic patch normalised 0..1. This
    service computes no solvent accessibility, so it reports composition and
    says so. An empty sequence gives 0.0.
    """
    if not sequence:
        return 0.0
    hits = sum(1 for residue in sequence if residue in HYDROPHOBIC_RESIDUES)
    return hits / len(sequence)


def pae_interface(
    chain_pair_pae: Sequence[Sequence[float | None]],
    designed_indices: Iterable[int],
) -> float | None:
    """Mean chain-pair PAE across the designed/target boundary, in angstroms.

    Adapter-derived, not model-reported. RF3 reports the chain-pair PAE
    matrix but no single interface PAE. This averages the entries where
    exactly one endpoint is a designed chain, which is the boundary the
    harness scores. When no such entry exists (for example a single-chain
    job), it falls back to the mean over every non-empty entry. That fallback
    is a whole-complex mean, not an interface-specific quantity, and it is
    reported under the same field name. When the matrix is empty this returns
    None, and `adapter.confidence_for` then falls back to RF3's
    `overall_pae`, which is likewise not interface-specific. Both fallbacks
    are disclosed in `adapter.py`'s provenance list and in the service
    README.
    """
    designed = set(designed_indices)
    boundary: list[float] = []
    everything: list[float] = []
    for i, row in enumerate(chain_pair_pae):
        for j, value in enumerate(row):
            if value is None:
                continue
            everything.append(value)
            if (i in designed) != (j in designed):
                boundary.append(value)
    if boundary:
        return sum(boundary) / len(boundary)
    if everything:
        return sum(everything) / len(everything)
    return None
