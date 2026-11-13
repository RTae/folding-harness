"""The derived metrics against hand-computed arithmetic.

Every expected number in this file is computed by hand in the test's
docstring or comments, so a reader can check the derivation without running
anything.
"""

from __future__ import annotations

import pytest

from foundry_service import derive
from tests.fakes import atom


def test_interface_plddt_and_contacts_match_hand_arithmetic():
    """Hand arithmetic, cutoff 5.0 A:

    Chain 0 residue 0: CA (0,0,0) pLDDT 80, CB (1,0,0) pLDDT 90.
      Residue mean = (80 + 90) / 2 = 85.
    Chain 0 residue 1: CA (0,20,0) pLDDT 50. Far from chain 1 (>= 15 A).
    Chain 1 residue 0: CA (4,0,0) pLDDT 60.
      Closest pair to chain 0 residue 0: (1,0,0)-(4,0,0) = 3.0 <= 5.0, so
      this is the one contact and both residues are interface residues.
    Chain 1 residue 1: CA (20,0,0) pLDDT 40. Closest chain-0 atom is 19 A
      away ((1,0,0) to (20,0,0)), so it is not an interface residue.

    interface contacts = 1  (the pair (0,0)-(1,0), counted once)
    interface pLDDT     = mean(85, 60) = 145 / 2 = 72.5
    """
    atoms = [
        atom(0, 0, "CA", (0.0, 0.0, 0.0), 80.0),
        atom(0, 0, "CB", (1.0, 0.0, 0.0), 90.0),
        atom(0, 1, "CA", (0.0, 20.0, 0.0), 50.0),
        atom(1, 0, "CA", (4.0, 0.0, 0.0), 60.0),
        atom(1, 1, "CA", (20.0, 0.0, 0.0), 40.0),
    ]
    mask, contacts = derive.interface(atoms)
    assert contacts == 1
    assert mask == {(0, 0), (1, 0)}
    assert derive.interface_plddt(atoms) == pytest.approx(72.5)
    assert derive.interface_contacts(atoms) == 1


def test_the_contact_cutoff_boundary_is_inclusive_at_five_angstroms():
    # Exactly 5.0 A apart counts; 5.01 A does not.
    at_cutoff = [atom(0, 0, "CA", (0.0, 0.0, 0.0), 70.0), atom(1, 0, "CA", (5.0, 0.0, 0.0), 70.0)]
    past_cutoff = [
        atom(0, 0, "CA", (0.0, 0.0, 0.0), 70.0),
        atom(1, 0, "CA", (5.01, 0.0, 0.0), 70.0),
    ]
    assert derive.interface_contacts(at_cutoff) == 1
    assert derive.interface_contacts(past_cutoff) == 0


def test_no_interface_gives_zero_plddt_not_a_silent_default():
    # One chain only: nothing is within any cutoff of another chain, so the
    # derived interface pLDDT is 0.0 and the harness gate sees a real zero.
    atoms = [atom(0, 0, "CA", (0.0, 0.0, 0.0), 90.0), atom(0, 1, "CA", (3.8, 0.0, 0.0), 90.0)]
    assert derive.interface_plddt(atoms) == 0.0
    assert derive.interface_contacts(atoms) == 0


def test_clash_count_counts_only_inter_chain_pairs_closer_than_the_cutoff():
    """The cutoff is 1.1 A, the distance foundry's MIN_CLASH_DISTANCE sets
    (models/rf3/src/rf3/metrics/clashing_chains.py). The count covers
    inter-chain pairs only; whether foundry's own `has_clash` flag also
    covers intra-chain pairs is unverified, because no real fold has run.

    Pair 1: chains 0/1 at distance 1.0 -> a clash.
    Pair 2: chains 0/1 at distance 1.2 -> not a clash.
    Pair 3: two atoms of the SAME chain at distance 0.5 -> never counted by
    this function, whatever foundry's flag would say about it.
    """
    atoms = [
        atom(0, 0, "CA", (0.0, 0.0, 0.0), 70.0),
        atom(0, 0, "CB", (-0.5, 0.0, 0.0), 70.0),  # 0.5 A from CA, same chain
        atom(1, 0, "CA", (1.0, 0.0, 0.0), 70.0),  # 1.0 A from chain 0 CA
        atom(1, 1, "CA", (50.0, 0.0, 0.0), 70.0),
        atom(0, 1, "CA", (51.2, 0.0, 0.0), 70.0),  # 1.2 A inter-chain
    ]
    # CB at (-0.5,0,0) is 1.5 A from the chain-1 CA, so only the CA-CA pair
    # at 1.0 A counts.
    assert derive.clash_count(atoms) == 1


def test_net_charge_is_exact_from_the_sequence():
    # K +1, K +1, D -1, D -1, E -1, H 0  ->  -1
    assert derive.net_charge("KKDDEH") == -1.0
    assert derive.net_charge("") == 0.0


def test_hydrophobic_fraction_is_a_composition_proxy():
    # Set is {A,V,L,I,M,F,W,Y}. "AAWK": A, A, W are in the set, K is not.
    # 3 / 4 = 0.75
    assert derive.hydrophobic_fraction("AAWK") == 0.75
    assert derive.hydrophobic_fraction("") == 0.0


def test_pae_interface_averages_the_designed_boundary_only():
    """chain_pair_pae, chains 0 and 1 designed, chain 2 not:

        [[None, 2.0, 3.0],
         [None, None, 9.0],
         [None, None, None]]

    Boundary entries (exactly one endpoint designed): [0][2] = 3.0 and
    [1][2] = 9.0. The designed-designed entry 2.0 is not a boundary.
    Mean = (3.0 + 9.0) / 2 = 6.0.
    """
    matrix = ((None, 2.0, 3.0), (None, None, 9.0), (None, None, None))
    assert derive.pae_interface(matrix, designed_indices=[0, 1]) == pytest.approx(6.0)


def test_pae_interface_falls_back_when_there_is_no_boundary():
    # A single chain has no designed/target boundary; fall back to the mean
    # over every entry rather than inventing zero.
    assert derive.pae_interface(((None, 4.0), (None, None)), [0]) == pytest.approx(4.0)
    assert derive.pae_interface((), [0]) is None
