"""The adapter from RF3 samples to the harness's FoldResult."""

from __future__ import annotations

import pytest

from foundry_service import adapter
from foundry_service.contract import StructureFormat
from tests.fakes import atom, job_spec, sample

# Residue names matching "QVQL" and "MKTV", so check_residues passes.
BINDER_NAMES = ["GLN", "VAL", "GLN", "LEU"]
TARGET_NAMES = ["MET", "LYS", "THR", "VAL"]


def _interface_atoms():
    """Binder residue 3 contacts target residue 0 at 4.0 A; nothing else is
    within the 5.0 A cutoff. Interface pLDDT = mean(binder-res-3, target-res-0).
    """
    atoms = []
    for index, name in enumerate(BINDER_NAMES):
        atoms.append(atom(0, index, "CA", (float(index), 0.0, 0.0), 80.0 + index, res_name=name))
    # Target placed 4 A from binder residue 3 (at x=3): target residue 0 at x=7.
    atoms.append(atom(1, 0, "CA", (7.0, 0.0, 0.0), 60.0, res_name=TARGET_NAMES[0]))
    for index in range(1, 4):
        atoms.append(
            atom(1, index, "CA", (30.0 + index, 0.0, 0.0), 90.0, res_name=TARGET_NAMES[index])
        )
    return tuple(atoms)


def test_the_derived_field_list_names_the_two_metrics_the_harness_gates_on():
    # AGENTS.md: the adapter must state which numbers it derives. The loss
    # weights interface_plddt and interface_contacts and the gates threshold
    # both, so the record is incomplete without them.
    assert "interface_plddt" in adapter.ADAPTER_DERIVED_FIELDS
    assert "interface_contacts" in adapter.ADAPTER_DERIVED_FIELDS


def test_pick_best_sample_takes_the_highest_ranking_score():
    weak = sample(ranking_score=0.5, sample_idx=0)
    strong = sample(ranking_score=0.9, sample_idx=1)
    assert adapter.pick_best_sample([weak, strong]) is strong
    with pytest.raises(adapter.AdapterError, match="no samples"):
        adapter.pick_best_sample([])


def test_fold_result_carries_reported_and_derived_numbers_in_their_places():
    job = job_spec(binder="QVQL", target="MKTV")
    atoms = _interface_atoms()
    chosen = sample(
        atoms,
        ptm=0.94,
        iptm=0.88,
        overall_plddt=85.0,
        chain_pair_pae=((None, 3.0), (None, None)),
        ranking_score=0.9,
    )
    result = adapter.fold_result("job-000001", job, [chosen], elapsed_ms=1234)

    assert result.job_id == "job-000001"
    assert result.model == "rf3"
    assert result.elapsed_ms == 1234

    conf = result.confidence
    # Model-reported, passed through.
    assert conf.iptm == 0.88
    assert conf.ptm == 0.94
    assert conf.plddt_mean == 85.0
    assert conf.atom_confidence is True
    # Derived: the adapter's mean over the chain-pair PAE matrix. Here the
    # designed/target boundary holds a single entry, 3.0.
    assert conf.pae_interface == 3.0
    # Derived, hand arithmetic. Target residue 0 sits at x=7. Binder residue
    # 3 (x=3) is 4.0 A away and binder residue 2 (x=2) is exactly 5.0 A
    # away, and the cutoff is inclusive, so both pairs are contacts:
    #   interface contacts = 2
    #   interface residues = binder res 2 (pLDDT 82), binder res 3 (83),
    #                        target res 0 (60)
    #   interface pLDDT = (82 + 83 + 60) / 3 = 225 / 3 = 75.0
    assert conf.interface_plddt == pytest.approx(75.0)
    assert conf.interface_contacts == 2
    assert conf.clashes == 0
    # Derived from the designed sequence "QVQL". The hydrophobic set is
    # {A,V,L,I,M,F,W,Y}: V and L are in it, Q is not, so 2 / 4 = 0.5.
    # Net charge of QVQL: no K, R, D or E, so 0.
    assert conf.hydrophobic_patch == 0.5
    assert conf.net_charge == 0.0
    # Per-residue pLDDT for the designed chain only, one per sent residue.
    assert conf.plddt == [80.0, 81.0, 82.0, 83.0]

    assert result.structure.format == StructureFormat.CA_TRACE
    assert len(result.structure.ca_trace) == 8


def test_a_multi_chain_job_without_iptm_fails_rather_than_substituting():
    job = job_spec(binder="QVQL", target="MKTV")
    blind = sample(_interface_atoms(), iptm=None)
    with pytest.raises(adapter.AdapterError, match="ipTM"):
        adapter.confidence_for(job, blind)


def test_need_atom_confidence_with_no_atoms_fails_the_job():
    job = job_spec(need_atom_confidence=True)
    empty = sample((), iptm=0.8)
    with pytest.raises(adapter.AdapterError, match="need_atom_confidence"):
        adapter.confidence_for(job, empty)


def test_a_residue_name_disagreement_fails_and_names_the_chain_and_position():
    job = job_spec(binder="QVQL", target="MKTV")
    atoms = list(_interface_atoms())
    atoms[1] = atom(0, 1, "CA", (1.0, 0.0, 0.0), 81.0, res_name="ALA")  # sent V
    with pytest.raises(adapter.AdapterError, match=r"chain H position 2") as info:
        adapter.check_residues(job, sample(tuple(atoms)))
    assert "sent V" in str(info.value)
    assert "ALA" in str(info.value)


def test_extra_returned_residues_fail_the_job():
    job = job_spec(binder="QVQL", target="MKTV")
    atoms = list(_interface_atoms())
    atoms.append(atom(0, 4, "CA", (9.0, 9.0, 9.0), 70.0, res_name="GLY"))
    with pytest.raises(adapter.AdapterError, match="extra residue"):
        adapter.check_residues(job, sample(tuple(atoms)))


def test_a_designed_chain_with_no_atoms_fails_but_a_missing_target_does_not():
    job = job_spec(binder="QVQL", target="MKTV")
    target_only = tuple(
        atom(1, index, "CA", (float(index), 0.0, 0.0), 70.0, res_name=TARGET_NAMES[index])
        for index in range(4)
    )
    with pytest.raises(adapter.AdapterError, match="designed chain"):
        adapter.check_residues(job, sample(target_only))

    binder_only = tuple(
        atom(0, index, "CA", (float(index), 0.0, 0.0), 70.0, res_name=BINDER_NAMES[index])
        for index in range(4)
    )
    adapter.check_residues(job, sample(binder_only))  # no raise
