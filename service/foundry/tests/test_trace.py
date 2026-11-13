"""The ca_trace must satisfy `verify_output` in `src/fold/mod.rs`.

The Rust verifier is not callable from Python, and `AGENTS.md` forbids a
Rust test that requires this service, so this test asserts exactly the
properties `verify_output` checks, against the wire JSON the Rust client
would actually parse:

1. one entry per residue sent, per chain (src/fold/mod.rs:222-231),
2. each chain's indices, sorted, are exactly 0..n (src/fold/mod.rs:237-246),
3. the residue letters equal the sequence sent, in order (src/fold/mod.rs:248-259),
4. chain_index stays inside the chains the job sent (src/fold/mod.rs:173-182).
"""

from __future__ import annotations

import json

from foundry_service.contract import StructureFormat, StructurePayload
from foundry_service.trace import build_ca_trace
from tests.fakes import atom, job_spec, sample


def _sample_with_a_gap():
    """Binder "QVQL" fully resolved, target "MKTV" missing residue 1 (K).

    The target's resolved residues carry a CA atom; residue 0 also carries
    N and C atoms listed BEFORE the CA, to pin that the builder picks the CA
    coordinate rather than the first atom it sees.
    """
    atoms = []
    for index in range(4):
        atoms.append(atom(0, index, "CA", (float(index), 0.0, 0.0), 80.0 + index, res_name="GLY"))
    atoms += [
        atom(1, 0, "N", (10.0, 0.0, 0.0), 70.0, res_name="MET"),
        atom(1, 0, "C", (10.5, 0.0, 0.0), 90.0, res_name="MET"),
        atom(1, 0, "CA", (10.2, 0.0, 0.0), 80.0, res_name="MET"),
        # residue 1 (K) absent: the model resolved nothing for it
        atom(1, 2, "CA", (11.0, 0.0, 0.0), 60.0, res_name="THR"),
        atom(1, 3, "CA", (12.0, 0.0, 0.0), 50.0, res_name="VAL"),
    ]
    return sample(tuple(atoms))


def test_the_trace_carries_one_entry_per_residue_sent_even_with_a_gap():
    job = job_spec(binder="QVQL", target="MKTV")
    trace = build_ca_trace(job.chains, _sample_with_a_gap())

    # Property 1: 4 + 4 entries, none dropped for the unresolved residue.
    assert len(trace) == 8
    per_chain = {0: [], 1: []}
    for entry in trace:
        per_chain[entry.chain_index].append(entry)
    assert len(per_chain[0]) == 4
    assert len(per_chain[1]) == 4

    # Property 2: indices exactly 0..n per chain, in sent order.
    assert [entry.index for entry in per_chain[0]] == [0, 1, 2, 3]
    assert [entry.index for entry in per_chain[1]] == [0, 1, 2, 3]

    # Property 3: letters equal the sent sequences.
    assert "".join(entry.residue for entry in per_chain[0]) == "QVQL"
    assert "".join(entry.residue for entry in per_chain[1]) == "MKTV"

    # Property 4: chain_index only references sent chains.
    assert {entry.chain_index for entry in trace} == {0, 1}


def test_an_unresolved_residue_gets_zero_coordinates_and_zero_confidence():
    job = job_spec(binder="QVQL", target="MKTV")
    trace = build_ca_trace(job.chains, _sample_with_a_gap())
    unresolved = [entry for entry in trace if entry.chain_index == 1 and entry.index == 1][0]
    assert unresolved.xyz == (0.0, 0.0, 0.0)
    assert unresolved.plddt == 0.0
    assert unresolved.residue == "K"  # the letter comes from the sent sequence


def test_resolved_entries_carry_the_ca_coordinate_and_mean_atom_plddt():
    job = job_spec(binder="QVQL", target="MKTV")
    trace = build_ca_trace(job.chains, _sample_with_a_gap())
    by_key = {(entry.chain_index, entry.index): entry for entry in trace}
    # Target residue 0 has atoms N=70, C=90, CA=80; mean = 240/3 = 80.
    # The xyz is the CA atom's, not the first atom's.
    assert by_key[(1, 0)].plddt == 80.0
    assert by_key[(1, 0)].xyz == (10.2, 0.0, 0.0)
    # Binder residue 2 is a lone CA at pLDDT 82.
    assert by_key[(0, 2)].plddt == 82.0
    assert by_key[(0, 2)].xyz == (2.0, 0.0, 0.0)


def test_designed_flags_follow_the_job_not_the_model():
    job = job_spec(binder="QVQL", target="MKTV")
    trace = build_ca_trace(job.chains, _sample_with_a_gap())
    assert all(entry.designed for entry in trace if entry.chain_index == 0)
    assert not any(entry.designed for entry in trace if entry.chain_index == 1)


def test_the_serialised_trace_holds_the_properties_the_rust_verifier_checks():
    """Round-trip through wire JSON: the exact bytes the Rust client parses."""
    job = job_spec(binder="QVQL", target="MKTV")
    trace = build_ca_trace(job.chains, _sample_with_a_gap())
    payload = StructurePayload(format=StructureFormat.CA_TRACE, ca_trace=trace)
    reloaded = json.loads(json.dumps(payload.to_wire()))["ca_trace"]

    sent = {0: "QVQL", 1: "MKTV"}
    for chain_index, sequence in sent.items():
        entries = sorted(
            (entry for entry in reloaded if entry["chain_index"] == chain_index),
            key=lambda entry: entry["index"],
        )
        assert [entry["index"] for entry in entries] == list(range(len(sequence)))
        assert "".join(entry["residue"] for entry in entries) == sequence
        for entry in entries:
            assert set(entry) == {
                "chain_index",
                "index",
                "residue",
                "xyz",
                "plddt",
                "designed",
            }
            assert len(entry["xyz"]) == 3
