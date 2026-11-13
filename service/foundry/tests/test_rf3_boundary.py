"""The pure side of `rf3.py`: input building, engine arguments, conversion.

`run_rf3` itself needs foundry and a GPU and is not exercised here. It is
the only function in the package that imports foundry, and everything it
calls before and after that import is covered below.
"""

from __future__ import annotations

import pytest

from foundry_service import rf3
from foundry_service.contract import MoleculeKind
from tests.fakes import chain, fake_output, job_spec

LETTERS_AB = rf3.chain_letters(job_spec())


def test_the_input_json_matches_foundrys_own_fixture_shape():
    job = job_spec(binder="QVQL", target="MKTV", name="design-7")
    payload = rf3.build_input_json(job)
    assert payload == {
        "name": "design-7",
        "components": [
            {"seq": "QVQL", "chain_id": "A"},
            {"seq": "MKTV", "chain_id": "B"},
        ],
    }


def test_use_msa_true_fails_with_a_reason_that_names_the_field():
    job = job_spec(use_msa=True)
    with pytest.raises(rf3.UnsupportedRequest, match="use_msa"):
        rf3.build_input_json(job)


def test_a_non_protein_chain_fails_with_a_reason_that_names_the_chain():
    job = job_spec()
    job.chains[1].kind = MoleculeKind.DNA
    with pytest.raises(rf3.UnsupportedRequest, match="chain A"):
        rf3.build_input_json(job)


def test_more_chains_than_single_letters_fails():
    job = job_spec(binder=None, target=None)
    job.chains = [chain("X", "AA", designed=False) for _ in range(27)]
    with pytest.raises(rf3.UnsupportedRequest, match="26"):
        rf3.build_input_json(job)


def test_early_stopping_is_disabled_in_the_invocation_not_assumed():
    """The trap: an early-stopped RF3 run writes no structure at all, only a
    metrics CSV and a score file (models/rf3/src/rf3/inference_engines/rf3.py,
    "Handle early stopping"). The Hydra config path defaults the threshold to
    0.5 (models/rf3/configs/inference_engine/rf3.yaml:20), and a run without
    an MSA is exactly what triggers it. This asserts the engine arguments
    disable it explicitly rather than trusting a constructor default.
    """
    job = job_spec(diffusion_samples=3, diffusion_steps=20, seeds=(42,))
    kwargs = rf3.build_engine_kwargs(job)
    assert "early_stopping_plddt_threshold" in kwargs
    assert kwargs["early_stopping_plddt_threshold"] is None
    assert kwargs["diffusion_batch_size"] == 3
    assert kwargs["num_steps"] == 20
    assert kwargs["seed"] == 42


def test_no_seeds_fails_with_a_reason_that_names_the_field():
    job = job_spec(seeds=())
    with pytest.raises(rf3.UnsupportedRequest, match="seeds"):
        rf3.build_engine_kwargs(job)


def test_more_than_one_seed_fails_rather_than_dropping_the_extras():
    """The harness may send several seeds (src/config.rs takes a list and
    src/fold/mod.rs passes it through). One job is one RF3 invocation with
    one seed, so using seeds[0] alone would drop the rest silently. The job
    must fail with a reason that names the field and the count.
    """
    job = job_spec(seeds=(42, 7))
    with pytest.raises(rf3.UnsupportedRequest, match="seeds") as info:
        rf3.build_engine_kwargs(job)
    message = str(info.value)
    assert "2 seeds" in message
    assert "one seed per job" in message


def test_an_early_stopped_result_fails_the_job_instead_of_losing_the_structure():
    results = {"design-7": {"early_stopped": True, "mean_plddt": 0.4213}}
    with pytest.raises(rf3.EarlyStopped) as info:
        rf3.samples_from_results(results, LETTERS_AB)
    message = str(info.value)
    assert "early-stopped" in message
    assert "no structure" in message
    assert "0.4213" in message


def test_an_empty_result_set_fails_rather_than_passing_nothing_on():
    with pytest.raises(rf3.Rf3Error, match="empty"):
        rf3.samples_from_results({}, LETTERS_AB)


def _two_chain_output(**overrides):
    defaults = dict(
        chain_ids=["A", "A", "B", "B"],
        res_ids=[1, 2, 1, 2],
        res_names=["GLN", "VAL", "MET", "LYS"],
        atom_names=["CA", "CA", "CA", "CA"],
        coords=[(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)],
        atom_plddts=[0.85, 0.75, 0.65, 0.55],
        overall_plddt=0.7,
    )
    defaults.update(overrides)
    return fake_output(**defaults)


def test_conversion_scales_plddt_to_the_harness_range():
    # foundry reports 0..1 (confidence_loss.yaml plddt.max_value: 1.0); the
    # harness Metrics document 0..100.
    converted = rf3.convert_sample(_two_chain_output(), LETTERS_AB, "design-7")
    assert [a.plddt for a in converted.atoms] == pytest.approx([85.0, 75.0, 65.0, 55.0])
    assert converted.overall_plddt == pytest.approx(70.0)
    assert converted.ptm == 0.9
    assert converted.iptm == 0.8
    assert converted.ranking_score == 0.76
    assert converted.chain_pair_pae == ((None, 3.0), (None, None))


def test_residue_indices_come_from_res_ids_not_token_res_ids():
    """The trap: confidences["token_res_ids"] is sequential token indexing
    ("Simplified; could map to actual res_id",
    models/rf3/src/rf3/utils/predicted_error.py:371), not residue numbering.
    Here res_ids are [1, 2] per chain and token_res_ids is a deliberately
    wrong [10, 20, 30, 40]; the conversion must yield res_index 0,1 per
    chain, sorted by res_id.
    """
    output = _two_chain_output(token_res_ids=[10, 20, 30, 40])
    converted = rf3.convert_sample(output, LETTERS_AB, "design-7")
    assert [(a.chain_index, a.res_index) for a in converted.atoms] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_residue_indices_follow_res_id_order_not_atom_order():
    # Same chain, res_ids reported out of order [7, 5]: sorted, 5 -> index 0.
    output = _two_chain_output(
        chain_ids=["A", "A", "B"],
        res_ids=[7, 5, 1],
        res_names=["GLN", "GLN", "MET"],
        atom_names=["CA", "CA", "CA"],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        atom_plddts=[0.8, 0.7, 0.6],
    )
    converted = rf3.convert_sample(output, LETTERS_AB, "design-7")
    assert [(a.chain_index, a.res_index) for a in converted.atoms] == [(0, 1), (0, 0), (1, 0)]


def test_misaligned_atom_plddts_fail_instead_of_guessing():
    output = _two_chain_output(atom_plddts=[0.85, 0.75, 0.65])  # 3 pLDDTs, 4 atoms
    with pytest.raises(rf3.Rf3Error, match="does not line up"):
        rf3.convert_sample(output, LETTERS_AB, "design-7")


def test_a_chain_id_the_job_never_sent_fails():
    output = _two_chain_output(chain_ids=["A", "A", "Z", "B"])
    with pytest.raises(rf3.Rf3Error, match="did not send"):
        rf3.convert_sample(output, LETTERS_AB, "design-7")


def test_missing_atom_plddts_fails_because_the_contract_says_so():
    output = _two_chain_output(atom_plddts=None)  # confidences is None
    with pytest.raises(rf3.Rf3Error, match="atom_plddts"):
        rf3.convert_sample(output, LETTERS_AB, "design-7")


def test_a_missing_ranking_score_is_recomputed_with_foundrys_formula():
    # 0.8 * iptm + 0.2 * ptm - 100 * has_clash, with ptm=0.9, iptm=0.8,
    # has_clash=True: 0.64 + 0.18 - 100 = -99.18
    output = _two_chain_output(has_clash=True, ranking_score=None)
    converted = rf3.convert_sample(output, LETTERS_AB, "design-7")
    assert converted.ranking_score == pytest.approx(-99.18)
    assert converted.has_clash is True


def test_samples_from_results_converts_the_whole_list():
    results = {
        "design-7": [
            _two_chain_output(sample_idx=0),
            _two_chain_output(sample_idx=1, ptm=0.5, iptm=0.4, ranking_score=0.4),
        ]
    }
    samples = rf3.samples_from_results(results, LETTERS_AB)
    assert len(samples) == 2
    assert [s.sample_idx for s in samples] == [0, 1]
    assert all(s.example_id == "design-7" for s in samples)


def test_the_model_name_maps_to_a_registered_checkpoint():
    assert rf3._checkpoint_for("rf3") == "rf3_foundry_01_24_latest_remapped.ckpt"
    with pytest.raises(rf3.UnsupportedRequest, match="model"):
        rf3._checkpoint_for("alphafold3")
