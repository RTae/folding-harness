"""The RF3 boundary: the only module that imports foundry.

Backend adapter record, required by `AGENTS.md`
------------------------------------------------

Backend: RosettaCommons/foundry, model RF3 (RosettaFold3).

Weights licence. foundry's `LICENSE.md` on the `production` branch is
BSD-3-Clause and states, in terms: "This copyright and license covers both
the source code and model weights referenced for download in the README
file." The weights download by plain HTTPS from `files.ipd.uw.edu` (see
`models/rf3/src/rf3/checkpoints.py`), with no click-through and no account.
Licence text at the pinned commit:
https://github.com/RosettaCommons/foundry/blob/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284/LICENSE.md

Pinned commit. foundry documents its inference API as not yet stable, so
this service pins one commit and installs from it:

    FOUNDRY_PIN = 0d0b0ba44fab7ca76aa4b625c014ce5964c7d284
    (production branch, https://github.com/RosettaCommons/foundry, rc-foundry
    version 1.0.0)

Every file:line citation in this service's comments refers to the tree at
that commit. Install with:

    pip install "rc-foundry[all] @ git+https://github.com/RosettaCommons/foundry.git@0d0b0ba44fab7ca76aa4b625c014ce5964c7d284"
    foundry install base-models

Runtime: Python >= 3.12, torch >= 2.11, CUDA 12, Linux. (foundry's README
describes a community macOS/MPS fork; this service targets the upstream
Linux+CUDA install.)

Adapter-derived metrics. RF3 does not report `interface_plddt`,
`interface_contacts`, `hydrophobic_patch`, a clash count (its `has_clash` is
a per-sample flag of 0 or 1), or a single interface PAE (it reports the
chain-pair PAE matrix). This service derives all five: the clash count
covers inter-chain pairs only, and `pae_interface` is the adapter's mean
over the PAE matrix with documented fallbacks. See
`adapter.ADAPTER_DERIVED_FIELDS` and the docstring of `derive.py`. Anyone
reading those numbers is reading adapter computations over RF3's per-atom
pLDDT, predicted coordinates and PAE matrix, not model-reported metrics.

What is verified and what is not
--------------------------------

`build_input_json`, `build_engine_kwargs`, `convert_sample` and
`samples_from_results` are pure and covered by tests that run without a GPU.
`run_rf3` is the only function that imports foundry and touches the model.
It has never been executed against a real checkpoint: this service was
written on a machine with no GPU and no Linux. The conversion assumes
`confidences["atom_plddts"]` aligns atom-for-atom with the predicted
`atom_array` in the reported order, and that the `chain_pair_*` matrices are
indexed in input chain order. `convert_sample` fails the job with a named
reason when the lengths disagree; the matrix ordering assumption needs one
real run to confirm.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from . import adapter
from .contract import FoldResult, JobSpec, MoleculeKind
from .rf3_output import Atom, Rf3Sample

FOUNDRY_PIN = "0d0b0ba44fab7ca76aa4b625c014ce5964c7d284"

# RF3 reports pLDDT in 0..1 (confidence_loss.yaml sets plddt.max_value: 1.0,
# and the early-stopping threshold default of 0.5 is on the same scale). The
# harness's Metrics documents pLDDT as 0..100, so the boundary rescales.
PLDDT_SCALE = 100.0


class UnsupportedRequest(Exception):
    """The job asks for something this service does not do. Names the field."""


class EarlyStopped(Exception):
    """RF3 early-stopped and wrote no structure. This is a job failure, never
    a missing-structure candidate."""


class Rf3Error(Exception):
    """foundry ran but returned something this adapter cannot map honestly."""


@dataclass(frozen=True)
class ChainLetter:
    """Maps one harness chain to the single-letter chain id sent to RF3."""

    index: int
    letter: str


def chain_letters(job: JobSpec) -> list[ChainLetter]:
    """Assign chain ids A, B, C... to `job.chains` in order.

    The harness sends labels like "H" and "A", which may repeat or exceed one
    character, so the RF3 input uses positional letters and every mapping
    back goes through this list.
    """
    if len(job.chains) > 26:
        raise UnsupportedRequest(
            f"job.chains holds {len(job.chains)} chains; this service maps chains to "
            "single-letter PDB chain ids and supports at most 26"
        )
    return [
        ChainLetter(index=index, letter=chr(ord("A") + index)) for index in range(len(job.chains))
    ]


def build_input_json(job: JobSpec) -> dict:
    """The foundry JSON input for one job (`InferenceInput.from_json_dict`).

    The shape matches foundry's own test fixture
    `models/rf3/tests/data/two_protein_chains.json` at the pinned commit:
    {"name": ..., "components": [{"seq": ..., "chain_id": ...}, ...]}.
    """
    if job.use_msa:
        raise UnsupportedRequest(
            "job.use_msa is true, but this service runs RF3 without MSA and does no "
            "MSA search; set use_msa to false or precompute MSAs and point foundry at "
            "them via LOCAL_MSA_DIRS"
        )
    letters = chain_letters(job)
    components = []
    for entry, chain in zip(letters, job.chains):
        if chain.kind != MoleculeKind.PROTEIN:
            raise UnsupportedRequest(
                f"chain {chain.label} has kind {chain.kind.value}; this service folds "
                "proteins only, so set kind to protein or use a service that supports "
                "the molecule"
            )
        if not chain.sequence:
            raise UnsupportedRequest(f"chain {chain.label} has an empty sequence")
        components.append({"seq": chain.sequence, "chain_id": entry.letter})
    return {"name": job.name or "fh-job", "components": components}


def build_engine_kwargs(job: JobSpec) -> dict:
    """Keyword arguments for foundry's `RF3InferenceEngine`.

    One job is one RF3 invocation with one seed. An empty `job.seeds` or a
    list longer than one fails the job with a reason that names the field,
    the same way `build_input_json` handles the other unsupported inputs, so
    no seed is ever dropped silently.

    Early stopping is disabled explicitly. The Python constructor defaults
    `early_stopping_plddt_threshold` to None at the pinned commit
    (models/rf3/src/rf3/inference_engines/rf3.py:262), but the Hydra config
    path defaults it to 0.5 (models/rf3/configs/inference_engine/rf3.yaml:20),
    and a triggered early stop writes no structure at all. Passing None pins
    the behaviour this service needs regardless of which default a future
    foundry release moves. `samples_from_results` still fails the job with a
    reason if an early-stopped result shows up anyway.
    """
    if not job.seeds:
        raise UnsupportedRequest("job.seeds is empty; send at least one seed")
    if len(job.seeds) > 1:
        raise UnsupportedRequest(
            f"job.seeds holds {len(job.seeds)} seeds; one job is one RF3 invocation with "
            "one seed, and this service does not fan out, so seeds after the first "
            "would be dropped silently. Send exactly one seed per job, or split the "
            "work into one job per seed."
        )
    return {
        "n_recycles": 10,
        "diffusion_batch_size": max(1, job.diffusion_samples),
        "num_steps": max(1, job.diffusion_steps),
        "seed": job.seeds[0],
        # Disabled on purpose; see the docstring.
        "early_stopping_plddt_threshold": None,
        "compress_outputs": False,
    }


def convert_sample(output, letters: Sequence[ChainLetter], example_id: str) -> Rf3Sample:
    """Convert one foundry `RF3Output` into the internal `Rf3Sample`.

    `output` is duck-typed so tests can drive this with a hand-written fake:
    it needs `.summary_confidences`, `.confidences`, `.atom_array` (with
    `chain_id`, `res_id`, `res_name`, `atom_name`, `coord` annotations),
    `.sample_idx` and `.seed`.

    Residue indices are derived here, from the `res_id` values foundry
    reports per chain, sorted ascending and renumbered from zero. The
    `token_res_ids` field in `confidences` is deliberately ignored: the
    foundry source builds it as `list(range(len(chain_iid_token_lvl)))` with
    the comment "Simplified; could map to actual res_id"
    (models/rf3/src/rf3/utils/predicted_error.py:371). It is sequential
    token indexing, not residue numbering, and trusting it would misplace
    every residue of every chain after the first.
    """
    confidences = output.confidences or {}
    atom_plddts = confidences.get("atom_plddts")
    if atom_plddts is None:
        raise Rf3Error(
            "RF3 returned no atom_plddts; the harness needs per-atom confidence and "
            "the contract requires failing the job rather than omitting it"
        )

    atom_array = output.atom_array
    chain_ids = list(atom_array.chain_id)
    res_ids = list(atom_array.res_id)
    res_names = list(atom_array.res_name)
    atom_names = list(atom_array.atom_name)
    coords = list(atom_array.coord)

    if not (len(atom_plddts) == len(chain_ids) == len(coords)):
        raise Rf3Error(
            f"RF3 output does not line up: {len(atom_plddts)} atom_plddts for "
            f"{len(chain_ids)} atoms; refusing to guess the alignment"
        )

    letter_to_index = {entry.letter: entry.index for entry in letters}

    # res_id -> zero-based position within the chain, from the chain's sorted
    # res_ids, so the mapping follows residue order rather than first appearance.
    res_ids_by_chain: dict[str, set] = {}
    for chain_id, res_id in zip(chain_ids, res_ids):
        res_ids_by_chain.setdefault(chain_id, set()).add(res_id)
    res_index_by_chain = {
        chain_id: {res_id: position for position, res_id in enumerate(sorted(seen))}
        for chain_id, seen in res_ids_by_chain.items()
    }

    atoms: list[Atom] = []
    for chain_id, res_id, res_name, atom_name, xyz, plddt in zip(
        chain_ids, res_ids, res_names, atom_names, coords, atom_plddts
    ):
        chain_index = letter_to_index.get(chain_id)
        if chain_index is None:
            raise Rf3Error(
                f"RF3 returned chain id {chain_id!r}, which this job did not send "
                f"(sent {[entry.letter for entry in letters]})"
            )
        x, y, z = (float(v) for v in xyz)
        atoms.append(
            Atom(
                chain_index=chain_index,
                res_index=res_index_by_chain[chain_id][res_id],
                res_name=str(res_name),
                atom_name=str(atom_name),
                xyz=(x, y, z),
                plddt=float(plddt) * PLDDT_SCALE,
            )
        )

    summary = output.summary_confidences or {}
    ptm = summary.get("ptm")
    if ptm is None:
        raise Rf3Error("RF3 reported no ptm for this sample")
    iptm = summary.get("iptm")
    has_clash = bool(summary.get("has_clash", False))
    ranking = summary.get("ranking_score")
    if ranking is None:
        # foundry's formula, models/rf3/src/rf3/inference_engines/rf3.py.
        ranking = 0.8 * (ptm if iptm is None else iptm) + 0.2 * ptm - 100 * int(has_clash)
    overall_plddt = summary.get("overall_plddt")
    if overall_plddt is None:
        raise Rf3Error("RF3 reported no overall_plddt for this sample")

    chain_pair_pae = tuple(tuple(row) for row in (summary.get("chain_pair_pae") or []))

    return Rf3Sample(
        example_id=example_id,
        sample_idx=int(getattr(output, "sample_idx", 0)),
        seed=int(getattr(output, "seed", 0)),
        ranking_score=float(ranking),
        ptm=float(ptm),
        iptm=None if iptm is None else float(iptm),
        has_clash=has_clash,
        overall_plddt=float(overall_plddt) * PLDDT_SCALE,
        overall_pae=float(summary.get("overall_pae") or 0.0),
        chain_pair_pae=chain_pair_pae,
        atoms=tuple(atoms),
    )


def samples_from_results(results: dict, letters: Sequence[ChainLetter]) -> list[Rf3Sample]:
    """Convert foundry's in-memory `run()` result dict into samples.

    foundry maps example_id -> list of RF3Output, except when early stopping
    fired, in which case the entry is {"early_stopped": True, "mean_plddt":
    ...} and no structure exists (models/rf3/src/rf3/inference_engines/
    rf3.py, "Handle early stopping"). An early-stopped job fails here with a
    reason that says what happened. It never reaches the harness as a
    missing structure.
    """
    if not results:
        raise Rf3Error("RF3 returned an empty result set")

    samples: list[Rf3Sample] = []
    for example_id, entry in results.items():
        if isinstance(entry, dict) and entry.get("early_stopped"):
            mean_plddt = entry.get("mean_plddt")
            raise EarlyStopped(
                f"RF3 early-stopped on {example_id}: mean pLDDT "
                f"{mean_plddt if mean_plddt is None else round(float(mean_plddt), 4)} "
                "is below the early_stopping_plddt_threshold and no structure was "
                "written. This service passes early_stopping_plddt_threshold=None to "
                "disable it; if this fired anyway, the deployed foundry differs from "
                f"the pinned commit {FOUNDRY_PIN}."
            )
        if not isinstance(entry, (list, tuple)):
            raise Rf3Error(
                f"RF3 result for {example_id} is a {type(entry).__name__}, expected a "
                "list of RF3Output objects"
            )
        for output in entry:
            samples.append(convert_sample(output, letters, str(example_id)))
    return samples


def run_rf3(job: JobSpec) -> list[Rf3Sample]:
    """Run foundry's RF3 on one job and return its samples.

    THE ONLY FUNCTION THAT NEEDS A GPU. Everything else in this package is
    pure or standard-library. Not executed against a real checkpoint at the
    time of writing; see the module docstring for the unverified assumptions.
    """
    from rf3.inference_engines.rf3 import RF3InferenceEngine  # noqa: PLC0415
    from rf3.utils.inference import InferenceInput  # noqa: PLC0415

    letters = chain_letters(job)
    input_json = build_input_json(job)
    engine_kwargs = build_engine_kwargs(job)

    checkpoint = _checkpoint_for(job.model)
    engine = RF3InferenceEngine(ckpt_path=checkpoint, **engine_kwargs)
    inference_input = InferenceInput.from_json_dict(input_json)
    results = engine.run(inputs=inference_input, out_dir=None)
    if results is None:
        raise Rf3Error("foundry returned None from run(); expected in-memory results")
    return samples_from_results(results, letters)


def _checkpoint_for(model: str) -> str:
    """Resolve the checkpoint path for the requested model name.

    foundry searches ~/.foundry/checkpoints plus $FOUNDRY_CHECKPOINT_DIRS on
    its own, so the model name maps to the checkpoint file name foundry
    registers (models/rf3/src/rf3/checkpoints.py at the pinned commit).
    """
    if model in ("rf3", "rf3_latest"):
        return "rf3_foundry_01_24_latest_remapped.ckpt"
    if model == "rf3_preprint":
        return "rf3_foundry_09_21_preprint_remapped.ckpt"
    raise UnsupportedRequest(
        f"job.model is {model!r}; this service runs RF3, so set model to 'rf3', "
        "'rf3_latest' or 'rf3_preprint'"
    )


def rf3_runner(job: JobSpec) -> FoldResult:
    """The production job runner: run RF3, adapt, time.

    `jobs.JobTable` calls this on its worker thread. The job_id placeholder
    is overwritten by the table, the same way `src/fold/server.rs` does.
    """
    started = time.monotonic()
    samples = run_rf3(job)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return adapter.fold_result(job_id="", job=job, samples=samples, elapsed_ms=elapsed_ms)
