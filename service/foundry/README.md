# foundry compute service

A reference compute service for folding-harness. It speaks the `fh.fold/v1`
contract from `src/contract.rs` and runs the RF3 structure-prediction model
from [RosettaCommons/foundry][foundry] behind it. This is the sanctioned
`service/` exception in `AGENTS.md`: Python and the GPU stay here, and the
Rust crate in `src/` never imports from this directory.

## Backend record

- Backend: `RosettaCommons/foundry`, model RF3 (RosettaFold3).
- Pinned commit: `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284` (`production`
  branch, rc-foundry 1.0.0). foundry documents its inference API as not yet
  stable, so the service pins one commit. The same pin is recorded in
  `foundry_service/rf3.py` (`FOUNDRY_PIN`) and in `pyproject.toml`.
- Weights licence: foundry's [`LICENSE.md`][license] is BSD-3-Clause and
  states: "This copyright and license covers both the source code and model
  weights referenced for download in the README file." The weights download
  by plain HTTPS from `files.ipd.uw.edu`, with no click-through and no
  account.
- Runtime: Python 3.12 or newer, torch 2.11 or newer, CUDA 12, Linux.

## Adapter-derived numbers

RF3 reports pTM, ipTM, per-atom pLDDT, chain-pair PAE and PDE, and a 0/1
`has_clash` flag. It does not report everything the harness reads. These
five fields of `ConfidenceReport` are computed by this service, not by the
model, and the code that computes them is `foundry_service/derive.py`:

- `interface_plddt`: mean per-residue pLDDT over residues with an atom
  within 5 A of another chain.
- `interface_contacts`: count of inter-chain residue pairs within that 5 A
  cutoff.
- `clashes`: count of inter-chain heavy-atom pairs closer than 1.1 A. The
  1.1 A distance is foundry's own `MIN_CLASH_DISTANCE`, but the count covers
  inter-chain pairs only. RF3's `has_clash` is a per-sample flag of 0 or 1,
  not a count, and whether it also covers clashes within one chain is
  unverified, because no real fold has been run against a checkpoint. The
  derived count may therefore sit below what foundry itself would flag, and
  a structure RF3 flags as clashing can report `clashes: 0` and pass a
  `max_clashes: 2` gate. Until one real run settles the question, treat the
  gate as weaker than its name suggests.
- `hydrophobic_patch`: fraction of hydrophobic residues in the designed
  sequence. This is a composition proxy. The service computes no solvent
  accessibility.
- `pae_interface`: the adapter's mean over the chain-pair PAE entries across
  the designed/target boundary. When no boundary entry exists, it falls back
  to the mean over every entry of the matrix, and when the matrix is empty
  it falls back to RF3's `overall_pae`. Both fallbacks are whole-complex
  numbers reported under an interface-specific field name.

`net_charge` is exact from the sequence, and `ptm`, `iptm` and `plddt_mean`
come from RF3's own output. RF3 reports pLDDT in 0..1, and this service
rescales to the harness's 0..100 at the boundary in `rf3.py`.

Everything a real backend produces is a computational prediction and needs
experimental validation.

## Install

```bash
cd service/foundry
pip install -e ".[rf3]"     # installs foundry at the pinned commit
foundry install base-models # downloads weights to ~/.foundry/checkpoints
```

The `[rf3]` extra pulls torch and the CUDA stack and only works on Linux.
The tests below need none of it.

## Run

```bash
python -m foundry_service --host 127.0.0.1 --port 8080
```

The default bind is loopback. The service authenticates nobody. A
non-loopback `--host` puts an unauthenticated compute endpoint on that
interface, so a remote deployment needs protection of its own: a tunnel, a
firewall, or a reverse proxy. The precedent is `src/fold/server.rs`, which
states the same trade-off for the Rust stub service.

Point the harness at it with `execution_mode: service` and the base URL in
`task.yaml`.

## Framework choice

Standard library only: `http.server.ThreadingHTTPServer`, `threading`,
`queue`, `json`. The contract is three JSON routes, the polling side needs
a thread per connection, and the compute side already runs on its own
worker thread because one GPU runs one fold at a time. FastAPI and uvicorn
would add two dependencies and an ASGI runtime to serve what `BaseHTTPRequestHandler`
serves in about 200 lines, and this repo values few dependencies. The
service therefore adds zero Python dependencies of its own on top of
foundry's.

## Tests

```bash
cd service/foundry
uv run --no-project --python 3.13 --with pytest python -m pytest
```

The suite runs without a GPU, without foundry and without Linux. It pins:

- the wire models against `src/contract.rs`, parsed at test time, so a field
  added or renamed on either side fails (`tests/test_contract_conformance.py`);
- the derived metrics against hand-computed arithmetic
  (`tests/test_derive.py`);
- the ca_trace against the properties `verify_output` in `src/fold/mod.rs`
  checks: one entry per residue sent, indices exactly 0..n per chain,
  residues equal to the sequence sent (`tests/test_trace.py`);
- early stopping disabled in the engine arguments, asserted rather than
  assumed, and an early-stopped result failing the job with a reason
  (`tests/test_rf3_boundary.py`);
- the three routes over a real socket on port 0, with a hand-written fake
  runner (`tests/test_server.py`).

What the suite cannot verify: `rf3.run_rf3`, the only function that imports
foundry. It was written against the pinned commit's source and never
executed against a checkpoint. Its unverified assumptions are listed in the
`rf3.py` module docstring. Confirm them with one real fold on a GPU machine.

[foundry]: https://github.com/RosettaCommons/foundry
[license]: https://github.com/RosettaCommons/foundry/blob/0d0b0ba44fab7ca76aa4b625c014ce5964c7d284/LICENSE.md
