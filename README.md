# folding-harness

A design harness in Rust: the orchestration, typing and inspection around a
structure-prediction backend, with the backend behind a versioned HTTP contract.

It ships a deterministic fixture backend, so the whole loop runs on a laptop
with no GPU and no network, and a terminal dashboard, so a long run can be
watched over SSH without forwarding a port.

> **Status: scaffold.** The cycle, the contract, the gates, the population, the
> trace and the dashboard are implemented and tested. The model-backed skills
> (inverse folding, likelihood-guided mutation) are declared in the contract and
> gated by the router, but have no executor yet — `fh skills` says which. The
> fixture backend predicts nothing: it scores sequences against an optimum
> planted from the run seed, so that orchestration can be tested without a
> model. Anything a real run produces is a computational prediction and needs
> experimental validation.

## Why Rust is a reasonable choice here

The harness never touches a tensor. It submits a job, polls for a result, and
reads back confidence numbers and a structure. Everything on this side of that
boundary — orchestration, config validation, gates, population management,
provenance, the UI — is ordinary systems code, and three things get better in a
typed language:

- **The immutable-scope guarantee.** A sequence cannot be edited directly.
  Changes go through `MutableScope::propose`, which validates every substitution
  against the scope and is the only constructor of a `Proposal`; `Proposal` has
  no `Deserialize` impl, so one cannot arrive from a config file or an API
  payload either. "Framework residues are never mutated" is a property of the
  types, not a convention a skill has to remember. Tooling in this field
  disagrees about residue numbering — 1-based per chain, PDB-native with
  insertion codes, contig strings — and off-by-one errors there are silent. This
  harness converts once, at the config boundary, and the conversion has a test.
- **One process, one language, no second runtime.** The dashboard is the same
  binary as the runner; there is no Node or Python process beside it.
- **Cheap long-running processes.** A design run is mostly waiting on a GPU
  service. `tokio` makes bounded-concurrency folds and a tailable event log
  straightforward, and the dashboard reads files rather than holding a socket, so
  it cannot disturb a run.

The cost is real: structure parsing, metrics and the models themselves live in
Python's ecosystem. That is why they stay behind the contract. `pdbtbx` can read
mmCIF in Rust if a future adapter needs it.

## Install and run

Needs a recent stable Rust toolchain.

```bash
cargo build --release
cargo test --workspace

# write an example task, check it, run it
./target/release/fh init task.yaml
./target/release/fh validate task.yaml
./target/release/fh run task.yaml

# watch it in the terminal instead of streaming events
./target/release/fh run task.yaml --board --latency-ms 250

# open the dashboard on the most recent run
./target/release/fh board
```

To exercise the real network path, run the stub service and point a task at it:

```bash
./target/release/fh serve --task task.yaml --latency-ms 200
./target/release/fh run task.yaml --url http://127.0.0.1:8080
```

The only difference between the two modes is whether the bytes crossed a socket.

## The cycle

```
            ┌───────────────────────────────────────────────┐
            │                                               │
  founder ──┴─> population ──> parent selection             │
                   ▲                 │                      │
                   │                 ▼                      │
                   │           router: which skills          │
                   │           are legal right now           │
                   │                 │                      │
                   │                 ▼                      │
                   │           agent: which one              │
                   │                 │                      │
                   │                 ▼                      │
                   │           skill ──> MutableScope::propose
                   │                       │  (validates or refuses)
                   │                       ▼
                   │                   candidates
                   │                       │
                   │                       ▼
                   │                 fold (HTTP or fixture)
                   │                       │
                   │                       ▼
                   │                 loss + hard gates
                   │                       │
                   └───────── admission ◄──┘
                      (dedup, diversity, capacity)
```

Two separations do the work. **Legality is not choice**: the router decides what
may be used, from the user's enabled list, the capabilities the service
advertises, whether an executor exists, and the search state; the agent only
picks from that set, so it cannot reach a skill the service cannot serve. **Gates
are not the loss**: the loss ranks candidates that are acceptable, gates decide
what is acceptable at all, and a gate failure is recorded against the candidate
rather than weighted away.

## What a run leaves behind

```
~/.folding_harness/tasks/<task-id>/
  task.yaml          resolved config, as used
  run.json           manifest: seed, backend, model, protocol, config SHA-256
  trace.jsonl        every event, in order, flushed per line
  population.json    current members, loss curve, per-cycle records
  best.json          best candidate
  structures/        alpha-carbon trace per candidate
```

`run.json` is the part most pipelines leave implicit. With the seed, the config
digest and the service version, a result can be re-run and shown to have used the
same inputs. Set `FH_TASK_ROOT` to move the directory.

## The dashboard

`fh board` has five panes: **Overview** (loss curve, per-cycle table, rejection
counts, the last routing decision and why each skill was refused), **Population**
(ranked table, loss broken down by term, the sequence with fixed residues grey,
mutable green and changed yellow), **Lineage** (parent/child tree of retained
candidates), **Structure** (the backbone drawn with braille glyphs, coloured by
confidence band, rotatable with `h/l/u/d`, zoom `+/-`), and **Events** (the trace).

Keys: `q` quit, `1`–`5`/`Tab` panes, `j/k` select, `f` follow the best candidate.

## The compute contract

`fh-contract` is the whole protocol and depends only on `serde`, so a service can
implement it without the rest of the harness:

```
GET  /v1/health          -> Health   { protocol, capabilities, gpus, queue_depth }
POST /v1/fold            -> JobAccepted { job_id }      (returns immediately)
GET  /v1/fold/{job_id}   -> JobStatus   { state, result }
```

Submission is accept-then-poll, not a long blocking request: a fold outlives the
client, a crashed client can reattach to a job id, and the service can report a
queue. `fh-fold::server` is a working reference implementation.

Capabilities drive the router: a service that advertises only `fold` will never
be offered an inverse-folding skill.

## Layout

A Cargo workspace, one crate per layer, dependencies pointing one way:

| crate | holds | depends on |
| --- | --- | --- |
| `fh-core` | scaffold, mutable scope, proposals, candidates, metrics, gates, population, search state | — |
| `fh-contract` | the `fh.fold/v1` wire types | — |
| `fh-fold` | `FoldBackend` trait, fixture, HTTP client, stub server | core, contract |
| `fh-skills` | design skills and the router | core |
| `fh-agent` | the decision layer: heuristic agent, model-provider seam | core |
| `fh-trace` | event log, task directory layout, run manifest | core, contract |
| `fh-orchestrator` | the cycle | all of the above |
| `fh-tui` | the dashboard | core, contract, trace |
| `fh-cli` | the `fh` command | all of the above |

The split is a choice, not a Rust requirement — a single crate with modules
would work. It buys enforced dependency direction (`fh-core` cannot accidentally
depend on the UI), a contract crate publishable on its own, the stub server
behind a feature flag so a client build does not compile `axum`, and parallel
rebuilds.

## Roadmap

Nearest first:

1. **A real backend adapter.** Boltz-2 and Chai-1 ship code *and* weights under
   MIT/Apache-2.0, which makes them the practical first targets; both are CLI-only
   today, so the adapter is a service wrapping a CLI. AlphaFold 3's code is
   Apache-2.0 but its weights are non-commercial, must not be redistributed, and
   may not be used to train competing predictors — read
   `WEIGHTS_TERMS_OF_USE.md` before wiring it up.
2. **Output re-verification.** Check the returned structure's sequence against
   the proposal, so a backend that silently renumbers or substitutes is caught
   rather than scored.
3. **Inverse-folding and likelihood-guided executors**, behind the capabilities
   the router already checks.
4. **Resume.** The trace and manifest are sufficient to rebuild a population and
   continue; `TraceWriter::append` exists for it.
5. **An MCP server.** `rmcp` is the official Rust SDK, and MCP's Tasks extension
   is a close match for this job contract — that is how the harness gets driven
   by an agent instead of reimplementing an agent loop.
6. **Cost accounting.** GPU-seconds and token spend per run, which the open
   harnesses in this space generally do not surface.

## Licence

Apache-2.0. Models and weights reached through a backend carry their own terms,
which are often more restrictive than this code. Computational results require
experimental validation.
