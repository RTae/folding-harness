# folding-harness

A design harness in Rust: the orchestration, typing and inspection around a
structure-prediction backend, with the backend behind a versioned HTTP contract.

It ships a deterministic fixture backend, so the whole loop runs on a laptop
with no GPU and no network, and a terminal dashboard, so a long run can be
watched over SSH without forwarding a port.

> **Status: working proof of concept.** The cycle, the contract, the gates, the
> population, the trace, the manifest and the dashboard are implemented, and 51
> tests pass. `fh run` completes end to end both in-process and over HTTP, and
> produces the identical run either way.
>
> Not done yet: the model-backed skills (inverse folding, likelihood-guided
> mutation) are declared in the contract and gated by the router but have no
> executor — `fh skills` says which. There is no real folding backend yet; see
> the roadmap.
>
> The fixture backend **predicts nothing**. It scores a sequence by how closely
> the mutable positions match an optimum planted from the run seed, so that
> orchestration can be tested without a model. Anything a real backend produces
> is a computational prediction and needs experimental validation.

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

## Install

Needs a Rust toolchain at 1.82 or newer. If you have `rustup`, the pin in
`rust-toolchain.toml` takes care of it; otherwise install Rust from
[rustup.rs](https://rustup.rs).

```bash
git clone <this repo> && cd folding-harness
cargo install --path . --locked      # puts `fh` in ~/.cargo/bin
fh --version
```

If `fh` is not found afterwards, `~/.cargo/bin` is not on your `PATH` — add it,
or run `rustup update` which sets it up. To remove it later:
`cargo uninstall folding-harness`.

Prefer not to install? `cargo build --release` and use
`./target/release/fh` everywhere `fh` appears below.

## First run

No GPU, no network, no API key: the default backend is a deterministic fixture.

```bash
fh init task.yaml        # write an example task config
fh validate task.yaml    # check it, and print what may and may not change
fh run task.yaml         # run it, streaming events to stdout
```

`fh validate` is worth reading before the first run — it prints the mutable
scope, which is the thing most likely to be wrong, and the routing table, which
says which design skills are available and why the others are not.

Then look at the result:

```bash
fh board                                  # dashboard on the most recent run
fh run task.yaml --board --latency-ms 250 # or watch a run as it happens
```

`--latency-ms` slows the fixture down so the cycles are visible; without it a
14-cycle run finishes in under a second. In the dashboard: `1`–`5` or `Tab`
switch panes, `j`/`k` select a candidate, `h`/`l`/`u`/`d` rotate the backbone,
`+`/`-` zoom, `q` quits.

Other commands: `fh skills` lists the design skills and which are implemented,
`fh ls` lists past runs, `fh doctor task.yaml` checks the setup and the backend.

## Setting up a run by chatting

`fh chat` is the conversational entry point: describe what you want, and the
model turns it into a plan.

```bash
export ANTHROPIC_API_KEY=...        # or: ANTHROPIC_AUTH_TOKEN from `ant auth login`
fh chat --binder binder.fasta --target target.fasta
# or inline, without files:
# fh chat --binder QVQLVESGGG... --target MKTVRQERLK...
```

```
┌─ conversation ───────────────────────────────┐┌─ plan ──────────────────────┐
│ you:                                         ││ task        vhh-cdr-redesign │
│   design the three CDR loops, keep H:52,     ││ topology    vhh              │
│   20 cycles                                  ││ chain H      115 residues    │
│ model:                                       ││ chain A       65 residues    │
│   Three regions on H, H:52 pinned, gates     ││                              │
│   permissive for a first run. {...}          ││ mutable     24 of 115 ...    │
│ harness:                                     ││   loop1  residues 26..33     │
│   Plan validated: 24 mutable of 115 binder   ││   loop2  residues 51..58     │
│   residues across 3 regions. Ctrl-R to run.  ││   loop3  residues 99..108    │
└──────────────────────────────────────────────┘│   fixed  91 cannot change    │
┌─ message ────────────────────────────────────┐│                              │
│ ▮                                            ││ Ctrl-R to run                │
└──────────────────────────────────────────────┘└──────────────────────────────┘
Enter send   Ctrl-R run plan   Ctrl-S save   PgUp/PgDn scroll   Esc quit
```

The pane on the right is **not the model's description of the plan** — it is what
the typed config actually resolves to after the same validation a hand-written
file gets. A hallucinated field fails `deny_unknown_fields`; a region past the
end of a chain fails the scaffold check; a plan whose regions are all pinned
fails as "nothing left to design". When a plan is rejected, the error goes back
to the model so it can correct itself, and `Ctrl-R` does nothing until one
validates. Nothing runs until you press it.

Two deliberate limits:

- **The model is never asked for a sequence.** There is no sequence field in the
  plan schema. Sequences come from `--binder` and `--target` (a FASTA path, or a
  bare sequence on the command line). A model that guessed a target and had it
  folded would produce a confident, meaningless result.
- **It proposes permissive gates for a first run**, because a gate above what the
  starting sequence already scores admits nothing and leaves the search with no
  ratchet. Tighten them once a run is clearing them.

`fh task "<prose>" --binder ... --target ...` is the same thing without the
interface: it prints the plan, writes `task.yaml`, and with `--run` starts it
after a confirmation.

The model only sees chain labels and lengths — not your sequences — when
drafting a plan.

## Running against a service

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

One crate, `src/`, layered bottom up. Everything is Rust; there is no Python,
no Node, and no second runtime.

| path | holds |
| --- | --- |
| `src/ids.rs` | newtype identifiers, so a chain index cannot be passed as a candidate id |
| `src/scaffold.rs` | chains, regions, pinned residues, and `MutableScope` — **the invariant lives here** |
| `src/proposal.rs` | `SkillId`, `Capability`, and the validated `Proposal` |
| `src/candidate.rs` | a candidate and where it came from |
| `src/metrics.rs` | metrics, loss weights, the weighted loss and its breakdown |
| `src/gate.rs` | hard gates and rejection reasons |
| `src/population.rs` | admission: dedup, diversity, capacity, parent selection |
| `src/search.rs` | per-cycle history, skill statistics, stagnation |
| `src/config.rs` | the task YAML, and the one-based to zero-based conversion |
| `src/contract.rs` | the `fh.fold/v1` wire types |
| `src/fold/` | `FoldBackend` trait, `fixture.rs`, `http.rs`, `server.rs` |
| `src/skills/` | design skills and `router.rs` |
| `src/agent.rs` | the decision layer and the `ModelProvider` trait |
| `src/provider.rs` | the Claude Messages API over raw HTTP |
| `src/plan.rs` | prose to a validated task config |
| `src/fasta.rs` | a minimal FASTA reader |
| `src/trace.rs` | event log, task directory layout, run manifest |
| `src/orchestrator.rs` | the cycle |
| `src/tui/` | the dashboard, `chat.rs`, and `structure.rs` (the backbone projection) |
| `src/main.rs` | the `fh` command |
| `src/tests.rs` | tests for the domain layer |

The layering is a convention here rather than something the compiler enforces,
so `CLAUDE.md` states it: `scaffold`/`proposal`/`candidate`/`metrics`/`gate`/
`population`/`search`/`config` do no I/O, `contract` depends only on `serde`,
and nothing below `orchestrator` reaches upward.

## Roadmap

Nearest first:

1. **A real backend adapter.** Boltz-2 (MIT) and Chai-1 (Apache-2.0) ship code
   *and* weights openly, which makes them the practical first targets; both are
   CLI-only, so the adapter is a thin service wrapping a CLI behind
   `src/contract.rs`. `RosettaCommons/foundry` is also worth a look — BSD-3
   covering code *and* weights, one install for RFD3, RoseTTAFold3 and
   ProteinMPNN/LigandMPNN. Tamarind Bio exposes a hosted REST API with a real
   job queue if you would rather not run GPUs. AlphaFold 3's code is Apache-2.0
   but its weights are non-commercial, must not be redistributed (including
   derived weights), and may not be used to train competing predictors — read
   `WEIGHTS_TERMS_OF_USE.md` before wiring it up, and note that AlphaFold Server
   has no API.
   Licensing is not uniform even within one repo: some projects' LICENSE covers
   weights explicitly, some are silent, and some put weights under a separate
   agreement. Record which, per backend, in its adapter.
2. **Output re-verification.** Check the returned structure's sequence against
   the proposal, so a backend that silently renumbers or substitutes is caught
   rather than scored.
3. **Inverse-folding and likelihood-guided executors**, behind the capabilities
   the router already checks.
4. **Resume.** The trace and manifest are sufficient to rebuild a population and
   continue; `TraceWriter::append` exists for it.
5. **An MCP server.** `rmcp` is the official Rust SDK, and MCP's Tasks extension
   is a close match for this job contract — that is how the harness gets driven
   by an agent instead of only driving one.
6. **Cost accounting.** GPU-seconds and token spend per run, which the open
   harnesses in this space generally do not surface.

## Licence

Apache-2.0. Models and weights reached through a backend carry their own terms,
which are often more restrictive than this code. Computational results require
experimental validation.
