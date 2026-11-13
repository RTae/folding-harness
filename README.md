# folding-harness

A design harness in Rust. It holds the orchestration, typing and inspection
around a structure-prediction backend, and the backend sits behind a versioned
HTTP contract.

It ships a deterministic fixture backend, so the whole loop runs on a laptop
with no GPU and no network. It has three interfaces: a terminal dashboard for
watching a run over SSH without forwarding a port, a terminal chat screen for
setting a run up by describing it, and a local web view for when a browser
renders better than a terminal does.

> **Status: working proof of concept.** The cycle, the contract, the gates,
> output re-verification, the population, the trace, the manifest, the
> dashboard, the chat screen, the web view and the chat history are
> implemented, and 281 tests pass. Count them with `cargo test` and sum the
> `passed` figures in its `test result` lines. `fh run` completes end to end
> both in-process and over HTTP, and produces the identical run either way.
>
> Not done yet: the model-backed skills, inverse folding and likelihood-guided
> mutation, are declared in the contract and gated by the router but have no
> executor. `fh skills` says which. There is no real folding backend yet. See
> the roadmap.
>
> The fixture backend **predicts nothing**. It scores a sequence by how closely
> the mutable positions match an optimum planted from the run seed, so that
> orchestration can be tested without a model. Anything a real backend produces
> is a computational prediction and needs experimental validation.

## Install

Needs a Rust toolchain at 1.82 or newer. If you have `rustup`, the pin in
`rust-toolchain.toml` takes care of it. Otherwise install Rust from
[rustup.rs](https://rustup.rs).

```bash
git clone <this repo> && cd folding-harness
cargo install --path . --locked      # puts `fh` in ~/.cargo/bin
fh --version
```

If `fh` is not found afterwards, `~/.cargo/bin` is not on your `PATH`. Add it,
or run `rustup update`, which sets it up. To remove `fh` later, run
`cargo uninstall folding-harness`.

To skip installing, run `cargo build --release` and use `./target/release/fh`
everywhere `fh` appears below.

## First run

No GPU, no network, no API key. The default backend is a deterministic fixture.

```bash
fh init                  # writes task.yaml, binder.fasta and target.fasta
fh validate task.yaml    # check it, and print what may and may not change
fh run task.yaml         # run it, streaming events to stdout
```

`fh init` writes the two FASTA files beside the config because `fh chat` and
`fh task` take sequences as files or arguments. Both are placeholders. The
target is synthetic, not a known protein, so replace them before doing anything
real.

Read the `fh validate` output before the first run. It prints the mutable
scope, which is the thing most likely to be wrong, and the routing table, which
says which design skills are available and why the others are not.

Then look at the result:

```bash
fh board                                  # dashboard on the most recent run
fh run task.yaml --board --latency-ms 250 # or watch a run as it happens
```

`--latency-ms` slows the fixture down so the cycles are visible. Without it a
14-cycle run finishes in under a second. Press `?` or `F1` in the dashboard for
the key list.

Other commands: `fh skills` lists the design skills and which are implemented,
`fh ls` lists past runs, and `fh doctor task.yaml` checks the setup and the
backend.

## Setting up a run by chatting

`fh chat` is the conversational entry point. You describe what you want, and the
model turns it into a plan.

```bash
export ANTHROPIC_API_KEY=...        # or: ANTHROPIC_AUTH_TOKEN from `ant auth login`
fh chat --binder binder.fasta --target target.fasta
# or inline, without files:
# fh chat --binder QVQLVESGGG... --target MKTVRQERLK...
```

```
┌─ conversation ───────────────────────────────┐┌─ plan ──────────────────────┐
│ ▌ you                                        ││ task        vhh-cdr-redesign │
│   design the three CDR loops, keep H:52,     ││ topology    vhh              │
│   20 cycles                                  ││ chain H      115 residues    │
│ ▌ model                                      ││ chain A       65 residues    │
│   Three regions on H, H:52 pinned, gates     ││                              │
│   permissive for a first run. {...}          ││ mutable     24 of 115 ...    │
│ ▌ harness                                    ││   loop1  residues 26..33     │
│   Plan validated: 24 mutable of 115 binder   ││   loop2  residues 51..58     │
│   residues across 3 regions. Ctrl-R to run.  ││   loop3  residues 99..108    │
└──────────────────────────────────────────────┘│   fixed  91 cannot change    │
┌─ message ────────────────────────────────────┐│                              │
│ > ▮                                          ││ Ctrl-R to run                │
└──────────────────────────────────────────────┘└──────────────────────────────┘
Enter send   Ctrl-R run   F1 or /help for keys   PgUp/PgDn scroll   Esc quit
```

Replies stream in as the model writes them. The message box is a real editor
and takes several lines. Enter sends, Alt-Enter adds a line, and a paste of
several lines stays one message, because the screen turns on bracketed paste
and reads a paste as one event rather than as a burst of Enter presses. Left
and Right and Home and End move within a line, Ctrl-W deletes a word, Ctrl-U
clears the message, and Up and Down move between lines or recall earlier
messages. Typing `/` opens the command list, which narrows as you type. Up and Down
choose, Tab completes, Esc hides it. The commands `/help`, `/provider`,
`/model`, `/url`, `/retry`, `/plan`, `/run`, `/save [path]`, `/clear` and
`/quit` run locally and never reach the model. `/retry` asks again without adding a message, for a reply that was cut
off mid-JSON. `/model <name>` and `/url <url>` switch where prompts go and
remember the choice for later commands. `F1` prints the same list as `/help`.

The pane on the right is **not the model's description of the plan**. It is what
the typed config resolves to after the same validation a hand-written file
gets. A hallucinated field fails `deny_unknown_fields`. A region past the end of
a chain fails the scaffold check. A plan whose regions are all pinned fails as
"nothing left to design". When a plan is rejected, the error goes back to the
model so it can correct itself, and `Ctrl-R` does nothing until one validates.
Nothing runs until you press it.

Two deliberate limits:

- **The model is never asked for a sequence.** There is no sequence field in the
  plan schema. Sequences come from `--binder` and `--target`, as a FASTA path or
  a bare sequence on the command line. A model that guessed a target and had it
  folded would produce a confident, meaningless result.
- **It proposes permissive gates for a first run.** A gate above what the
  starting sequence already scores admits nothing, so the population never grows
  past the founder and the search has nothing better to select. Tighten the
  gates once a run is clearing them.

`fh task "<prose>" --binder ... --target ...` is the same thing without the
interface. It prints the plan, writes `task.yaml`, and with `--run` starts the
run after a confirmation.

### Provider profiles

Endpoints are named, so switching is a word rather than a URL and a model id
remembered correctly:

```bash
fh config add gateway --url https://gateway.example.com --model claude-opus-4-7
fh config add fast --url https://gateway.example.com --model claude-opus-5-5
fh config list
fh config use anthropic
```

```
* anthropic      (default model) via api.anthropic.com
  fast           claude-opus-5-5 via gateway.example.com
  gateway        claude-opus-4-7 via gateway.example.com
```

`anthropic` is built in and always offered, so the list is never empty and a
deleted profile cannot strand the configuration. Inside `fh chat`, `/provider`
opens a picker over the same profiles: Up and Down to choose, Enter to switch,
Esc to cancel. The switch applies to the running session and is remembered.

`fh config show` prints the active profile's settings and where each value came
from, `fh config set` changes the active profile, and `fh config clear` forgets
everything.

Precedence, highest first: a flag, then the environment, then the saved file,
then the default. The file is a convenience and never a trap, because whatever
you type on the command line is what happens. Inside `fh chat`, `/provider`
shows the same thing and `/model` and `/url` change it without leaving the
screen.

**No API key is stored.** Credentials come from `ANTHROPIC_API_KEY` or
`ANTHROPIC_AUTH_TOKEN`. A config file outlives the session and is easy to
commit by accident, so the settings type has nowhere to put one, and a file
that tries to smuggle one in is rejected.

### A custom endpoint

Any Messages-compatible endpoint works, including a proxy, a gateway, a router,
or something self-hosted:

```bash
export ANTHROPIC_API_KEY=...
fh doctor --probe-model --probe-stream \
  --url https://gateway.example.com \
  --model claude-opus-4-7 \
  --api-version 2023-06-01
```

`--probe-model` sends one tiny request and prints what came back, so a wrong
URL, key or model name is a one-line answer rather than a failure in the middle
of a plan. `--probe-stream` checks the streaming path, which is what `fh chat`
uses. An endpoint can serve ordinary requests without supporting server-sent
events, and then `fh chat` will not work while `fh task` still will.

Probe each model you intend to use. A gateway can serve one model and fail on
another, including by declining a request that another model accepts. The same
flags work on `fh chat` and `fh task`, and `ANTHROPIC_BASE_URL`,
`ANTHROPIC_MODEL` and `ANTHROPIC_API_VERSION` set them from the environment.

Two things change when the URL is not the first-party API:

- **The request drops to the compatible subset.** It sends `model`,
  `max_tokens`, `system` and `messages`. Server-side refusal fallback and
  `output_config.effort` are features of Anthropic's own API, and a gateway that
  does not know a field may reject the whole request over it. Pass
  `--effort high` to send it anyway if your endpoint supports it, or
  `--max-tokens` to lower the cap.
- **The harness prints the host before it sends any prompt**, and flags it as
  third-party. Prompts leave for that host, so whoever runs it can read them. A
  gateway may also return its own errors as ordinary HTTP 200 replies, which
  read as though the model said them. When a reply cannot be parsed, the harness
  prints it verbatim and names the host it came from.

When drafting a plan the harness sends chain labels and lengths, never your
sequences. That holds for this one code path, and it is not a guarantee about
anything you type into the chat.

Keys belong in the environment, never in a task config or a committed file.

## Running against a service

To exercise the real network path, run the stub service and point a task at it:

```bash
fh serve --task task.yaml --latency-ms 200
fh run task.yaml --url http://127.0.0.1:8080
```

`fh serve` binds loopback on `--port`, 8080 by default, and authenticates
nobody. The only difference between the two modes is whether the bytes crossed
a socket.

## The foundry compute service

`service/foundry/` is the first real backend: a Python service that speaks
`fh.fold/v1` and runs RF3 from RosettaCommons/foundry behind it, pinned to
one commit. It is the `service/` exception in `AGENTS.md`: a folding model
needs Python and a GPU, `src/` never imports from `service/`, and no Rust
test requires it.

Two things a reader of its numbers must know. RF3 does not report
`interface_plddt` or `interface_contacts`, and the loss weights both and the
gates threshold both, so the service derives them from the predicted
coordinates and records them as adapter-derived in
`foundry_service/derive.py`. And an early-stopped RF3 run writes no structure
at all, so the service disables early stopping explicitly and fails the job
with a reason if one shows up anyway. The weights are BSD-3: foundry's
`LICENSE.md` covers the weights in terms, and the licence sentence and the
pinned commit are recorded in `foundry_service/rf3.py`.

Deploy it on the GPU machine and point the harness at it. Its own README
carries the install steps and the honest list of what no test here could
verify, since the RF3 call needs a GPU.

```bash
cd service/foundry
pip install -e ".[rf3]" && foundry install base-models
python -m foundry_service --host 127.0.0.1 --port 8080
```

The service's own tests need no GPU, no foundry and no Linux, and they bind
port 0:

```bash
cd service/foundry
uv run --no-project --python 3.13 --with pytest python -m pytest
```

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

Two separations carry the design.

**Legality is not choice.** The router decides what may be used, from the
user's enabled list, the capabilities the service advertises, whether an
executor exists, and the search state. The agent only picks from that set, so
it cannot reach a skill the service cannot serve.

**Gates are not the loss.** The loss ranks candidates that are acceptable.
Gates decide what is acceptable at all. A gate failure is recorded against the
candidate rather than weighted away.

## What a run leaves behind

```
~/.folding_harness/tasks/<task-id>/
  task.yaml          resolved config, as used
  run.json           manifest: seed, backend, model, protocol, config SHA-256, chat id
  trace.jsonl        every event, in order, flushed per line
  population.json    current members, loss curve, per-cycle records
  best.json          best candidate
  structures/        alpha-carbon trace per candidate
```

`run.json` is the part most pipelines leave implicit. With the seed, the config
digest and the service version, you can re-run a result and show that it used
the same inputs. Set `FH_TASK_ROOT` to move the directory.

### Resume a run

`fh resume` continues a recorded run from its last completed cycle.

```bash
fh run task.yaml --cycles 4     # stops after 4 cycles, or is interrupted
fh resume                       # continue the most recent task
fh resume --cycles 16           # extend the planned total and continue
fh resume --task <task-id>      # or --path <task-dir>, like fh board
```

The resumed run rebuilds the population, the per-cycle records and the
admission counters from the task directory. It appends to `trace.jsonl` with
the sequence numbers continuing, and candidate ids continue after the highest
id the recorded run assigned. `run.json` gains `resumed_from_cycle`,
`generator_reseeded`, `resumed_protocol` and `resumed_service_version`, and
its `usage` reports both runs' spend. Its original `protocol` and
`service_version` keep describing the recorded run, because `folds` and
`best_candidate` span both runs.

Three limits are deliberate.

- **The config must match.** The resume loads the `task.yaml` saved in the
  task directory and refuses to run when its digest is not the `config_sha256`
  in `run.json`. Changing the gates mid-search would mix two acceptance
  criteria in one population and one lineage, so no flag allows it. Start a
  new task with `fh run` instead. `--cycles` extends the planned total
  without touching the config, so extending a run is not a config change.
- **The snapshot must be genuine.** A resume folds the members it restores
  and designs from them, so it first checks every member of `population.json`
  against the config and the trace: the scaffold's binder chains at the
  scaffold's lengths, the config's residue at every position outside the
  mutable scope, and the loss the member's own metrics and the trace's
  scoring event produce. One failing member refuses the whole resume, and the
  refusal names the file and the candidate. Restore the file from a copy the
  recorded run wrote, or run the task again.
- **The generator is re-seeded.** The manifest records the seed but not the
  position in its stream, so a resumed run seeds a fresh generator and its
  parent selection and proposals diverge from what one uninterrupted run
  would have produced. The divergence has a second source: `snapshot()`
  writes members in ranked order, so the resumed population restores a
  different member order from the recorded run's insertion order, and
  `select_parent` iterates members in that order. The original order is not
  recoverable. The same config, seed and backend reproduce the same
  trace for one uninterrupted run, but not across a resume. The
  `task_resumed` trace event and the manifest record the re-seed rather than
  hiding it.

## The dashboard

`fh board` has five panes. **Overview** shows the loss curve, the per-cycle
table, rejection counts, and the last routing decision with the reason each
skill was refused. **Population** shows a ranked table, the loss broken down by
term, and the sequence with fixed residues in grey, mutable in green and
changed in yellow. **Lineage** shows a parent and child tree of the retained
candidates. **Structure** is below. **Events** shows the trace.

Press `?` or `F1` for the full key list, grouped, with the pane you are on
marked. The short version: `q` quits, `1` to `5` and `Tab` switch panes, `j`
and `k` select a candidate, and `f` follows the best one.

### The structure pane

The predicted backbone, drawn with braille glyphs. Each cell holds 2 by 4
addressable dots, so a modest pane holds a few hundred points. The pane joins
alpha carbons into a backbone and colours them by confidence band, with blue,
cyan, yellow and red for very high through very low, and the fixed target in
grey. `h` and `l` turn it, `u` and `d` tilt it, `+` and `-` zoom, and `0`
resets.

The projection is orthographic. The canvas bounds correct for the braille dot
aspect so the shape is not stretched sideways, because a cell is 2 dots wide
and 4 tall, so equal world distances need unequal world *ranges* on the two
axes.

To see a pane without taking over the terminal, over a pipe, in a note, or in
CI, render one frame as text:

```bash
fh board --snapshot --pane structure --width 100 --height 32
fh board --snapshot --pane structure --view 0.8,1.2    # yaw,pitch in radians
fh board --snapshot --pane overview --path <task dir>
fh board --snapshot --keys                             # print the key list
```

Colour carries the confidence bands in the live pane and a snapshot drops it.
The braille still carries the shape.

## Measuring the search

`fh bench` asks whether a search policy works, which a loss curve going down
does not answer:

```bash
fh bench                                  # all policies, 5 seeds
fh bench --seeds 24 --cycles 30
fh bench --policy heuristic --policy random --json out.json
```

The fixture plants an optimum derived from the seed, so unlike a real run this
one has ground truth. The benchmark reports **recovery**: the share of planted
positions the best candidate actually found. `random`, which draws uniformly
from the legal skills, is the control, and `founder-only` folds the starting
sequence and stops, which is the floor.

What it measures is orchestration against a synthetic landscape. It says
nothing about whether a design would bind, and it cannot, because the fixture
does not fold.

## The web view

`fh web` serves the same recorded runs in a browser, for when pixels beat
braille:

```bash
fh web                      # http://127.0.0.1:8750
fh web --port 9000 --task-root ./runs
```

It binds to loopback only. `fh::web::serve` takes a port and binds the loopback
address itself, and the router it serves is not public, so no external crate can
obtain that router and bind it to a wider address. The view serves local run
data and authenticates nobody. The page, its stylesheet and its script are
compiled into the binary, so there is still no Node, no asset directory and no
build step. The server
only reads the task directory, so having the page open cannot disturb a run,
and a run does not need the page.

Panes: the loss curve with best and mean and a band between them, hover for
the cycle behind any point; a per-cycle strip of admitted against rejected; the
retained population; a lineage tree whose nodes are coloured green through red
by loss and which you can click to select; the loss broken down by term; the
sequence with its fixed, mutable and changed residues; the event stream; and
the backbone.

The backbone canvas drags to rotate, scrolls to zoom, and has a spin toggle.
Four colour modes answer different questions: **confidence** is the usual
pLDDT banding, **changed residues** shows in 3D which positions differ from the
starting sequence, **mutable scope** shows which the config allows to change at
all, and **chain** separates binder from target. The canvas is sized to the
display's pixel density, so it is crisp rather than upscaled.

While a run is live the page polls every 1.5 seconds and stops when the run
finishes.

The server resolves the mutable scope itself and sends the browser one boolean
per residue, so the page never parses a task config. One definition of the
scope, in Rust, shared by both interfaces.

## Chat history

Every conversation is saved after each exchange, so closing the terminal does
not lose it:

```bash
fh chats                       # list saved chats
fh chats --show <id>           # print one transcript
fh chat --continue             # carry on the most recent
fh chat --resume <id>          # carry on a particular one
```

Resuming reuses the sequences stored with the chat, so `--binder` and
`--target` are only needed for a fresh one. A resumed chat re-derives its plan
from the last reply rather than trusting a stored verdict, so the same
validation applies either way.

A run started from a chat is linked in both directions: the chat record lists
the task ids it produced, and the run's `run.json` carries the `chat_id`. That
answers "which conversation asked for this run" from either end.

Chat records live in `~/.folding_harness/chats`, or `$FH_CHAT_ROOT`. They
contain the sequences you supplied, the same material the task config already
stores beside each run.

## The compute contract

`src/contract.rs` is the whole protocol and depends only on `serde`, so a
service can implement it without the rest of the harness:

```
GET  /v1/health          -> Health   { protocol, capabilities, gpus, queue_depth }
POST /v1/fold            -> JobAccepted { job_id }      (returns immediately)
GET  /v1/fold/{job_id}   -> JobStatus   { state, result }
```

Submission is accept-then-poll rather than one long blocking request. A fold
outlives the client, a crashed client can reattach to a job id, and the service
can report a queue. `src/fold/server.rs` is a working reference implementation.

Capabilities drive the router. A service that advertises only `fold` will never
be offered an inverse-folding skill.

## Layout

One crate, `src/`, layered bottom up. Everything in the crate is Rust. There
is no Python, no Node, and no second runtime inside it. The one exception is
`service/`, which holds reference compute services implementing
`src/contract.rs` from the other side, the way `AGENTS.md` allows.

| path | holds |
| --- | --- |
| `src/ids.rs` | newtype identifiers, so a chain index cannot be passed as a candidate id |
| `src/scaffold.rs` | chains, regions, pinned residues, and `MutableScope`, where the invariant lives |
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
| `src/provider/` | the Claude Messages and OpenAI-compatible APIs over raw HTTP, with streaming, and the SSE parser |
| `src/plan.rs` | prose to a validated task config |
| `src/fasta.rs` | a minimal FASTA reader |
| `src/trace.rs` | event log, task directory layout, run manifest |
| `src/history.rs` | saved chats, and the link between a chat and its runs |
| `src/bench.rs` | policy comparison against the fixture's planted optimum |
| `src/settings.rs` | remembered provider settings, and where each value came from |
| `src/usage.rs` | what a run spent: backend milliseconds and model tokens |
| `src/web/` | the local web view: `mod.rs`, plus the page, script and stylesheet |
| `src/orchestrator.rs` | the cycle |
| `src/tui/` | the dashboard, `chat.rs`, the `input.rs` line editor, `command.rs` for slash commands, and `structure.rs` for the backbone projection |
| `src/cli/` | the `fh` subcommands. These belong to the binary, so `anyhow` and the terminal stay out of the library |
| `src/main.rs` | argument parsing, then a hand-off to `cli::dispatch` |
| `src/tests.rs` | tests for the domain layer |
| `service/foundry/` | the Python compute service running foundry RF3 behind `fh.fold/v1`. Not part of the crate |

The layering is a convention here rather than something the compiler enforces,
so `CLAUDE.md` states it. The modules `scaffold`, `proposal`, `candidate`,
`metrics`, `gate`, `population`, `search` and `config` do no I/O, `contract`
depends only on `serde`, and nothing below `orchestrator` reaches upward.

## Roadmap

Nearest first.

1. **A real backend adapter.** Boltz-2 under MIT and Chai-1 under Apache-2.0
   ship code *and* weights openly, which makes them the practical first
   targets. Both are CLI-only, so the adapter is a thin service wrapping a CLI
   behind `src/contract.rs`. `RosettaCommons/foundry` is also worth a look,
   since its BSD-3 licence covers code and weights and one install gives RFD3,
   RoseTTAFold3 and ProteinMPNN with LigandMPNN. Tamarind Bio exposes a hosted
   REST API with a real job queue if you would rather not run GPUs. AlphaFold
   3's code is Apache-2.0 but its weights are non-commercial, must not be
   redistributed including derived weights, and may not be used to train
   competing predictors. Read `WEIGHTS_TERMS_OF_USE.md` before wiring it up,
   and note that AlphaFold Server has no API.
   Licensing is not uniform even within one repository. Some projects' LICENSE
   covers weights explicitly, some are silent, and some put weights under a
   separate agreement. Record which, per backend, in its adapter.
2. **Inverse-folding and likelihood-guided executors**, behind the capabilities
   the router already checks.
3. **Prompt caching in the chat screen.** A long conversation resends its whole
   history each turn. Caching the system prompt would cut that, and it only
   works against the first-party API.
4. **An MCP server.** `rmcp` is the official Rust SDK, and MCP's Tasks
   extension is a close match for this job contract. That is how the harness
   gets driven by an agent instead of only driving one.
5. **Cost accounting.** GPU-seconds and token spend per run, which the open
   harnesses in this space generally do not surface.

## Licence

Apache-2.0. Models and weights reached through a backend carry their own terms,
which are often more restrictive than this code. Computational results require
experimental validation.
