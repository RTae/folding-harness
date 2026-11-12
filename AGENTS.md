<!--
SPDX-FileCopyrightText: Copyright (c) 2026 folding-harness contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

This repo is **folding-harness**. One Rust crate ships both the `fh` library
and the `fh` binary. The harness owns orchestration, typing and inspection
around a structure-prediction backend. The backend sits behind a versioned
HTTP contract, so nothing here touches a tensor or a GPU.

Read [`README.md`][readme] for what the tool does and how to drive it. This
file states how to change it. Model weights and backends carry their own
terms, which are often stricter than the Apache-2.0 on this code.

## Project checks

- **One Rust crate in `src/`.** No Python, no Node, no second runtime, no
  build step for assets. The web view's page, script and stylesheet compile
  into the binary.
- **Run all four commands before you call a change done.**

  ```bash
  cargo build
  cargo test
  cargo fmt --all
  cargo clippy --all-targets -- -D warnings
  ```

- **Install the pre-commit hook once.** It runs the two fast checks, so a
  misformatted or warning-carrying commit cannot land. There is still no CI, so
  the hook and your own runs are the whole gate.

  ```bash
  cp scripts/pre-commit .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit
  ```

  The hook is a copy rather than a symlink or a `core.hooksPath` setting, so it
  stays out of the git config and out of anyone else's clone until they install
  it. Fix what it reports. Do not skip it.

- **The fixture path must work with nothing else present.** A GPU, a compute
  service and the network may all be absent. A change that makes `fh run`
  depend on one of them is a regression.
- **Say what you did not run.** Report the checks you skipped and why. A claim
  of verification that did not happen is worse than a gap you named.

## The invariant that must not regress

A binder sequence may change only through `MutableScope::propose`, which
validates every substitution against the scope, and `Candidate::from_proposal`.
`Proposal` has no `Deserialize` impl, so a sequence edit cannot arrive from a
config file or a network payload. The property is carried by the types rather
than by a rule someone has to remember. Concretely:

- Do not add a public constructor for `Proposal`, do not derive `Deserialize`
  for it, and do not add a method that mutates a `BinderSequence` outside
  `src/scaffold.rs`. `BinderSequence::replace` is `pub(crate)` on purpose.
- Do not let a skill assemble a `Candidate` directly and bypass `propose`.
- Residue positions are **one-based inclusive in YAML** and **zero-based
  half-open everywhere else**. The single conversion is in
  `TaskConfig::to_scaffold`.

Three places cover this: `src/tests.rs`, `src/skills/mod.rs`, and the
end-to-end check `no_run_ever_changes_a_position_outside_the_mutable_scope` in
`src/orchestrator.rs`. A change that makes those pass by weakening an assertion
is not a fix.

## Layering

The compiler does not enforce this, so keep to it by hand. Nothing lower
reaches upward.

- **Domain, no I/O.** `scaffold`, `proposal`, `candidate`, `metrics`, `gate`,
  `population`, `search`, `config`. These must not gain an HTTP or filesystem
  dependency.
- **Wire types.** `contract` depends only on `serde`, so a service implementer
  can lift it into its own crate. New wire fields are additive and
  `#[serde(default)]`, or the protocol major version changes.
- **Above the domain.** `fold`, `skills`, `agent`, `provider`, `trace`.
- **Top.** `orchestrator`, `tui`, `web`, `cli`, `main`.

Two separations carry the design and should survive refactors:

- **Legality is not choice.** `skills::router` decides what may run, from the
  user's enabled list, the capabilities the service advertises, whether an
  executor exists, and the search state. `agent` only picks from that set. An
  agent must never be able to widen it.
- **Gates are not the loss.** `gate` decides what is acceptable at all.
  `metrics` ranks what already is. A gate failure is recorded against the
  candidate, never weighted away.

## Honesty in output

- The fixture backend is a test double and predicts nothing. It scores a
  sequence by how closely the mutable positions match an optimum planted from
  the run seed. No user-facing text may imply otherwise.
- Every rejected candidate gets a recorded reason. Never drop one silently.
- Do not claim a capability the router has not checked, or a verification that
  did not run.
- Model weights often carry terms stricter than their code. When you add a
  backend adapter, record its weights licence in the adapter.
- Anything a real backend produces is a computational prediction and needs
  experimental validation.

## Code and tests

- `thiserror` for library errors. `anyhow` only in the binary, under `src/cli/`
  and `src/main.rs`.
- Every user-causable error names the field or the file and says what to do.
- Tests live beside the code in `#[cfg(test)] mod tests`, named after the
  behaviour they pin rather than the function they call.
- No test sleeps and no test depends on a time window. One exception: a bound
  that turns a deadlock into a named failure is a guard, not a timing
  assertion. It is allowed only where the test waits on a channel that the code
  under test signals, the healthy case completes orders of magnitude inside the
  bound, and the failure message names the property that did not hold. A test
  that proves incremental behaviour needs this, because the alternative is a
  suite that hangs instead of reporting.
- The test runner takes no seed, so a rule that asks for three runs under
  three seeds translates rather than applies. `rust-toolchain.toml` pins
  stable, and libtest's `--shuffle` seed is nightly only. Run the suite three
  times, once with `--test-threads=1`, and say that the runs varied by thread
  scheduling rather than by seed. Run randomness comes from `design.seed`
  inside the tests and is meant to stay fixed.
- Randomness goes through `crate::RunRng` (ChaCha8) seeded from `design.seed`,
  never from entropy. A run must replay from its manifest. ChaCha8 rather than
  `StdRng` because `rand` documents `StdRng`'s output as unstable across
  releases, and a recorded seed would stop reproducing its run after a
  dependency bump.
- Prefer a fake you write by hand to a mocking library.

## Git history and commits

- **Conventional Commits**: `<type>(<scope>): <subject>`. The scope is a module
  name (`scaffold`, `fold`, `tui`, …). Lowercase subject, no trailing period.
- **Commit and push only when asked.** Never commit with a failing check. Never
  skip a hook to get past one.
- **Write the commit body as a briefing.** State the previous problem, the new
  behaviour, and the reason for the approach. Let the diff explain how. Wrap
  prose at 72 characters and separate paragraphs with a blank line. Do not
  repeat the subject and do not inventory files.
- **Post validation results as a comment, not as a body section.** Link the
  exact revision. State what each result proves and what stays unverified.
- Before any command that could discard work, run `git status`. Note that
  `.gitignore` excludes `/task.yaml`, `/tasks/` and `/runs/`, so run artifacts
  are never tracked.

## Running work in parallel

When two changes touch no file in common, start them at the same time rather
than one after the other. Give each its own git worktree and branch, so a check
run reads the code under change and not the copy on `main`.

- Create each isolation with `git worktree add -b <branch> <path> main`, and
  read the commit it landed on. `main` moves while work is in flight, and a
  branch that predates a doc fix conflicts with it.
- Keep exactly one owner per file. Two changes editing one file corrupt each
  other, and a check run in one tree tells you nothing about the other.
- Merge back only when asked. Confirm a commit landed with `git log` and
  `git status` rather than trusting an exit code, because more than one session
  may share this tree and commit first.
- Never reword or rebase a commit another session made.
- Prepare work that needs an approval, then bring the question. A push, a
  merge, a launch and a delete all wait for a yes.

## Writing and documentation

Docs, PR descriptions and commit messages are writing. Apply the
`technical-writing` skill to them, and `unslop` to every doc you touch.

- **Write terse, direct, active prose.** Lead with facts. Short sentences, one
  idea each. Cut every word that does no work.
- **No em-dashes and no prose semicolons.** Start a new sentence instead. Do not
  open a sentence with "Since" or "Because".
- **Connect adjacent sentences.** When one is not a direct continuation of the
  last, add a sentence that says how they relate. Do not make the reader guess.
- **Stack at most two modifiers before a noun.** When you need a third, explain
  the innermost concept in a full sentence first, then build outward.
- **Use at most two abstract subjects with action verbs per paragraph.** From
  the third sentence on, make a concrete person, thing or scenario the subject.
- **Call each thing by one name, everywhere.** The codebase is the word list.
  Write the real symbol, file, flag or command name, not a synonym for it.
- **State a ratio in words.** Write "A takes r times the time of B" or "r times
  as fast as B". Never "r times faster than".
- **Cite `file:line` for any claim about code behaviour**, and trace the
  resolved value rather than infer a default.
- **Prefer a list to a table.** Use a table only where the cells line up for
  numerical data or a comparison.
- **Make every count, path and tree claim true at the commit that lands it**,
  and include the command that regenerates it.

## Agent skills and agents

Skills install globally, under `~/.qwen/skills/`. They come from
[vtno/skills][vtno], and each installed folder carries a `SOURCE.md` that names
the upstream commit and any local change. Agents live in `~/.qwen/agents/`.

- `implement` drives one plan step to a review-clean change. It briefs an
  author agent, then a separate report-only reviewer, and repeats until no hard
  violation, wrong or missing requirement, or real bug remains.
- `engineering-rules` is how code gets built here: innermost layer first,
  interfaces defined by the caller, hand-written fakes, tests that do not sleep.
- `code-review` reviews a diff on two axes, Standards and Spec, and reports them
  side by side without ranking one against the other.
- `technical-writing` and `unslop` cover docs, PR descriptions and commit
  messages.
- The `code-author` agent builds one plan step. The `code-reviewer` agent
  reports on a diff and changes nothing. Keep them separate: a review by the
  author of the diff is not a review.

[readme]: README.md
[vtno]: https://github.com/vtno/skills
