this codebase is your. what it's for and how it's for is you call and you can build on this

---

# Repository rules

One Rust crate, `src/`, no Python and no second runtime. `fh` is both the
library name and the binary.

## The invariant that must not regress

A binder sequence may only change through `MutableScope::propose`, which
validates every substitution against the scope, and `Candidate::from_proposal`.
Concretely:

- Do not add a public constructor for `Proposal`, do not derive `Deserialize`
  for it, and do not add a method that mutates a `BinderSequence` outside
  `src/scaffold.rs`. `BinderSequence::replace` is `pub(crate)` on purpose.
- Do not let a skill assemble a `Candidate` directly and bypass `propose`.
- Residue positions are **one-based inclusive in YAML** and **zero-based
  half-open everywhere else**. The single conversion is in
  `TaskConfig::to_scaffold`.

Covered by `src/tests.rs`, `src/skills/mod.rs`, and the end-to-end check
`no_run_ever_changes_a_position_outside_the_mutable_scope` in
`src/orchestrator.rs`. A change that makes those pass by weakening an assertion
is not a fix.

## Layering

The compiler does not enforce this, so keep to it by hand. The domain modules
(`scaffold`, `proposal`, `candidate`, `metrics`, `gate`, `population`, `search`,
`config`) do no I/O and must not gain an HTTP or filesystem dependency.
`contract` depends only on `serde`, so the wire types stay liftable into their
own crate for a service implementer. `fold`, `skills`, `agent` and `trace` sit
above the domain; `orchestrator`, `tui` and `main` above those. Nothing lower
reaches upward.

Two separations carry the design and should survive refactors:

- **Legality is not choice.** `skills::router` decides what may run; `agent`
  only picks from that set. An agent must never be able to widen it.
- **Gates are not the loss.** `gate` decides what is acceptable at all; `metrics`
  ranks what already is. A gate failure is recorded against the candidate, never
  weighted away.

## Honesty in output

- The fixture backend is a test double and predicts nothing. No user-facing text
  may imply otherwise.
- Every rejected candidate gets a recorded reason. Never drop one silently.
- Do not claim a capability the router has not checked, or a verification that
  did not run.
- Model weights often carry terms stricter than their code. When adding a
  backend adapter, record its weights licence in the adapter.

## Commands

```bash
cargo build
cargo test
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

A GPU, a compute service and the network may all be absent; the fixture path
must keep working without them. Say in your report what you did not run.

## Code and tests

- `thiserror` for library errors, `anyhow` only in `src/main.rs`.
- Every user-causable error should name the field or file and say what to do.
- Tests live beside the code in `#[cfg(test)] mod tests`, named after the
  behaviour they pin rather than the function they call.
- New wire fields are additive and `#[serde(default)]`, or the protocol major
  version changes.
- Randomness goes through `crate::RunRng` (ChaCha8) seeded from
  `design.seed`, never from entropy — a run must replay from its manifest.

## Commits

Conventional Commits, `<type>(<scope>): <subject>`, scope is a module name
(`scaffold`, `fold`, `tui`, …), lowercase subject, no trailing period. Commit
and push only when asked, and never with failing checks.
