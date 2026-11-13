"""Reference `fh.fold/v1` compute service running RosettaCommons/foundry RF3.

Lives in `service/foundry/` under the repo's one-Rust-crate rule: `src/`
never imports from here and no Rust test requires this package. The wire
contract is defined by `src/contract.rs`; `foundry_service.contract` is a
transcription of it, and `tests/test_contract_conformance.py` fails when the
two drift.
"""

__version__ = "0.1.0"
