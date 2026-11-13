"""Conformance: the Python wire models must match `src/contract.rs`.

The Rust file is parsed at test time, so adding, renaming or removing a
field, an enum variant or a serde rule on either side fails here. Field
types are compared too: `_python_type` maps every Rust type to the Python
annotation it must be transcribed as, and each unsigned integer field is
probed at the boundary of the width the Rust source declares. Boolean and
string fields are probed with the values their Rust type rejects. Nothing
is hardcoded that the parser can read, except the type maps and the example
wire payloads.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from pathlib import Path

import pytest

from foundry_service import contract as wire

CONTRACT_RS = Path(__file__).resolve().parents[3] / "src" / "contract.rs"


class RustField:
    def __init__(self, name: str, rust_type: str, has_default: bool, has_skip: bool) -> None:
        self.name = name
        self.type = rust_type
        self.has_default = has_default
        self.has_skip = has_skip


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def parse_contract() -> tuple[dict[str, list[RustField]], dict[str, list[str]], dict[str, str]]:
    text = CONTRACT_RS.read_text(encoding="utf-8")
    structs: dict[str, list[RustField]] = {}
    enums: dict[str, list[str]] = {}

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        struct_match = re.match(r"pub struct (\w+)", line)
        enum_match = re.match(r"pub enum (\w+)", line)
        if struct_match:
            name = struct_match.group(1)
            fields: list[RustField] = []
            attrs: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("}"):
                stripped = lines[index].strip()
                if stripped.startswith("#["):
                    attrs.append(stripped)
                elif stripped.startswith("pub ") and ":" in stripped:
                    field_match = re.match(r"pub (\w+):\s*(.+)", stripped)
                    joined = " ".join(attrs)
                    fields.append(
                        RustField(
                            name=field_match.group(1),
                            rust_type=field_match.group(2).rstrip(",").strip(),
                            has_default="serde(default)" in joined,
                            has_skip="skip_serializing_if" in joined,
                        )
                    )
                    attrs = []
                elif not stripped.startswith("//"):
                    attrs = []
                index += 1
            structs[name] = fields
        elif enum_match:
            name = enum_match.group(1)
            variants: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("}"):
                stripped = lines[index].strip()
                variant = re.match(r"^(\w+),?$", stripped)
                if variant and variant.group(1) not in ("default",):
                    variants.append(_snake(variant.group(1)))
                index += 1
            enums[name] = variants
        index += 1

    consts = dict(re.findall(r'pub const (\w+): &str = "([^"]+)"', text))
    return structs, enums, consts


STRUCTS, ENUMS, CONSTS = parse_contract()

PYTHON_TYPES = {
    "ChainSpec": wire.ChainSpec,
    "JobSpec": wire.JobSpec,
    "FoldRequest": wire.FoldRequest,
    "JobAccepted": wire.JobAccepted,
    "JobStatus": wire.JobStatus,
    "ConfidenceReport": wire.ConfidenceReport,
    "CaResidue": wire.CaResidue,
    "StructurePayload": wire.StructurePayload,
    "FoldResult": wire.FoldResult,
    "GpuInfo": wire.GpuInfo,
    "Health": wire.Health,
    "ErrorBody": wire.ErrorBody,
}

ENUM_TYPES = {
    "MoleculeKind": wire.MoleculeKind,
    "JobState": wire.JobState,
    "StructureFormat": wire.StructureFormat,
}

# JSON carries one number type, so a payload cannot tell a Rust integer from
# a Rust float when the value is whole, and Python's int is unbounded, so an
# annotation cannot tell a u8 from a u32. The annotation check below still
# separates int from float, and
# test_integer_fields_enforce_the_rust_bit_width probes the width the
# annotation cannot carry.
SCALAR_TYPES = {
    "String": str,
    "bool": bool,
    # A Rust char travels as a one-character JSON string. Python's str is the
    # counterpart. The one-character limit is not expressible in a Python type
    # annotation, so CaResidue.from_wire checks it at parse time. That check
    # covers the length only. A char redeclared as String in Rust still maps
    # to str and passes, the way an f32 redeclared as f64 passes.
    "char": str,
    "u8": int,
    "u32": int,
    "u64": int,
    # f32 and f64 both arrive as the one JSON number type and both land in
    # Python's float, so the two Rust widths share one counterpart.
    "f32": float,
    "f64": float,
}

UINT_BITS = {"u8": 8, "u32": 32, "u64": 64}


def _python_type(rust_type: str, type_name: str):
    """The Python annotation a Rust wire type must be transcribed as."""
    name = rust_type.strip()
    for table in (SCALAR_TYPES, ENUM_TYPES, PYTHON_TYPES):
        if name in table:
            return table[name]
    vec = re.fullmatch(r"Vec<(.+)>", name)
    if vec:
        return list[_python_type(vec.group(1), type_name)]
    option = re.fullmatch(r"Option<(.+)>", name)
    if option:
        return _python_type(option.group(1), type_name) | None
    array = re.fullmatch(r"\[(.+); (\d+)\]", name)
    if array:
        inner = _python_type(array.group(1), type_name)
        return tuple[(inner,) * int(array.group(2))]
    raise AssertionError(
        f"{type_name}: the Rust type {rust_type!r} has no Python mapping. "
        "Extend SCALAR_TYPES, ENUM_TYPES or _python_type in this test"
    )


def _uint_shape(rust_type: str):
    """(bits, wrap) when the Rust type carries an unsigned integer, either
    bare, in an Option or in a Vec. wrap turns a boundary value into the
    shape the field holds on the wire. None for every other type."""
    name = rust_type.strip()
    if name in UINT_BITS:
        return UINT_BITS[name], lambda value: value
    option = re.fullmatch(r"Option<(.+)>", name)
    if option:
        return _uint_shape(option.group(1))
    vec = re.fullmatch(r"Vec<(.+)>", name)
    if vec and _uint_shape(vec.group(1)):
        return _uint_shape(vec.group(1))[0], (lambda value: [value])
    return None


def _str_shape(rust_type: str):
    """(wrap, null_ok) when the Rust type carries a String, either bare, in an
    Option or in a Vec. wrap turns a probe value into the shape the field holds
    on the wire, and null_ok says whether JSON null means absent rather than a
    type error. None for every other type."""
    name = rust_type.strip()
    if name == "String":
        return (lambda value: value), False
    option = re.fullmatch(r"Option<(.+)>", name)
    if option and _str_shape(option.group(1)):
        return _str_shape(option.group(1))[0], True
    vec = re.fullmatch(r"Vec<(.+)>", name)
    if vec and _str_shape(vec.group(1)):
        return (lambda value: [value]), False
    return None


def _transcribed_names(type_name: str) -> set[str]:
    """The Rust fields the Python model must carry on the wire: every field
    except an additive rust-only one that carries #[serde(default)]."""
    python_names = {f.name for f in dataclasses.fields(PYTHON_TYPES[type_name])}
    return {
        field.name
        for field in STRUCTS[type_name]
        if field.name in python_names or not field.has_default
    }


def _job_spec_wire() -> dict:
    return {
        "name": "job",
        "model": "rf3",
        "chains": [{"label": "H", "sequence": "QVQL", "kind": "protein", "designed": True}],
        "seeds": [1],
        "diffusion_samples": 2,
        "diffusion_steps": 8,
        "use_msa": False,
        "need_atom_confidence": True,
    }


def _confidence_wire(with_plddt: bool) -> dict:
    payload = {
        "iptm": 0.8,
        "ptm": 0.9,
        "plddt_mean": 80.0,
        "interface_plddt": 72.5,
        "pae_interface": 3.0,
        "interface_contacts": 1,
        "clashes": 0,
        "hydrophobic_patch": 0.5,
        "net_charge": -1.0,
        "atom_confidence": True,
    }
    if with_plddt:
        payload["plddt"] = [80.0]
    return payload


def _ca_wire() -> dict:
    return {
        "chain_index": 0,
        "index": 0,
        "residue": "Q",
        "xyz": [1.0, 2.0, 3.0],
        "plddt": 80.0,
        "designed": True,
    }


def _structure_wire(full: bool) -> dict:
    if full:
        return {
            "format": "ca_trace",
            "inline": "data",
            "artifact_url": "http://x/y.cif",
            "ca_trace": [_ca_wire()],
        }
    return {"format": "cif"}


def _result_wire() -> dict:
    return {
        "job_id": "job-1",
        "model": "rf3",
        "confidence": _confidence_wire(True),
        "structure": _structure_wire(True),
        "elapsed_ms": 5,
    }


def _gpu_wire(full: bool) -> dict:
    payload = {"index": 0, "name": "A100", "memory_total_mb": 40960}
    if full:
        payload["leased_by"] = "job-1"
    return payload


# (full wire, minimal wire) per type. The full payload populates every
# field; the minimal one drops every `#[serde(default)]` field and leaves
# every `skip_serializing_if` field at its None/empty value.
PAYLOADS: dict[str, tuple[dict, dict]] = {
    "ChainSpec": (
        {"label": "H", "sequence": "QVQL", "kind": "protein", "designed": True},
        {"label": "H", "sequence": "QVQL"},
    ),
    "JobSpec": (_job_spec_wire(), _job_spec_wire()),
    "FoldRequest": (
        {"protocol": "fh.fold/v1", "job": _job_spec_wire()},
        {"protocol": "fh.fold/v1", "job": _job_spec_wire()},
    ),
    "JobAccepted": (
        {"job_id": "job-1", "queue_position": 0},
        {"job_id": "job-1"},
    ),
    "JobStatus": (
        {"job_id": "job-1", "state": "succeeded", "message": "done", "result": _result_wire()},
        {"job_id": "job-1", "state": "queued"},
    ),
    "ConfidenceReport": (_confidence_wire(True), _confidence_wire(False)),
    "CaResidue": (_ca_wire(), _ca_wire()),
    "StructurePayload": (_structure_wire(True), _structure_wire(False)),
    "FoldResult": (_result_wire(), _result_wire()),
    "GpuInfo": (_gpu_wire(True), _gpu_wire(False)),
    "Health": (
        {
            "protocol": "fh.fold/v1",
            "service": "svc",
            "version": "0.1.0",
            "capabilities": ["fold"],
            "models": ["rf3"],
            "gpus": [_gpu_wire(True)],
            "queue_depth": 2,
        },
        {
            "protocol": "fh.fold/v1",
            "service": "svc",
            "version": "0.1.0",
            "capabilities": ["fold"],
            "queue_depth": 0,
        },
    ),
    "ErrorBody": (
        {"error": "unknown_job", "detail": "no job x"},
        {"error": "unknown_job"},
    ),
}


def test_the_parser_reads_the_same_structs_the_contract_declares():
    assert set(STRUCTS) == set(PAYLOADS), (
        "src/contract.rs and this test disagree about which structs exist; "
        "a new wire type needs a Python model and a payload here"
    )


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_field_names_match_the_rust_struct(type_name):
    rust_names = {field.name for field in STRUCTS[type_name]}
    python_names = {f.name for f in dataclasses.fields(PYTHON_TYPES[type_name])}
    # AGENTS.md lets the Rust contract grow by additive fields that carry
    # #[serde(default)]. The Python model may lag behind exactly those. A
    # rust-only field without the default still breaks a client that reads
    # the struct as declared, so it fails here like a python-only field.
    undeclared = {
        field.name
        for field in STRUCTS[type_name]
        if field.name not in python_names and not field.has_default
    }
    assert not undeclared and not (python_names - rust_names), (
        f"{type_name}: rust-only fields without #[serde(default)] {undeclared}, "
        f"python-only fields {python_names - rust_names}"
    )


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_field_types_match_the_rust_types(type_name):
    hints = typing.get_type_hints(PYTHON_TYPES[type_name])
    for field in STRUCTS[type_name]:
        if field.name not in hints:
            continue  # an additive rust-only field the Python model may omit
        expected = _python_type(field.type, type_name)
        actual = hints[field.name]
        assert actual == expected, (
            f"{type_name}.{field.name}: the Rust type {field.type} transcribes as "
            f"{expected!r}, but the Python model annotates {actual!r}"
        )


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_integer_fields_enforce_the_rust_bit_width(type_name):
    full, _ = PAYLOADS[type_name]
    python_names = {f.name for f in dataclasses.fields(PYTHON_TYPES[type_name])}
    for field in STRUCTS[type_name]:
        if field.name not in python_names:
            continue  # an additive rust-only field the Python model may omit
        shape = _uint_shape(field.type)
        if shape is None:
            continue
        bits, wrap = shape
        accepted = dict(full)
        accepted[field.name] = wrap(2**bits - 1)
        PYTHON_TYPES[type_name].from_wire(accepted)
        rejected = dict(full)
        rejected[field.name] = wrap(2**bits)
        with pytest.raises(ValueError, match=field.name):
            PYTHON_TYPES[type_name].from_wire(rejected)


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_boolean_fields_reject_every_non_boolean(type_name):
    full, _ = PAYLOADS[type_name]
    python_names = {f.name for f in dataclasses.fields(PYTHON_TYPES[type_name])}
    for field in STRUCTS[type_name]:
        if field.name not in python_names:
            continue  # an additive rust-only field the Python model may omit
        if field.type != "bool":
            continue
        accepted = dict(full)
        accepted[field.name] = False
        PYTHON_TYPES[type_name].from_wire(accepted)
        for rejected_value in ("no", 1, None):
            rejected = dict(full)
            rejected[field.name] = rejected_value
            with pytest.raises(ValueError, match=field.name):
                PYTHON_TYPES[type_name].from_wire(rejected)


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_string_fields_reject_every_non_string(type_name):
    full, _ = PAYLOADS[type_name]
    python_names = {f.name for f in dataclasses.fields(PYTHON_TYPES[type_name])}
    for field in STRUCTS[type_name]:
        if field.name not in python_names:
            continue  # an additive rust-only field the Python model may omit
        shape = _str_shape(field.type)
        if shape is None:
            continue
        wrap, null_ok = shape
        accepted = dict(full)
        accepted[field.name] = wrap("probed")
        PYTHON_TYPES[type_name].from_wire(accepted)
        if null_ok:
            nulled = dict(full)
            nulled[field.name] = None
            PYTHON_TYPES[type_name].from_wire(nulled)
        for rejected_value in (5, None):
            if rejected_value is None and null_ok:
                continue
            rejected = dict(full)
            rejected[field.name] = wrap(rejected_value)
            with pytest.raises(ValueError, match=field.name):
                PYTHON_TYPES[type_name].from_wire(rejected)


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_a_full_payload_serialises_every_field(type_name):
    full, _ = PAYLOADS[type_name]
    obj = PYTHON_TYPES[type_name].from_wire(full)
    assert set(obj.to_wire()) == _transcribed_names(type_name)


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_a_minimal_payload_omits_exactly_the_skipped_fields(type_name):
    _, minimal = PAYLOADS[type_name]
    obj = PYTHON_TYPES[type_name].from_wire(minimal)
    skipped = {field.name for field in STRUCTS[type_name] if field.has_skip}
    assert set(obj.to_wire()) == _transcribed_names(type_name) - skipped


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_a_serde_default_field_may_be_absent_on_input(type_name):
    full, _ = PAYLOADS[type_name]
    defaults = [field.name for field in STRUCTS[type_name] if field.has_default]
    reduced = dict(full)
    for name in defaults:
        if name in reduced:
            del reduced[name]
    # Parses without raising; that is what #[serde(default)] promises.
    PYTHON_TYPES[type_name].from_wire(reduced)


@pytest.mark.parametrize("type_name", sorted(PAYLOADS))
def test_unknown_fields_are_tolerated_like_serde_without_deny(type_name):
    _, minimal = PAYLOADS[type_name]
    extended = dict(minimal)
    extended["a_field_rust_never_declared"] = 1
    PYTHON_TYPES[type_name].from_wire(extended)


def test_enum_variants_carry_the_snake_case_wire_spelling():
    assert set(ENUMS) == {"MoleculeKind", "JobState", "StructureFormat"}
    python_enums = {
        "MoleculeKind": wire.MoleculeKind,
        "JobState": wire.JobState,
        "StructureFormat": wire.StructureFormat,
    }
    for name, variants in ENUMS.items():
        python_values = [member.value for member in python_enums[name]]
        assert variants == python_values, f"{name}: rust {variants}, python {python_values}"


def test_protocol_and_routes_are_the_rust_constants():
    assert set(CONSTS) == {"PROTOCOL", "HEALTH", "FOLD", "JOB"}, (
        "src/contract.rs declares a string constant this test does not assert. "
        "Extend this test to bind it"
    )
    assert wire.PROTOCOL == CONSTS["PROTOCOL"] == "fh.fold/v1"
    assert wire.ROUTE_HEALTH == CONSTS["HEALTH"] == "/v1/health"
    assert wire.ROUTE_FOLD == CONSTS["FOLD"] == "/v1/fold"
    assert wire.route_job("abc123") == "/v1/fold/abc123"
    # routes::JOB is the axum pattern. route_job must fill its :job_id slot
    # with the same prefix the pattern declares.
    assert wire.route_job("abc123") == CONSTS["JOB"].replace(":job_id", "abc123")


def test_major_version_takes_the_segment_after_the_last_slash():
    # Mirrors contract::major_version, including the no-slash case, so the
    # client's major-version comparison behaves identically on both sides.
    assert wire.major_version("fh.fold/v1") == "v1"
    assert wire.major_version("fh.fold/v2") == "v2"
    assert wire.major_version("no-slash") == "no-slash"


def test_a_missing_required_field_names_it():
    _, minimal = PAYLOADS["JobSpec"]
    broken = dict(minimal)
    del broken["diffusion_samples"]
    with pytest.raises(ValueError, match="diffusion_samples"):
        wire.JobSpec.from_wire(broken)


def test_residue_enforces_the_one_character_wire_form_of_a_rust_char():
    # serde rejects any string that is not one character for a Rust char, so
    # from_wire rejects it too rather than letting the Rust client find it.
    for residue in ("Gln", ""):
        broken = dict(_ca_wire())
        broken["residue"] = residue
        with pytest.raises(ValueError, match="residue"):
            wire.CaResidue.from_wire(broken)
