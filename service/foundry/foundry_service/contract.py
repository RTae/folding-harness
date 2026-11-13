"""The `fh.fold/v1` wire types, mirroring `src/contract.rs` field for field.

The Rust file is the definition; this module is a transcription of it. Every
field name, every enum's snake_case wire spelling, and every serde default or
`skip_serializing_if` rule has an exact counterpart here, and
`tests/test_contract_conformance.py` re-reads `src/contract.rs` and fails when
the two drift apart.

Optional fields follow the Rust serialisation rules rather than JSON norms:
a field marked `skip_serializing_if = "Option::is_none"` is omitted from the
wire dict when it is None, and one marked `skip_serializing_if =
"Vec::is_empty"` is omitted when the list is empty. A field marked
`#[serde(default)]` may be absent on input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

PROTOCOL = "fh.fold/v1"

ROUTE_HEALTH = "/v1/health"
ROUTE_FOLD = "/v1/fold"


def route_job(job_id: str) -> str:
    return f"/v1/fold/{job_id}"


def major_version(protocol: str) -> str:
    """The part after the last '/', e.g. 'v1'.

    Mirrors `contract::major_version`, which is `rsplit('/').next()`: a
    string with no '/' is its own last segment.
    """
    return protocol.rsplit("/", 1)[-1]


class MoleculeKind(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    LIGAND = "ligand"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED)


class StructureFormat(str, Enum):
    CIF = "cif"
    PDB = "pdb"
    CA_TRACE = "ca_trace"


@dataclass
class ChainSpec:
    label: str
    sequence: str
    kind: MoleculeKind = MoleculeKind.PROTEIN
    designed: bool = False

    def to_wire(self) -> dict:
        return {
            "label": self.label,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "designed": self.designed,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> ChainSpec:
        label = _require_str(wire, "label", "chains")
        sequence = _require_str(wire, "sequence", "chains")
        kind_value = wire.get("kind", MoleculeKind.PROTEIN.value)
        try:
            kind = MoleculeKind(kind_value)
        except ValueError:
            raise ValueError(
                f"chains[{label}].kind is {kind_value!r}, expected one of "
                f"{[k.value for k in MoleculeKind]}"
            ) from None
        return cls(
            label=label, sequence=sequence, kind=kind, designed=bool(wire.get("designed", False))
        )


@dataclass
class JobSpec:
    name: str
    model: str
    chains: list[ChainSpec]
    seeds: list[int]
    diffusion_samples: int
    diffusion_steps: int
    use_msa: bool
    need_atom_confidence: bool

    def to_wire(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "chains": [chain.to_wire() for chain in self.chains],
            "seeds": list(self.seeds),
            "diffusion_samples": self.diffusion_samples,
            "diffusion_steps": self.diffusion_steps,
            "use_msa": self.use_msa,
            "need_atom_confidence": self.need_atom_confidence,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> JobSpec:
        chains_wire = _require(wire, "chains", "job")
        if not isinstance(chains_wire, list) or not chains_wire:
            raise ValueError("job.chains must be a non-empty list of chain objects")
        return cls(
            name=_require_str(wire, "name", "job"),
            model=_require_str(wire, "model", "job"),
            chains=[ChainSpec.from_wire(chain) for chain in chains_wire],
            seeds=[int(seed) for seed in _require(wire, "seeds", "job")],
            diffusion_samples=int(_require(wire, "diffusion_samples", "job")),
            diffusion_steps=int(_require(wire, "diffusion_steps", "job")),
            use_msa=bool(_require(wire, "use_msa", "job")),
            need_atom_confidence=bool(_require(wire, "need_atom_confidence", "job")),
        )


@dataclass
class FoldRequest:
    protocol: str
    job: JobSpec

    def to_wire(self) -> dict:
        return {"protocol": self.protocol, "job": self.job.to_wire()}

    @classmethod
    def from_wire(cls, wire: dict) -> FoldRequest:
        return cls(
            protocol=_require_str(wire, "protocol", "request"),
            job=JobSpec.from_wire(_require(wire, "job", "request")),
        )


@dataclass
class JobAccepted:
    job_id: str
    queue_position: int | None = None

    def to_wire(self) -> dict:
        wire: dict = {"job_id": self.job_id}
        if self.queue_position is not None:
            wire["queue_position"] = self.queue_position
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> JobAccepted:
        position = wire.get("queue_position")
        return cls(
            job_id=_require_str(wire, "job_id", "job_accepted"),
            queue_position=None if position is None else int(position),
        )


@dataclass
class ConfidenceReport:
    iptm: float
    ptm: float
    plddt_mean: float
    interface_plddt: float
    pae_interface: float
    interface_contacts: int
    clashes: int
    hydrophobic_patch: float
    net_charge: float
    atom_confidence: bool
    plddt: list[float] = field(default_factory=list)

    def to_wire(self) -> dict:
        wire = {
            "iptm": self.iptm,
            "ptm": self.ptm,
            "plddt_mean": self.plddt_mean,
            "interface_plddt": self.interface_plddt,
            "pae_interface": self.pae_interface,
            "interface_contacts": self.interface_contacts,
            "clashes": self.clashes,
            "hydrophobic_patch": self.hydrophobic_patch,
            "net_charge": self.net_charge,
            "atom_confidence": self.atom_confidence,
        }
        if self.plddt:
            wire["plddt"] = list(self.plddt)
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> ConfidenceReport:
        where = "confidence"
        return cls(
            iptm=float(_require(wire, "iptm", where)),
            ptm=float(_require(wire, "ptm", where)),
            plddt_mean=float(_require(wire, "plddt_mean", where)),
            interface_plddt=float(_require(wire, "interface_plddt", where)),
            pae_interface=float(_require(wire, "pae_interface", where)),
            interface_contacts=int(_require(wire, "interface_contacts", where)),
            clashes=int(_require(wire, "clashes", where)),
            hydrophobic_patch=float(_require(wire, "hydrophobic_patch", where)),
            net_charge=float(_require(wire, "net_charge", where)),
            atom_confidence=bool(_require(wire, "atom_confidence", where)),
            plddt=[float(value) for value in wire.get("plddt", [])],
        )


@dataclass
class CaResidue:
    chain_index: int
    index: int
    residue: str
    xyz: tuple[float, float, float]
    plddt: float
    designed: bool

    def to_wire(self) -> dict:
        return {
            "chain_index": self.chain_index,
            "index": self.index,
            "residue": self.residue,
            "xyz": list(self.xyz),
            "plddt": self.plddt,
            "designed": self.designed,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> CaResidue:
        where = "ca_trace entry"
        xyz = _require(wire, "xyz", where)
        if len(xyz) != 3:
            raise ValueError(f"ca_trace entry xyz must hold 3 numbers, got {len(xyz)}")
        return cls(
            chain_index=int(_require(wire, "chain_index", where)),
            index=int(_require(wire, "index", where)),
            residue=_require_str(wire, "residue", where),
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            plddt=float(_require(wire, "plddt", where)),
            designed=bool(_require(wire, "designed", where)),
        )


@dataclass
class StructurePayload:
    format: StructureFormat
    inline: str | None = None
    artifact_url: str | None = None
    ca_trace: list[CaResidue] = field(default_factory=list)

    def to_wire(self) -> dict:
        wire: dict = {"format": self.format.value}
        if self.inline is not None:
            wire["inline"] = self.inline
        if self.artifact_url is not None:
            wire["artifact_url"] = self.artifact_url
        if self.ca_trace:
            wire["ca_trace"] = [residue.to_wire() for residue in self.ca_trace]
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> StructurePayload:
        where = "structure"
        format_value = _require_str(wire, "format", where)
        try:
            payload_format = StructureFormat(format_value)
        except ValueError:
            raise ValueError(
                f"structure.format is {format_value!r}, expected one of "
                f"{[f.value for f in StructureFormat]}"
            ) from None
        return cls(
            format=payload_format,
            inline=wire.get("inline"),
            artifact_url=wire.get("artifact_url"),
            ca_trace=[CaResidue.from_wire(entry) for entry in wire.get("ca_trace", [])],
        )


@dataclass
class FoldResult:
    job_id: str
    model: str
    confidence: ConfidenceReport
    structure: StructurePayload
    elapsed_ms: int

    def to_wire(self) -> dict:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "confidence": self.confidence.to_wire(),
            "structure": self.structure.to_wire(),
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> FoldResult:
        where = "result"
        return cls(
            job_id=_require_str(wire, "job_id", where),
            model=_require_str(wire, "model", where),
            confidence=ConfidenceReport.from_wire(_require(wire, "confidence", where)),
            structure=StructurePayload.from_wire(_require(wire, "structure", where)),
            elapsed_ms=int(_require(wire, "elapsed_ms", where)),
        )


@dataclass
class JobStatus:
    job_id: str
    state: JobState
    message: str | None = None
    result: FoldResult | None = None

    def to_wire(self) -> dict:
        wire: dict = {"job_id": self.job_id, "state": self.state.value}
        if self.message is not None:
            wire["message"] = self.message
        if self.result is not None:
            wire["result"] = self.result.to_wire()
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> JobStatus:
        where = "job status"
        state_value = _require_str(wire, "state", where)
        try:
            state = JobState(state_value)
        except ValueError:
            raise ValueError(
                f"state is {state_value!r}, expected one of {[s.value for s in JobState]}"
            ) from None
        result_wire = wire.get("result")
        return cls(
            job_id=_require_str(wire, "job_id", where),
            state=state,
            message=wire.get("message"),
            result=None if result_wire is None else FoldResult.from_wire(result_wire),
        )


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int
    leased_by: str | None = None

    def to_wire(self) -> dict:
        wire = {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
        }
        if self.leased_by is not None:
            wire["leased_by"] = self.leased_by
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> GpuInfo:
        where = "gpus entry"
        return cls(
            index=int(_require(wire, "index", where)),
            name=_require_str(wire, "name", where),
            memory_total_mb=int(_require(wire, "memory_total_mb", where)),
            leased_by=wire.get("leased_by"),
        )


@dataclass
class Health:
    protocol: str
    service: str
    version: str
    capabilities: list[str]
    queue_depth: int
    models: list[str] = field(default_factory=list)
    gpus: list[GpuInfo] = field(default_factory=list)

    def to_wire(self) -> dict:
        wire = {
            "protocol": self.protocol,
            "service": self.service,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "queue_depth": self.queue_depth,
        }
        if self.models:
            wire["models"] = list(self.models)
        if self.gpus:
            wire["gpus"] = [gpu.to_wire() for gpu in self.gpus]
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> Health:
        where = "health"
        return cls(
            protocol=_require_str(wire, "protocol", where),
            service=_require_str(wire, "service", where),
            version=_require_str(wire, "version", where),
            capabilities=[str(c) for c in _require(wire, "capabilities", where)],
            queue_depth=int(_require(wire, "queue_depth", where)),
            models=[str(m) for m in wire.get("models", [])],
            gpus=[GpuInfo.from_wire(g) for g in wire.get("gpus", [])],
        )


@dataclass
class ErrorBody:
    error: str
    detail: str | None = None

    def to_wire(self) -> dict:
        wire: dict = {"error": self.error}
        if self.detail is not None:
            wire["detail"] = self.detail
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> ErrorBody:
        return cls(error=_require_str(wire, "error", "error body"), detail=wire.get("detail"))


def _require(wire: dict, key: str, where: str):
    if key not in wire:
        raise ValueError(f"{where}.{key} is required and missing from the request")
    return wire[key]


def _require_str(wire: dict, key: str, where: str) -> str:
    value = _require(wire, key, where)
    if not isinstance(value, str):
        raise ValueError(f"{where}.{key} must be a string, got {type(value).__name__}")
    return value
