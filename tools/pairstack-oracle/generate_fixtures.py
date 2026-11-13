#!/usr/bin/env python3
"""Generate reference tensors for the four RF3 Pairformer triangle operations.

The script imports the real foundry classes (TriangleMultiplication and
TriangleAttention) and runs them on CPU in fp32 with a fixed seed. It records
every weight, the shared input, each operation's output, and the internal
tensors that a reimplementation is most likely to get wrong.

Internal tensors are captured without reimplementing any operation body.
Submodule forward hooks record the inputs and outputs of every nn.Module the
operation owns. Scoped monkeypatches record the functional steps that no
module owns: torch.sigmoid, torch.einsum, opt_einsum.contract (imported into
the attention module as `einsum`) and torch.nn.functional.softmax. Each patch
calls the original function and stores a clone of what passed through it.

Output format: one file per tensor, holding raw float32 values in
little-endian byte order, row-major (C) order, with no header. manifest.json
names every file and records its shape and dtype.

Usage:
    .venv/bin/python generate_fixtures.py [--out fixtures] \
        [--foundry-src /tmp/foundry/models/rf3/src]
"""

import argparse
import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

FOUNDRY_COMMIT = "0d0b0ba44fab7ca76aa4b625c014ce5964c7d284"
DEFAULT_FOUNDRY_SRC = "/tmp/foundry/models/rf3/src"

SEED = 20261113
BATCH = 1
SEQ_LEN = 13
C_Z = 128
C_HIDDEN_MUL = 128
N_HEAD = 4
D_HEAD = 32

# Construction order feeds the global RNG that initialises the weights, so it
# is part of the recipe and must not change without a new seed record.
OP_NAMES = [
    "tri_mul_outgoing",
    "tri_mul_incoming",
    "tri_attn_start",
    "tri_attn_end",
]


def check_foundry(foundry_src: Path) -> str:
    repo = foundry_src
    while repo != repo.parent:
        if (repo / ".git").exists():
            break
        repo = repo.parent
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != FOUNDRY_COMMIT:
        raise SystemExit(
            f"foundry at {repo} is {head}, expected {FOUNDRY_COMMIT}. "
            "Check out the pinned commit or update FOUNDRY_COMMIT deliberately."
        )
    return head


@contextmanager
def patched(target, name, replacement):
    original = getattr(target, name)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        setattr(target, name, original)


class CallRecorder:
    """Drop-in replacement for a function that records calls, in call order."""

    def __init__(self, original, keep_args=True):
        self.original = original
        self.keep_args = keep_args
        self.calls = []

    def __call__(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        if self.keep_args:
            self.calls.append((args, kwargs, result))
        else:
            self.calls.append(((), kwargs, result))
        return result

    def only(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0]

    def at(self, i):
        return self.calls[i]


class HookRecorder:
    """Captures (input, output) of every named submodule via forward hooks."""

    def __init__(self, module, names):
        self.records = {}
        self.handles = []
        for name in names:
            sub = getattr(module, name)

            def hook(_mod, inputs, output, name=name):
                self.records[name] = (inputs[0].clone(), output.clone())

            self.handles.append(sub.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()


def capture_triangle_multiplication(module, pair):
    """Runs one TriangleMultiplication forward and names every tensor."""
    hooks = HookRecorder(
        module, ["norm_in", "p_in", "g_in", "norm_out", "p_out", "g_out"]
    )
    sigmoid = CallRecorder(torch.sigmoid, keep_args=False)
    einsum = CallRecorder(torch.einsum)
    try:
        with (
            patched(torch, "sigmoid", sigmoid),
            patched(torch, "einsum", einsum),
            torch.no_grad(),
        ):
            output = module(pair)
    finally:
        hooks.close()

    assert len(sigmoid.calls) == 3, "expected left, right and output gate sigmoids"
    einsum_args, _, einsum_raw = einsum.only()
    eq, operands = einsum_args[0], einsum_args[1:]
    t = {}
    t["input"] = pair
    t["norm_pair"] = hooks.records["norm_in"][1]
    t["p_combined"] = hooks.records["p_in"][1]
    t["g_combined"] = hooks.records["g_in"][1]
    t["left_gate"] = sigmoid.at(0)[2]
    t["right_gate"] = sigmoid.at(1)[2]
    t["left_gated"] = operands[0]
    t["right_gated_over_l"] = operands[1]
    t["einsum_raw"] = einsum_raw
    t["norm_out_applied"] = hooks.records["norm_out"][1]
    t["out_proj"] = hooks.records["p_out"][1]
    t["out_gate_logit"] = hooks.records["g_out"][1]
    t["out_gate"] = sigmoid.at(2)[2]
    t["output"] = output
    descriptions = {
        "input": "shared pair input (B,L,L,c_z), fp32",
        "norm_pair": "norm_in(input), LayerNorm eps=1e-5",
        "p_combined": "p_in(norm_pair), last dim is 2*d_hidden, left half then right half",
        "g_combined": "g_in(norm_pair), last dim is 2*d_hidden, left half then right half",
        "left_gate": "sigmoid of the left half of g_combined",
        "right_gate": "sigmoid of the right half of g_combined",
        "left_gated": "einsum left operand: left_gate * left half of p_combined",
        "right_gated_over_l": "einsum right operand: (right_gate * right half of p_combined) / L",
        "einsum_raw": f"torch.einsum('{eq}', left_gated, right_gated_over_l), before norm_out",
        "norm_out_applied": "norm_out(einsum_raw), LayerNorm eps=1e-5",
        "out_proj": "p_out(norm_out_applied)",
        "out_gate_logit": "g_out(norm_pair), the gate reads the normalised input",
        "out_gate": "sigmoid(out_gate_logit)",
        "output": "out_gate * out_proj, the value the block adds to the residual",
    }
    return t, descriptions


def capture_triangle_attention(module, pair):
    """Runs one TriangleAttention forward (vanilla path) and names every tensor."""
    from rf3.model.layers import attention as attention_module
    from torch.nn import functional

    hooks = HookRecorder(
        module, ["norm", "to_b", "to_q", "to_k", "to_v", "to_g", "to_out"]
    )
    sigmoid = CallRecorder(torch.sigmoid, keep_args=False)
    softmax = CallRecorder(functional.softmax)
    contract = CallRecorder(attention_module.einsum)
    try:
        with (
            patched(torch, "sigmoid", sigmoid),
            patched(functional, "softmax", softmax),
            patched(attention_module, "einsum", contract),
            torch.no_grad(),
        ):
            output = module(pair)
    finally:
        hooks.close()

    gate = sigmoid.only()[2]
    qk_args, _, attn_logits = contract.at(0)
    qk_eq, qk_operands = qk_args[0], qk_args[1:]
    av_args, _, attn_out_heads = contract.at(1)
    av_eq, av_operands = av_args[0], av_args[1:]
    softmax_args, softmax_kwargs, attn_weights = softmax.only()
    attn_with_bias = softmax_args[0]
    t = {}
    t["input"] = pair
    t["norm_pair"] = hooks.records["norm"][1]
    t["bias"] = hooks.records["to_b"][1]
    t["qkv_pair"] = hooks.records["to_q"][0]
    t["query_proj"] = hooks.records["to_q"][1]
    t["key_proj"] = hooks.records["to_k"][1]
    t["value_proj"] = hooks.records["to_v"][1]
    t["gate_logit"] = hooks.records["to_g"][1]
    t["gate"] = gate
    t["query_scaled"] = qk_operands[0]
    t["key_heads"] = qk_operands[1]
    t["attn_logits"] = attn_logits
    t["attn_with_bias"] = attn_with_bias
    t["attn_weights"] = attn_weights
    t["value_heads"] = av_operands[1]
    t["attn_out_heads"] = attn_out_heads
    t["gated_out"] = hooks.records["to_out"][0]
    t["output"] = output
    end = "ending-node" if not module.start_node else "starting-node"
    descriptions = {
        "input": "shared pair input (B,L,L,c_z), fp32",
        "norm_pair": "norm(input), LayerNorm eps=1e-5",
        "bias": f"to_b(norm_pair), shape (B,L,L,{N_HEAD}), computed before the {end} transpose",
        "qkv_pair": "input to to_q/to_k/to_v/to_g; equals norm_pair transposed on dims -2,-1 for ending-node, norm_pair itself for starting-node",
        "query_proj": "to_q(qkv_pair), shape (B,L,L,h*d)",
        "key_proj": "to_k(qkv_pair), shape (B,L,L,h*d)",
        "value_proj": "to_v(qkv_pair), shape (B,L,L,h*d)",
        "gate_logit": "to_g(qkv_pair), shape (B,L,L,h*d)",
        "gate": "sigmoid(gate_logit)",
        "query_scaled": "query_proj reshaped to (B,L,L,h,d) and multiplied by 1/sqrt(32)",
        "key_heads": "key_proj reshaped to (B,L,L,h,d)",
        "attn_logits": f"opt_einsum contract '{qk_eq}' of query_scaled and key_heads, shape (B,i,j,k,h)",
        "attn_with_bias": "attn_logits + bias.unsqueeze(1); the bias keeps its untransposed orientation",
        "attn_weights": f"softmax(attn_with_bias, dim={softmax_kwargs.get('dim', -2)}), shape (B,i,j,k,h)",
        "value_heads": "value_proj reshaped to (B,L,L,h,d)",
        "attn_out_heads": f"opt_einsum contract '{av_eq}' of attn_weights and value_heads, shape (B,i,j,h,d)",
        "gated_out": "gate * attn_out_heads reshaped to (B,L,L,h*d), transposed back for ending-node; the input to to_out",
        "output": "to_out(gated_out), the value the block adds to the residual",
    }
    return t, descriptions


class FixtureWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        (out_dir / "tensors").mkdir(parents=True, exist_ok=True)
        (out_dir / "weights").mkdir(parents=True, exist_ok=True)
        self.manifest_tensors = {}
        self.manifest_weights = {}

    def write_tensor(self, name, tensor, description):
        rel = f"tensors/{name}.f32"
        data = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        assert data.dtype.byteorder in ("=", "<", "|"), "expected little-endian host"
        (self.out_dir / rel).write_bytes(data.astype("<f4").tobytes())
        self.manifest_tensors[name] = {
            "path": rel,
            "shape": list(tensor.shape),
            "dtype": "float32",
            "description": description,
        }

    def write_weights(self, op_name, module):
        entries = {}
        for key, tensor in module.state_dict().items():
            rel = f"weights/{op_name}/{key}.f32"
            (self.out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            data = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
            (self.out_dir / rel).write_bytes(data.astype("<f4").tobytes())
            entries[key] = {
                "path": rel,
                "shape": list(tensor.shape),
                "dtype": "float32",
            }
        self.manifest_weights[op_name] = entries


def build_ops():
    from rf3.model.layers.attention import TriangleAttention, TriangleMultiplication

    torch.manual_seed(SEED)
    ops = {}
    ops["tri_mul_outgoing"] = TriangleMultiplication(
        d_pair=C_Z,
        d_hidden=C_HIDDEN_MUL,
        direction="outgoing",
        bias=True,
        use_cuequivariance=True,
    )
    ops["tri_mul_incoming"] = TriangleMultiplication(
        d_pair=C_Z,
        d_hidden=C_HIDDEN_MUL,
        direction="incoming",
        bias=True,
        use_cuequivariance=True,
    )
    ops["tri_attn_start"] = TriangleAttention(
        C_Z,
        n_head=N_HEAD,
        d_hidden=D_HEAD,
        start_node=True,
        use_cuequivariance=True,
    )
    ops["tri_attn_end"] = TriangleAttention(
        C_Z,
        n_head=N_HEAD,
        d_hidden=D_HEAD,
        start_node=False,
        use_cuequivariance=True,
    )
    for op in ops.values():
        op.eval()
    return ops


def build_input():
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randn(
        (BATCH, SEQ_LEN, SEQ_LEN, C_Z), generator=generator, dtype=torch.float32
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="fixtures", help="output directory")
    parser.add_argument("--foundry-src", default=DEFAULT_FOUNDRY_SRC)
    args = parser.parse_args()

    foundry_src = Path(args.foundry_src).resolve()
    commit = check_foundry(foundry_src)
    sys.path.insert(0, str(foundry_src))

    # Pin the CPU thread count so reduction order, and therefore the bytes,
    # stay identical on machines with different core counts.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    from rf3._core import SHOULD_USE_CUEQUIVARIANCE

    if SHOULD_USE_CUEQUIVARIANCE:
        raise SystemExit(
            "cuEquivariance is active; the oracle requires the vanilla CPU path"
        )
    if torch.cuda.is_available():
        raise SystemExit("CUDA visible; the oracle requires a CPU-only run")

    ops = build_ops()
    pair = build_input()

    writer = FixtureWriter(Path(args.out))
    for op_name in OP_NAMES:
        module = ops[op_name]
        writer.write_weights(op_name, module)
        if op_name.startswith("tri_mul"):
            tensors, descriptions = capture_triangle_multiplication(module, pair)
        else:
            tensors, descriptions = capture_triangle_attention(module, pair)
        for name, tensor in tensors.items():
            writer.write_tensor(f"{op_name}.{name}", tensor, descriptions[name])

    manifest = {
        "meta": {
            "generator": "tools/pairstack-oracle/generate_fixtures.py",
            "foundry_commit": commit,
            "foundry_src": str(foundry_src),
            "seed": SEED,
            "rng": {
                "weights": f"torch.manual_seed({SEED}) before building the four ops in the order {OP_NAMES}",
                "input": f"torch.Generator(device='cpu').manual_seed({SEED}) passed to torch.randn",
            },
            "input_shape": [BATCH, SEQ_LEN, SEQ_LEN, C_Z],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "encoding": {
                "dtype": "float32",
                "byte_order": "little-endian",
                "layout": "row-major (C order)",
                "header": "none; the file is exactly prod(shape) * 4 bytes",
            },
            "constants": {
                "c_z": C_Z,
                "triangle_multiplication_d_hidden": C_HIDDEN_MUL,
                "triangle_attention_n_head": N_HEAD,
                "triangle_attention_d_head": D_HEAD,
                "layer_norm_eps": 1e-5,
                "attention_scaling": "1/sqrt(32)",
                "triangle_multiplication_right_operand": "divided by L",
            },
        },
        "tensors": writer.manifest_tensors,
        "weights": writer.manifest_weights,
    }
    (writer.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entries = list(writer.manifest_tensors.values())
    entries += [e for op in writer.manifest_weights.values() for e in op.values()]
    total = sum((writer.out_dir / e["path"]).stat().st_size for e in entries)
    print(
        f"wrote {len(writer.manifest_tensors)} tensors and "
        f"{sum(len(v) for v in writer.manifest_weights.values())} weight tensors "
        f"({total} bytes) to {writer.out_dir}"
    )


if __name__ == "__main__":
    main()
