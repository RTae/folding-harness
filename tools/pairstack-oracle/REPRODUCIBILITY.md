# Reproducing the pair-stack oracle fixtures

`generate_fixtures.py` imports the real RF3 triangle operations from a pinned
foundry checkout, injects the real trained weights of one Pairformer block
from the published RF3 checkpoint, runs the four operations on CPU in fp32
with a fixed-seed input, and writes every weight, the shared input, each
output and the error-prone internal tensors as headerless little-endian
float32 files. `fixtures/manifest.json` names each file and records its
shape, dtype and meaning. A Rust reader needs no Python and no network, only
the files under `fixtures/`.

The weight values under `fixtures/weights/` are derived from foundry's RF3
checkpoint and are redistributed under foundry's BSD-3 licence. Read
`WEIGHTS-NOTICE.md` before copying them anywhere. Never commit the checkpoint
itself to this repository.

## Pinned upstream

- Repository: https://github.com/RosettaCommons/foundry
- Commit: `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`
- Expected location: `/tmp/foundry` (override with `--foundry-src`)

The script runs `git rev-parse HEAD` on the checkout and refuses to generate
when the commit differs from the pin. Clone it with:

```bash
git clone https://github.com/RosettaCommons/foundry /tmp/foundry
git -C /tmp/foundry checkout 0d0b0ba44fab7ca76aa4b625c014ce5964c7d284
```

The script imports `rf3.model.layers.attention` and `rf3._core.compat`,
foundry's checkpoint loader, plus their dependency closure. It never imports
the trainer or the model assembly, so the full foundry dependency list does
not apply.

## Checkpoint

- File: `rf3_foundry_01_24_latest_remapped.ckpt`
- URL: https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt,
  the `rf3` entry of `REGISTERED_CHECKPOINTS` in foundry's
  `models/rf3/src/rf3/checkpoints.py`
- Size: 3,038,876,446 bytes
- SHA-256:
  `364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0`
- Injected block: 24 of the 48 Pairformer blocks, the middle of the stack,
  selected with `--block` and counted from zero

Download it to a path outside the repository. `~/.foundry/checkpoints` is
foundry's own default location. The server occasionally drops the connection
mid-transfer, so resume with `-C -` until the size matches:

```bash
mkdir -p ~/.foundry/checkpoints
curl -fL -C - --retry 10 --retry-delay 5 --retry-all-errors \
    -o ~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt \
    https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt
shasum -a 256 ~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt
```

The generator hashes the file and refuses to run when the digest differs from
the `CHECKPOINT_SHA256` constant in the script. The weights sit under the
`model` key of the checkpoint dictionary, and every parameter appears twice
inside it, because the checkpoint stores the net wrapped in foundry's EMA
module: raw weights prefixed `model.` and EMA shadow weights prefixed
`shadow.`. The generator injects the `shadow.` weights, because `EMA.forward`
dispatches to the shadow model at eval, so the shadow is what foundry's own
inference runs. After that prefix is stripped, the block's tensors sit under
the key prefix `recycler.pairformer_stack.<block>.`

## Environment

Created with [uv](https://docs.astral.sh/uv/) on macOS arm64:

```bash
cd tools/pairstack-oracle
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    torch einops jaxtyping beartype environs opt_einsum numpy omegaconf
```

This resolved to Python 3.12.12 and the following packages. `omegaconf` is
needed by foundry's checkpoint loader. Entries below the line are transitive
dependencies of the eight named packages.

```text
torch==2.15.0            einops==0.8.2           jaxtyping==0.3.11
beartype==0.23.1         environs==15.2.0        opt-einsum==3.4.0
numpy==2.5.4             omegaconf==2.4.0
filelock==4.2.10         fsspec==2026.9.0        jinja2==3.1.6
markupsafe==3.0.4        marshmallow==4.4.0      mpmath==1.3.0
networkx==3.7            python-dotenv==1.2.4    pyyaml==6.0.3
setuptools==84.2.1       sympy==1.14.0           typing-extensions==4.16.0
wadler-lindig==0.1.8
```

foundry's `pyproject.toml` asks for `torch>=2.11.0` and Python >=3.12. Both
hold here.

## Generate

```bash
cd tools/pairstack-oracle
.venv/bin/python generate_fixtures.py \
    --checkpoint ~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt \
    --out fixtures
```

## Determinism recipe

- Weights: real trained values from the SHA-256-pinned checkpoint, Pairformer
  block 24, injected into the standalone operations with
  `load_state_dict(strict=True)`. `torch.manual_seed(20261113)` still runs
  before the four operations are constructed in the order `tri_mul_outgoing`,
  `tri_mul_incoming`, `tri_attn_start`, `tri_attn_end`, but every parameter is
  overwritten, so the dumped weights do not depend on the seed.
- Input: `torch.Generator(device="cpu").manual_seed(20261113)` passed
  explicitly to `torch.randn((1, 13, 13, 128))`.
- `torch.set_num_threads(1)` pins the reduction order across machines.
- `torch.use_deterministic_algorithms(True)` fails loudly on any op without a
  deterministic CPU implementation. None occurred.
- The script exits before generating anything when a GPU is visible or when
  cuEquivariance is active.
- The manifest carries no timestamp. It records the checkpoint digest, the
  block index and the state-dict key prefix under `meta.checkpoint`, so
  identical runs produce identical bytes.

Verified: two consecutive runs into different directories compared equal with
`diff -r`. A third run reproduced the same bytes in `fixtures/`.

## Verifying a regeneration

```bash
.venv/bin/python generate_fixtures.py \
    --checkpoint ~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt \
    --out /tmp/fixtures-check
diff -r fixtures /tmp/fixtures-check && echo identical
```

## Verifying the fixtures are readable without Python

```bash
cd reader && cargo run --release -- ../fixtures
```

The reader checks every manifest entry against its file size and prints the
inventory. It exits nonzero on any mismatch.

## What the fixtures contain

- 64 activation tensors: 14 per triangle multiplication, 18 per triangle
  attention. Names carry the operation prefix, for example
  `tri_attn_end.attn_with_bias`. The description of each tensor is in the
  manifest.
- 36 weight tensors, saved under the exact `state_dict` key of a standalone
  instantiation and holding the trained values of Pairformer block 24:
  `norm_in.weight`, `norm_in.bias`, `p_in.weight`,
  `g_in.weight`, `norm_out.weight`, `norm_out.bias`, `p_out.weight`,
  `g_out.weight` for each multiplication, and `norm.weight`, `norm.bias`,
  `to_q.weight`, `to_k.weight`, `to_v.weight`, `to_b.weight`, `to_g.weight`,
  `to_g.bias`, `to_out.weight`, `to_out.bias` for each attention. A default
  random initialisation is not acceptable here: it zeroes `to_out` and
  `to_g.weight` and leaves every LayerNorm affine at its identity, which made
  the earlier fixture set blind to any attention-path or affine-transform
  defect.
- Total size: 6,862,080 bytes in 100 tensor files plus `manifest.json`.
  Regenerate the tensor-file count with `find fixtures -name '*.f32' | wc -l`,
  which prints 100. Counting every file with `find fixtures -type f | wc -l`
  prints 101 because it includes `manifest.json`.

## Format

Each `.f32` file holds exactly `prod(shape) * 4` bytes: float32,
little-endian, row-major (C order), no header, no padding. `manifest.json`
records `shape` and `dtype` per entry and the constants the operations use
under `meta.constants`.
