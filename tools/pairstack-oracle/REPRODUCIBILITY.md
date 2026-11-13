# Reproducing the pair-stack oracle fixtures

`generate_fixtures.py` imports the real RF3 triangle operations from a pinned
foundry checkout, runs them on CPU in fp32 under a fixed seed, and writes every
weight, the shared input, each output and the error-prone internal tensors as
headerless little-endian float32 files. `fixtures/manifest.json` names each
file and records its shape, dtype and meaning. A Rust reader needs no Python
and no network, only the files under `fixtures/`.

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

The script only imports `rf3.model.layers.attention` and its dependency
closure. It never imports the trainer, the model assembly or any checkpoint
loader, so the full foundry dependency list does not apply.

## Environment

Created with [uv](https://docs.astral.sh/uv/) on macOS arm64:

```bash
cd tools/pairstack-oracle
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    torch einops jaxtyping beartype environs opt_einsum numpy
```

This resolved to Python 3.12.12 and the following packages. Entries below the
line are transitive dependencies of the seven named packages.

```text
torch==2.15.0            einops==0.8.2           jaxtyping==0.3.11
beartype==0.23.1         environs==15.2.0        opt-einsum==3.4.0
numpy==2.5.4
filelock==4.2.10         fsspec==2026.9.0        jinja2==3.1.6
markupsafe==3.0.4        marshmallow==4.4.0      mpmath==1.3.0
networkx==3.7            python-dotenv==1.2.4    setuptools==84.2.1
sympy==1.14.0            typing-extensions==4.16.0  wadler-lindig==0.1.8
```

foundry's `pyproject.toml` asks for `torch>=2.11.0` and Python >=3.12. Both
hold here.

## Generate

```bash
cd tools/pairstack-oracle
.venv/bin/python generate_fixtures.py --out fixtures
```

## Determinism recipe

- Weight init: `torch.manual_seed(20261113)` runs once, then the four
  operations are constructed in the order `tri_mul_outgoing`,
  `tri_mul_incoming`, `tri_attn_start`, `tri_attn_end`. The construction order
  consumes the global RNG, so it is part of the recipe.
- Input: `torch.Generator(device="cpu").manual_seed(20261113)` passed
  explicitly to `torch.randn((1, 13, 13, 128))`.
- `torch.set_num_threads(1)` pins the reduction order across machines.
- `torch.use_deterministic_algorithms(True)` fails loudly on any op without a
  deterministic CPU implementation. None occurred.
- The manifest carries no timestamp, so identical runs produce identical
  bytes.

Verified: two consecutive runs into different directories compared equal with
`diff -r`. A third run reproduced the same bytes in `fixtures/`.

## Verifying a regeneration

```bash
.venv/bin/python generate_fixtures.py --out /tmp/fixtures-check
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
  instantiation: `norm_in.weight`, `norm_in.bias`, `p_in.weight`,
  `g_in.weight`, `norm_out.weight`, `norm_out.bias`, `p_out.weight`,
  `g_out.weight` for each multiplication, and `norm.weight`, `norm.bias`,
  `to_q.weight`, `to_k.weight`, `to_v.weight`, `to_b.weight`, `to_g.weight`,
  `to_g.bias`, `to_out.weight`, `to_out.bias` for each attention.
- Total size: 6,862,080 bytes in 100 tensor files plus `manifest.json`.
  Regenerate the count with `find fixtures -type f | wc -l`.

## Format

Each `.f32` file holds exactly `prod(shape) * 4` bytes: float32,
little-endian, row-major (C order), no header, no padding. `manifest.json`
records `shape` and `dtype` per entry and the constants the operations use
under `meta.constants`.
