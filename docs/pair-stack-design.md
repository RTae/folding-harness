# Pair-stack design

A common interface for the pair-representation operations shared by the
AlphaFold 3 model family, so one Rust implementation can serve several models
and several backends.

Status: design, not code. Everything below is read from source at pinned
commits, and each claim names where it came from. Two things in this document
are load-bearing enough that getting them wrong produces plausible output and
no error, and they are marked.

## Evidence base

- `RosettaCommons/foundry` at `0d0b0ba44fab7ca76aa4b625c014ce5964c7d284`, read
  under `models/rf3/src/rf3/`. BSD-3, and its `LICENSE.md` covers code and
  weights in terms.
- `NVIDIA-BioNeMo/BioNeMo-Inference-Runtime` at `8c18d7f`, read under
  `bionemo_ir/`. Apache-2.0 for the code read here, with the exceptions named
  under Licences below.

Foundry is the numerical oracle. BioNeMo is the design reference. They are not
interchangeable, and the reason is under CPU feasibility below.

## The operations

Four, and two independent codebases agree on their order inside a block.

Foundry `PairformerBlock.forward`, at
`models/rf3/src/rf3/model/layers/pairformer_layers.py:405-418`:

1. `Z += tri_mul_outgoing(Z)`
2. `Z += tri_mul_incoming(Z)`
3. `Z += tri_attn_start(Z)`
4. `Z += tri_attn_end(Z)`
5. `Z += z_transition(Z)`
6. when a single representation is present, `S += attention_pair_bias(S, Z)`
   then `S += s_transition(S)`

BioNeMo `PairformerModule._transform_z`, at
`bionemo_ir/_torch/layers/pairformer.py:175-216`, runs the same five steps in
the same order and then the same two single-representation steps. Agreement
between two independent implementations is the strongest evidence available
that this order is the family's order and not one lab's choice.

Triangle multiplication, from foundry
`models/rf3/src/rf3/model/layers/attention.py:289-328`:

```
pair_norm  = LayerNorm(pair)
left, right = split(p_in(pair_norm))
left_gate, right_gate = split(sigmoid(g_in(pair_norm)))
left, right = left_gate * left, right_gate * right
outgoing: out = einsum("bikd,bjkd->bijd", left, right / L)
incoming: out = einsum("bkid,bkjd->bijd", left, right / L)
out = sigmoid(g_out(pair_norm)) * p_out(LayerNorm(out))
```

Triangle attention, from foundry `attention.py:82-111` and `179-200`:

```
pair = LayerNorm(pair)
bias = to_b(pair)                     # before any transpose
if ending: pair = transpose(pair)
gate  = sigmoid(to_g(pair))
q, k, v = to_q(pair), to_k(pair), to_v(pair)      # no bias terms
q = q * (1 / sqrt(head_dim))
attn = einsum("bijhd,bikhd->bijkh", q, k) + bias
attn = softmax(attn, over k)
out  = gate * einsum("bijkh,bikhd->bijhd", attn, v)
out  = to_out(out)
if ending: out = transpose(out)
```

The outgoing and incoming multiplications differ only in the einsum
subscripts. The starting and ending attentions differ only in where a
transpose happens. That is the whole of the directional variation, and it is
why one parameterised implementation serves both.

## The variation axes

BioNeMo carries model differences as constructor flags on one shared class
rather than as a subclass per model, at
`bionemo_ir/_torch/layers/triangle_nodes.py:120-136` and `326-343`. That is the
right shape and this design copies it. The axes, with what each model needs:

- **Ending-node bias orientation.** Load-bearing. See the trap below.
- **Mean normalisation.** Foundry divides the right operand by the sequence
  length `L` at `attention.py:314` and `316`. BioNeMo calls the same thing
  `trimul_mean_normalization`. One flag, two names, same arithmetic.
- **Bias presence.** Boltz-1 puts no biases on its triangle operations, at
  `boltz1/config.py:61-83`. Foundry's RF3 has biases on the layer norms and on
  the input and output projections. Per-projection flags, not one flag.
- **Output projection precision.** BioNeMo's `trimul_high_precision` runs the
  output projections in fp32, at `triangle_nodes.py:397-401`.
- **Layer norm on the bias projection.** BioNeMo's `norm_bias`.
- **Initial norm on attention.** BioNeMo's `attention_initial_norm`, which
  Boltz-1 sets and OpenFold3 also sets.
- **Dimensions and counts.** Pair width, single width, head count, head width,
  block count.
- **Whether an MSA stack runs alongside.** AlphaFold2 and OpenFold2 wrap the
  four operations in an Evoformer with an MSA stack. The AlphaFold3 family
  runs the pair stack alone and puts the MSA module before it. Foundry's RF3
  runs its MSA module before the pairformer stack and deliberately drops that
  module's residual, at `RF3_structure.py:296-298`.

The dimensions converge more than expected. Foundry's RF3 uses pair width 128,
single width 384, 48 blocks, and 4 triangle-attention heads of width 32, at
`configs/model/components/rf3_net.yaml:48-55`. BioNeMo reports the same pair
width, the same 4 heads of 32, and 48 blocks for Boltz-1 and for OpenFold3, and
48 blocks for Protenix-v2. Boltz-2 uses 64. So one shape covers most of the
family and the block count is the main thing that moves.

## The trap

**This is the first of two load-bearing details.** Foundry and BioNeMo place
the ending-node bias projection at different points, and BioNeMo's flag name
reads as the opposite of what it does.

Foundry, at `attention.py:85-88`, computes the bias from the untransposed
normalised pair and then transposes the pair for the query, key, value and
gate. Its own comment says forward transposes the normalised pair and not the
bias.

BioNeMo, at `triangle_nodes.py:252-263`, transposes `x` before the norm and the
bias projection for the ending node, so its default computes the bias from the
transposed pair. Setting `transposed_bias=True` transposes the bias back, which
reproduces foundry's behaviour. The flag being true means the bias ends up
untransposed.

BioNeMo's own RF3 onboarding example sets it true, at
`examples/module_onboarding/rf3_pairformer/integration/config.py:53`, with the
comment "RF3's ending node projects its triangle bias before transposing the
pair", which agrees with foundry's source. Its config comment at
`configs/modules.py:41-43` says OpenFold-3 v0.5.0 builds the ending node's bias
from the transposed pair while AlphaFold-2, Boltz and Protenix do not, which is
a statement about a different model and a different version.

So the two codebases agree about what RF3 does. They disagree about which
default expresses it, and the flag name points the wrong way. An implementation
built from BioNeMo's default while targeting RF3 would produce plausible
structures and no error.

The interface should therefore not carry a flag named for the transpose. It
should carry the fact: which representation the ending node's bias is projected
from, named as `bias_from_untransposed_pair` or equivalent, so the value states
the behaviour rather than a transformation whose direction depends on where the
implementation happens to transpose.

**The second load-bearing detail** is in triangle multiplication. The output
gate is `sigmoid(g_out(pair_norm))`, taken from the normalised input rather than
the raw pair, at `attention.py:325`. And only the right operand is divided by
`L`, at `attention.py:314` and `316`, not both. Both are single-token errors
that shift magnitudes quietly.

## The backend boundary

BioNeMo's `AttentionBackend`, at
`bionemo_ir/_torch/attention_backend/interface.py:53-95`, is generic over a
metadata type and takes the query, key, value, a list of biases and that
metadata. The bias list is where the real contract lives: dense backends read
the first entry as an additive mask and the second as the triangle bias, while
row-length backends read the first entry as per-batch integer row lengths. One
signature carries both encodings.

Six backends register behind it, and a heuristic picks between them at first
call per shape, using a threshold of 23 million on a product of batch, pair
dimension, squared tokens and heads, at
`bionemo_ir/_torch/attention_backend/heuristic.py:40`.

**Do not copy the runtime router.** Resolve the backend once, at construction,
from explicit configuration, so a CPU build and a GPU build each take a
statically known path. A backend chosen at runtime by shape means the code path
a test exercised is not necessarily the path production takes, and it makes a
numerical mismatch hard to attribute. Copy the interface and the metadata
parameter. Leave the heuristic behind.

For Rust this means one trait with an associated metadata type, and a config
value that names the implementation. Two implementations to start: a CPU
reference and, later, a tile-based GPU one.

## Numerical validation

BioNeMo's own tolerances, from its tests, are the ladder to copy.

- Two implementations of the same operation must agree bit-exactly, `atol=0`
  and `rtol=0`, at `tests/_torch/test_triangle_nodes.py:459`, `480` and `759`.
  This is the check for a fused path against an unfused one.
- Against an fp32 reference, triangle attention allows `atol=1e-2` and
  `rtol=1e-3`, at `tests/_torch/test_triangle_attention.py:112`, and triangle
  multiplication allows `atol=1e-3` and `rtol=1e-4`, at
  `tests/_torch/test_triangle_nodes.py:211`.
- A reduced-precision path is not held to a fixed tolerance. It is held to the
  reference's own error, at `tests/_torch/test_triangle_nodes.py:150-158`: the
  maximum deviation must be within 1.5 times the reference's maximum deviation
  plus `1e-3`, and the mean deviation within the reference's mean plus `0.05`.
- Two runs of the same model must be bit-identical, at
  `tests/_torch/model_forwards/test_model_forward_eager_determinism.py:325`.
- Downstream, agreement is measured as lDDT and DockQ against ground truth, and
  a gap between two sides is treated as a finding rather than as noise.

The CPU reference in Rust is held to the bit-exact standard against the fp32
foundry path, not to the reduced-precision ladder, because the oracle calls the
operations directly in fp32 with no trainer and no autocast.

## CPU feasibility

**Foundry is the oracle and BioNeMo is not.** This is the reason the two
repositories play different roles.

Foundry's `SHOULD_USE_CUEQUIVARIANCE` is false without CUDA, at
`models/rf3/src/rf3/_core/__init__.py:31-47`, so on a CPU both triangle classes
route to `_forward_vanilla`, which is plain PyTorch with no compiled kernel. The
whole pair stack is mirrorable on a CPU in fp32.

Precision needs care, because the obvious reading of the config code is wrong.
Foundry forces `32-true` only when `FOUNDRY_DEVICE` explicitly asks for cpu or
mps, at `_core/utils/ddp.py:44-48`, or when MPS is auto-detected, at `:57-61`.
The no-accelerator fallback sets the accelerator to cpu but never overrides
precision, at `:62-68`, so the configured `bf16-mixed` default survives. Running
foundry's inference engine on a machine with no accelerator therefore does not
automatically give fp32. The oracle sidesteps this entirely: it instantiates the
four operations directly and calls them in fp32 with no trainer and no autocast,
so no precision policy applies.

BioNeMo's triangle multiplication has no CPU path. Its `_forward_impl` calls a
fused `layer_norm_transpose` unconditionally, at `triangle_nodes.py:682` and
`708`, and that call raises without a CUDA input, at
`dsl_kernels/triton/fused_layer_norm_transpose.py:555-556`. Its triangle
attention does have a CPU path, through the VANILLA backend.

So one fused call with no fallback makes an otherwise portable operation
CUDA-mandatory. Every fused path in this design keeps a named eager path beside
it, and the eager path is what the tests run.

## Licences

From BioNeMo's Apache-2.0 code, safe to read and copy: `triangle_nodes.py`,
`attention_backend/interface.py`, the pairformer and evoformer wrappers, and the
VANILLA and SDPA backends.

Not Apache, per `THIRD_PARTY_NOTICES.md`: the `cuequivariance-ops-cu13` runtime
package carries an NVIDIA math libraries SDK licence; the ClaudeKit kernel is
Apache-2.0 except `fa3_utils.h`, which retains FlashAttention-3's BSD-3; and
CuTeDSL-derived sources are BSD-3 or an NVIDIA software licence marked per file.
Read those for design ideas. Do not copy them.

Foundry's BSD-3 permits a port and a derivative, and covers the weights, so a
Rust implementation loading RF3 weights is licence-clean. One caveat: RF3's
optimised path depends on cuEquivariance, which is Linux-only and separately
licensed, and that path is not the one this design mirrors.

## Checkpoint layout

Settled 2026-11-13 by reading the published
`rf3_foundry_01_24_latest_remapped.ckpt` under foundry's own loader,
`rf3._core.compat.load_checkpoint`. No GPU needed. The checkpoint dict carries
`current_epoch`, `global_step`, `model`, `optimizer`, `scheduler_cfg` and
`train_cfg`. The weights sit under `model`, and every parameter appears twice
inside it: raw weights prefixed `model.` and EMA shadow weights prefixed
`shadow.`. The net is stored EMA-wrapped, so the earlier open question is
answered: a loader must strip a prefix. `EMA.forward` dispatches to the shadow
at eval, at `_core/training/EMA.py:63-68`, so the shadow weights are the ones
foundry's inference runs. All values are float32.

After the `shadow.` prefix, the main stack's keys run
`recycler.pairformer_stack.<block>.` as the loader analysis inferred, for
example
`recycler.pairformer_stack.24.tri_attn_end.to_out.weight`. The MSA module
keeps its own spelling of the same operations: `recycler.msa_module.` carries
`tri_mult_outgoing` and `tri_mult_incoming` where the pairformer blocks carry
`tri_mul_outgoing` and `tri_mul_incoming`. The attention attributes are
spelled `tri_attn_start` and `tri_attn_end` in both. A loader that builds keys
from one spelling silently misses the other module.

## Open questions

- Whether the single-representation pair-biased attention belongs inside the
  pair-stack abstraction or outside it. Foundry runs it as step 6 of the block
  with 16 heads of width 24, which is a different shape from the triangle
  attention's 4 heads of 32, so it shares no dimensions with the four
  operations this document covers.
- Whether recycling belongs in the interface. Foundry recycles 10 times at
  inference, re-initialising the pair representation each time from the previous
  single representation, at `RF3_structure.py:294`.
