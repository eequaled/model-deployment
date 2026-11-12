# Design Document

## Overview

V2 keeps the five-graph split and the HTTP surface of V1, and adds three changes that
are independent enough to measure one at a time:

1. **Baseline-exact post-processing.** The 0.939 score is mostly whitespace: our
   `pure_text` keeps spaces that the organizer's `batch_decode` cleanup removes. A
   second mode reproduces that cleanup, selected by `POSTPROCESS`.
2. **Selective quantization.** Dynamic int8 on `MatMul` only for the two encoders;
   weight-only `MatMulNBits` for the decoder step graph; `decoder_prefill` and
   `embed_tokens` stay fp32. The 8-bit step graph is the baseline; a mixed 4-bit
   transformer plus 8-bit LM head is the candidate that has to pass the same accuracy
   gate. This is the configuration that preserved accuracy in the reference
   implementation we inspected; our earlier failure came from quantizing the whole
   decoder, including norms and attention activations. fp16 is rejected everywhere.
3. **Decoder graph surgery.** The step graph contains six `Transpose` nodes that
   re-layout the cross-attention keys on every token, even though those keys are
   constant after prefill. Prefill will emit them already transposed and the step
   graph will consume them directly.

Batching is explicitly out of scope until all three are measured. The final task
estimates its value; the default stays one worker and one waiting slot.

## Architecture

```text
BUILD STAGE (torch allowed)
  microsoft/Florence-2-base weights
        |
        v
  export_onnx.py
        |
        +--> 5 fp32 graphs -------------------------+
        |                                          |
        v                                          |
  quantize_onnx.py                                 |
        |                                          |
        +--> 3 int8 graphs (vision, encoder, step) |
        +--> 1 int4 graph  (step, mixed)           |
        |                                          |
        v                                          v
      optimize_decoder.py (fp32, int8, int4 step graphs)
                        |
                        v
                 models + quant.json

RUNTIME STAGE (no torch)
  models + quant.json
        |
        v
  ocr_engine.py   PRECISION = auto | fp32 | int8 | int4
        |
        v
  app.py          queue + fast 503
        |
        +--> GET /health
        |
        +--> POST /ocr --> postprocess (pure_text | baseline)
```

Both graph sets ship in the image; `PRECISION` only selects which files are loaded, so an
A/B needs no rebuild. Quantization and surgery run in the exporter stage and their
output is the only thing the runtime ever sees.

### Why five graphs instead of the architecture's three modules

Florence-2 is conceptually three modules: a DaViT vision encoder, a BART text encoder,
and a BART text decoder. The five ONNX files are a deployment split, not a different
model:

| File | Role | Why it is separate |
|---|---|---|
| `vision_encoder.onnx` | DaViT, 577 image tokens | compute-bound, runs once |
| `embed_tokens.onnx` | the token embedding table (one `Gather`); the decoder LM head is its transpose | our step graph takes `inputs_embeds`; prompt and start embeddings are precomputed at startup, so a request pays one tiny Gather per token |
| `encoder_model.onnx` | BART encoder over 577 image + 10 prompt tokens | compute-bound, runs once; dynamic MatMul INT8 suits it |
| `decoder_prefill.onnx` | first decoder pass, emits logits + 24 present tensors | computes the cross-attention KV once; stays fp32 because 8-bit MNB is slower on its matrix-matrix shapes |
| `decoder_step.onnx` | one cached token per call | hot loop, 200–256 calls per page; weight-only 8-bit is the baseline, with a gated mixed 4-bit + 8-bit-LM-head candidate |

The decoder is split at the KV-cache boundary because prefill and step are different
computations with different shapes and different bottlenecks. A merged `If` graph (the
three-file alternative) stores the decoder weights once, but every token crosses an
`If` container and both branches must expose identical outputs. That forces the step
branch to pass the cross-attention cache through to its outputs: at batch 1 the cache
is 12 tensors of `1 × 12 × 577 × 64` fp32, about 21 MB, so a 225-token page would copy
roughly 4.8 GB unless the graph is surgically changed. The reference implementation
had exactly those 12 `Identity` nodes and removed them with a custom pass; our step
graph instead takes 24 past tensors and returns only the 12 self-attention presents, so
the cross cache is never copied out. The cost is that the decoder weights are stored
twice: measured **359.4 MB fp32** (201.8 MB transformer plus the 157.6 MB LM head). In
V2 the prefill copy stays fp32 (388 MB) and the step copy is 8-bit (104 MB), so the
split costs **104 MB** versus a merged 8-bit decoder, or **388 MB** versus a merged
fp32 decoder. We accept that for a smaller hot-loop contract, separate quantization
schemes per graph, and ORT optimizing each graph for its real shapes. The five sessions
run strictly in sequence, so their five intra-op pools never compete for the CPU quota.

Measured on one 255-token FUNSD page at `--cpus=4` with the V1 fp32 graphs:

| Stage | Time | Share | Calls |
|---|---:|---:|---:|
| `vision_encoder` | 1.82–1.97 s | 25–26% | 1 |
| `embed_tokens` | 0.012 s | 0.2% | 255 |
| `encoder_model` | 0.20–0.27 s | 3% | 1 |
| `decoder_prefill` | 0.043–0.066 s | 0.6–0.9% | 1 |
| `decoder_step` | 4.96–5.21 s | 69–70% | 255 |
| Python glue | 0.14–0.20 s | 2% | — |

The hot loop is 69–70% of the request, so it is the graph that must get weight-only
quantization and the transpose surgery. Prefill is under 1%, so merging it into the hot
graph saves one session call but forces the hot loop to share the prefill quantization
scheme. The embed graph is 0.2%; folding it into the hot graph cannot pay for the
157.6 MB table it would add. Vision is 25–26% and the text encoder is 3%; both are
compute-bound and benefit from dynamic INT8.

`embed_tokens` could be folded back into the step graph, and was tried in V1: it added
150.3 MiB to the hot graph and broke the step I/O contract for about 0.02 ms/token, so
it was reverted. Keeping it separate is the cheaper trade.

### Why not fuse everything

Fusion is a memory optimization that taxes the hot loop. The measured weight bill for
the three layouts:

| Layout | fp32 weights | int8 estimate | Hot-loop contract | Quantization |
|---|---:|---:|---|---|
| Five graphs (chosen) | 1.445 GB | ~0.55 GB | step returns only 12 self presents | per graph |
| Three graphs (merged decoder) | 1.085 GB | ~0.45 GB | `If` plus cross-cache passthrough unless surgered | one scheme for both decoder branches |
| One graph (whole pipeline) | 1.085 GB | ~0.45 GB | ONNX `Loop`, dynamic cache, argmax inside the graph | one scheme for the whole model |

The three-graph layout saves the measured 359 MB fp32 duplicate (388 MB in the V2
layout, where prefill stays fp32 and step is 8-bit), but it forces prefill and step to
share one quantization scheme. Dynamic activation INT8
suits prefill (matrix-matrix, compute-bound); weight-only `MatMulNBits` suits step
(matrix-vector, memory-bound). It also reintroduces the 12 cross-cache passthrough
outputs unless we repeat the reference implementation's surgery, adds `If` dispatch to
every token, and loses the `prefill_ms` versus `decode_ms` attribution that
Requirement 11 needs.

The one-graph layout is worse. Exporting the autoregressive loop with a growing KV
cache into ONNX is fragile, removes host-side EOS and max-token control, makes
per-stage timing impossible without internal profiling, and forces one quantization
scheme across a compute-bound vision encoder and a memory-bound decoder. ORT would also
have to optimize a ~1.1 GB graph at load time; the five V1 graphs already take about
9 s to become healthy.

A four-graph variant would have to fold `embed_tokens` into the step graph. The naive
fold adds the 157.6 MB table to the hot graph (V1 measured 150.3 MiB) and saves only
the measured 12 ms per 255-token page; the only way to avoid the table is to replace
the LM-head `MatMul` with a transposed `Gemm` so the Gather and the LM head share one
initializer, which changes the hot-loop kernel and needs its own A/B. A six-graph
variant would split a homogeneous stage (vision blocks, step attention/FFN, or the LM
head), adding a session call (measured 0.047 ms) and an intermediate tensor without
changing the quantization or shape regime. Each additional session also costs
0.4–2.3 s of load time and 150–400 MB of RSS in the V1 measurements, so a sixth graph
has to earn its keep. Neither four nor six is a free win.

Fusion would only win if memory were the binding constraint (for example a 2 GB limit)
or if we gave up mixed precision and per-stage metrics. Under the 4 GB limit, int8
already brings peak to roughly 2.4–2.8 GiB, so the 90 MB saved by merging prefill and
step is about 3–4% of peak. The five-graph split is the better trade for this
challenge.

### Precision variants

The runtime version is resolved: **ONNX Runtime 1.31.1** is installed and verified
against the existing five fp32 graphs (same text, 7.0 s load). ORT 1.20.1 shipped only
the 4-bit `MatMul4BitsQuantizer`; its CPU `MatMulNBits` kernel rejects 8-bit weights.
ORT 1.22 added 8-bit, and 1.31.1 is the newest release with the faster AVX2 4-bit
kernels. The exporter and runtime therefore move together to 1.31.1.

The four combinations the user asked about resolve as follows:

| Base | Quantization | Verdict |
|---|---|---|
| fp32 | dynamic INT8 / `MatMulNBits` | **use** |
| fp16 | none | reject: CPU inserts fp16↔fp32 casts, no speed win |
| fp16 | INT8 | reject: casts to fp32 before quantizing, extra rounding |
| fp16 | 4-bit | reject: packed weights are the same size, fp16 scales are noise |

fp16 is a memory-only change on x86-64 CPU. ORT maintainers state that the CPU
provider computes in fp32 and adds fp16↔fp32 conversions around unsupported nodes, so
a float16 model can be slower than the float32 one. `MatMulNBits` also converts fp16
activations to fp32 on non-ARM CPUs. No fp16 anywhere.

`MatMulNBits` supports bits 2–8, block sizes that are powers of two and at least 16,
and `accuracy_level` 0 (unset), 1 (fp32 activations), 2 (fp16), 3 (bf16), 4 (int8
activations). **`accuracy_level=4` is the fastest and least accurate mode**, not the
most accurate; it quantizes activations to int8. The 8-bit baseline uses it. If the
4-bit variant is marginal on the 50-image gate, the first knob to test is
`accuracy_level=1` on the LM head only, accepting its slower fp32 activation path.

4-bit is a **decode-only** win. On this Zen 2 CPU (AVX2, no AVX-VNNI), synthetic
MatMul medians at 4 threads, ORT 1.31.1:

| Shape | fp32 | dyn INT8 | 8-bit MNB | 4-bit MNB a4 |
|---|---:|---:|---:|---:|
| 768×768, M=1 | 0.028 ms | 0.020 | 0.019 | 0.020 |
| 768×3072, M=1 | 0.056 ms | 0.042 | — | 0.033 |
| 768×51289 LM head, M=1 | 5.14 ms | 1.79 | 1.75 | **0.49** |
| 768×768, M=128 | 0.44 ms | 0.42 | 0.93 | 0.59 |
| 6-layer, 48-MatMul chain, M=1 | 6.90 ms | 1.99 | 2.43 | **1.28** |

4-bit wins only when M=1 and the weight stream dominates. For prefill-shaped
matrix-matrix work, 4-bit is slower than fp32 and 8-bit is nearly 2× slower than fp32.
**Never 4-bit the prefill graph.** The decoder step is the only 4-bit target.

The LM head is the most sensitive matrix: it produces 51k logits directly, and 4-bit
error can flip rare tokens over 200+ steps. The recommended layout is therefore
**4-bit transformer + 8-bit LM head**, not 4-bit everywhere. Measured on the real step
graph:

| Step graph | File size | Quantized MatMuls |
|---|---:|---:|
| fp32 | 359.6 MB | 0 |
| 8-bit all | 103.9 MB | 49 × 8-bit |
| 4-bit all | 59.5 MB | 49 × 4-bit |
| mixed: 4-bit transformer + 8-bit LM head | **79.2 MB** | 48 × 4-bit + 1 × 8-bit |

The mixed graph is built in two passes: first 4-bit with
`nodes_to_exclude=["/language_model/lm_head/MatMul"]`, then 8-bit on the result. After
pass one the only remaining `MatMul` with a constant weight is the LM head, so pass two
quantizes exactly that node. This was verified: the mixed file contains 48 four-bit
`MatMulNBits` nodes and one eight-bit `MatMulNBits` node named
`/language_model/lm_head/MatMul_Q8`.

Measured on one real 255-token FUNSD page, fresh process per variant, 4 intra-op
threads (the same thread count used at `--cpus=4`):

| Variant | Total | Decode | Step tok/s | Output |
|---|---:|---:|---:|---:|
| fp32 | 6545.7 ms | 4743.4 ms | 54.55 | 802 chars |
| 8-bit step | 4390.1 ms | 2606.2 ms | 100.38 | 804 chars |
| 4-bit mixed step | 3895.9 ms | 2005.8 ms | **131.51** | 760 chars |

The mixed graph is 31% faster than 8-bit on decode and 2.4× faster than fp32. But the
same page lost 42 characters under 4-bit, versus a 2-character difference under 8-bit.
One page is not the gate. The 50-image set decides, and 8-bit remains the safe default
until the mixed variant passes it.

The fallback ladder, in order: mixed 4-bit + 8-bit LM head → 8-bit all → 8-bit
attention + 4-bit FFN + 8-bit LM head → fp32. The first rung is the only one that
plausibly beats the friend's 8-bit decoder; the others are safety nets.

**Triton is rejected in both senses.** NVIDIA Triton Inference Server with the ONNX
Runtime backend on x86 CPU simply embeds ORT and inherits its kernels; it adds no 4-bit
CPU acceleration and its serving overhead is wrong for one 4 GB container at batch 1.
OpenAI Triton is a GPU kernel compiler whose CPU backend is experimental; it cannot
execute an ORT graph and would require PyTorch at serve time.

The experiment order is: (1) fp32 and 8-bit accuracy on the 50 images; (2) mixed 4-bit
on the same 50; (3) only if the mixed variant passes, compare decode tok/s and peak
memory; (4) if it misses, walk the fallback ladder. The 8-bit graph is built and
measured first because it is the baseline the mixed variant must beat.

### Research basis

The split follows the standard ONNX export for encoder-decoder models. Hugging Face
Optimum exports `encoder_model.onnx`, `decoder_model.onnx`, and
`decoder_with_past_model.onnx` as separate sessions because the first decoder pass and
the cached pass have different inputs and outputs. Optimum's merged-decoder option
exists to save memory, and its own issue tracker notes that the two branches do not
share the same output set, which makes an `If` merge harder. ONNX Runtime's `Loop`
implementation documents that loop-carried variables whose shape changes cannot avoid
a copy when the final shape is needed, which is what a growing KV cache does. At the
model level, Splitwise and DistServe make the same distinction: prefill is
compute-bound and decode is memory-bound, so the two phases are tuned separately.

- Optimum ONNX export: <https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model>
- Optimum merged-decoder caveat: <https://github.com/huggingface/optimum/pull/872>
- ORT `Loop` copy: <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/controlflow/loop.cc>
- Splitwise: <https://arxiv.org/abs/2311.18677>
- DistServe: <https://arxiv.org/abs/2401.09670>

## Metrics and instrumentation

Every request records wall-clock boundaries with `time.perf_counter()` around the five
session calls and the Python glue:

| Field | Definition |
|---|---|
| `vision_ms` | `vision_encoder.run` |
| `embed_ms` | total `embed_tokens.run` time for the request (startup precompute excluded) |
| `text_encoder_ms` | `encoder_model.run` |
| `prefill_ms` | `decoder_prefill.run` plus the first-token argmax |
| `decode_ms` | from after prefill to the last token: step ORT calls, per-token embeddings, argmax, cache swaps |
| `total_ms` | preprocess through decoded text |
| `steps` | number of `decoder_step.run` calls |
| `tokens` | generated tokens, including the first token produced by prefill and EOS if emitted |
| `decode_tokens_per_s` | `steps * 1000 / decode_ms` |
| `total_tokens_per_s` | `tokens * 1000 / total_ms` |

Example response:

```json
{
  "text": "...",
  "timings": {
    "vision_ms": 1210.4, "embed_ms": 4.2, "text_encoder_ms": 240.1,
    "prefill_ms": 310.7, "decode_ms": 2840.9, "total_ms": 4606.3,
    "steps": 224, "tokens": 225,
    "decode_tokens_per_s": 78.8, "total_tokens_per_s": 48.8
  }
}
```

`decode_tokens_per_s` is the metric that isolates the hot loop; `total_tokens_per_s`
includes the fixed encoder cost and is the one that tracks end-to-end latency. The
existing `encoder_ms` and `decode_ms` keys stay for V1 compatibility; `encoder_ms`
remains vision + text encoder.

`bench/load.py` aggregates both rates and every stage across a step and reports mean,
p50, and p95. Instrumentation adds about five `perf_counter` pairs per request, well
under the 0.5% gate.

## Components and Interfaces

### `export/quantize_onnx.py` (new)

```text
python export/quantize_onnx.py --models /models
```

- `vision_encoder`, `encoder_model`: `quantize_dynamic(..., weight_type=QInt8,
  op_types_to_quantize=["MatMul"])`, written as `<name>.int8.onnx`.
- `decoder_prefill`: copied unchanged as fp32. It is under 1% of request time and
  8-bit `MatMulNBits` is slower than fp32 on its matrix-matrix shapes. It is never
  4-bit.
- `decoder_step`: two outputs.
  - `decoder_step.int8.onnx`: `MatMulNBitsQuantizer(bits=8, block_size=32,
    is_symmetric=True, accuracy_level=4)`.
  - `decoder_step.int4.onnx`: pass one `MatMulNBitsQuantizer(bits=4, block_size=32,
    is_symmetric=True, accuracy_level=4,
    nodes_to_exclude=["/language_model/lm_head/MatMul"])`; pass two
    `MatMulNBitsQuantizer(bits=8, block_size=32, is_symmetric=True,
    accuracy_level=4)` on the result, which now has exactly one constant-weight
    `MatMul` left, the LM head.
- `embed_tokens`: copied unchanged; recorded as `fp32`.
- Writes `quant.json` and prints source/target sizes per graph.
- Fails the build if a graph produces zero quantized nodes, so a silent no-op cannot
  masquerade as a win. For the mixed file, assert 48 four-bit `MatMulNBits` nodes and
  one eight-bit LM-head node.
- **Version:** ORT 1.31.1 is the resolved pin. It supports 8-bit `MatMulNBits`; 1.20.1
  did not. The exporter stage installs `onnxruntime==1.31.1`, `onnx_ir`, and
  `ml_dtypes`.

### `export/optimize_decoder.py` (new)

```text
python export/optimize_decoder.py --models /models
```

Operates on the fp32, int8, and int4 step graphs and the fp32 prefill graph:

- In `decoder_prefill`, rename the internal cross-key value, append a `Transpose`
  with `perm=[0,1,3,2]`, and make that the `present.{i}.encoder.key` output. The
  prefill's own attention keeps the original layout.
- In `decoder_step`, find the `Transpose` whose input is
  `past_key_values.{i}.encoder.key`, assert its permutation, delete it, and rename its
  output to the input name.
- Update `ValueInfo` shapes for the affected inputs/outputs.
- `onnx.checker.check_model` after each edit; a failure aborts the build.

### `server/ocr_engine.py`

- `load_precision(models_dir, requested) -> dict[str, Path]` reads `quant.json`.
  `requested` is `auto` (use the measured winner), `fp32` (ignore quantized files),
  `int8` (8-bit step), or `int4` (mixed 4-bit step). A missing file for an explicitly
  requested precision is a startup error, not a silent fallback.
- `decode_text(tokenizer, tokens, mode)` implements `pure_text` and `baseline`.
  `baseline` applies the transformers cleanup replacements after the special-token
  removal.
- `generate()` records the stage boundaries listed in **Metrics and
  instrumentation**: `vision_ms`, `embed_ms`, `text_encoder_ms`, `prefill_ms`,
  `decode_ms`, `total_ms`, `steps`, `tokens`, and the two rates. The existing
  `encoder_ms`/`decode_ms` keys stay for compatibility.
- Cache maps and precomputed embeddings are unchanged; the surgery only changes the
  layout of the cross-key tensors, which the maps already pass by name.

### `server/app.py`

New env vars: `PRECISION` (`auto` default, resolving to `int8` until the 4-bit gate
passes), `POSTPROCESS` (`baseline` default if it wins the measurement),
`ALLOW_FP32_FALLBACK` (`0` default). Queue, worker, pixel cap, and upload cap are
unchanged.

### `bench/load.py`

Extend `summarize()` with mean/p50/p95 for `decode_tokens_per_s`,
`total_tokens_per_s`, and each stage timing (`vision_ms`, `text_encoder_ms`,
`prefill_ms`, `decode_ms`). Missing timing fields in a response are skipped, so the
same bench still works against a V1 server.

### `bench/precision_sweep.sh` (new)

Starts the image once per `PRECISION` value (`fp32`, `int8`, `int4`) at
`--memory=4g --cpus=4`, then runs `bench.accuracy`, `bench.load --mode latency`,
`bench.load --mode open`, and reads `VmHWM`. Writes one JSON per config under
`results/precision/`.

### `bench/postprocess_eval.py` (new)

Scores a directory of images twice, once per post-processing mode, against the same
references, and prints per-image and mean CER. Runs against a live server; the mode is
selected by env, so the script restarts the container or takes `--url` per mode.

### `bench/batch_probe.py` (new)

Sequential only: sends a batch of N images in one request body is not part of the API,
so this probes the server's `MAX_BATCH` env by starting containers with 1, 2, and 4 and
running the same open-loop step. Reports RPS, p95, and peak memory. This is the input
to the final batching decision.

## Data Models

`models/quant.json`:

```json
{
  "onnxruntime": "1.31.1",
  "source": "microsoft/Florence-2-base@<revision>",
  "variants": {
    "fp32": {
      "vision_encoder": "vision_encoder.onnx",
      "embed_tokens": "embed_tokens.onnx",
      "encoder_model": "encoder_model.onnx",
      "decoder_prefill": "decoder_prefill.onnx",
      "decoder_step": "decoder_step.onnx"
    },
    "int8": {
      "vision_encoder": "vision_encoder.int8.onnx",
      "embed_tokens": "embed_tokens.onnx",
      "encoder_model": "encoder_model.int8.onnx",
      "decoder_prefill": "decoder_prefill.onnx",
      "decoder_step": "decoder_step.int8.onnx"
    },
    "int4": {
      "vision_encoder": "vision_encoder.int8.onnx",
      "embed_tokens": "embed_tokens.onnx",
      "encoder_model": "encoder_model.int8.onnx",
      "decoder_prefill": "decoder_prefill.onnx",
      "decoder_step": "decoder_step.int4.onnx"
    }
  },
  "schemes": {
    "vision_encoder": "dynamic-matmul-qint8",
    "embed_tokens": "fp32",
    "encoder_model": "dynamic-matmul-qint8",
    "decoder_prefill": "fp32",
    "decoder_step.int8": "matmulnbits-qint8-b32-a4",
    "decoder_step.int4": "matmulnbits-mixed-4b8-b32-a4"
  }
}
```

`POSTPROCESS` values: `pure_text` (V1 behaviour) and `baseline` (adds the cleanup).
`PRECISION` values: `auto`, `fp32`, `int8`, `int4`. `auto` selects the measured winner;
until the 50-image gate says otherwise it resolves to `int8`, not `int4`.

## Expected Wins

Measured V1 baseline, measured step-graph variants on one real 255-token page, and
projected V2 ranges. The encoder saving assumes 40–45% (the reference implementation
measured 48% on a CPU with VNNI; Zen 2 has no VNNI, so the low end is used). The
decoder rows are measured with the fp32 encoders still in place; the encoder rows are
still projections.

| Metric | V1 measured | V2 measured / projected | Basis |
|---|---|---|---|
| Vision + text encoder | 2.44 s | 1.3–1.7 s | MatMul-only dynamic int8 (projected) |
| Decode, 255-token page | 4.74 s | 2.61 s int8 / 2.01 s int4 | measured, step graph only |
| Step tokens/s | 54.6 | 100.4 int8 / 131.5 int4 | measured |
| Total, 255-token page | 6.55 s | 4.39 s int8 / 3.90 s int4 | measured, fp32 encoders |
| p50 / p95, 50 images | 8.29 / 9.35 s | pending 50-image sweep | gate |
| Sustained RPS | 0.15 | 0.25–0.30 projected | inverse of latency |
| Peak memory | 3.36 GiB | 2.4–2.8 GiB int8, ~2.3–2.7 GiB int4 | weight sizes measured |
| Accuracy (50 refs) | 0.939 | int8 TBD, int4 TBD | 50-image gate |

The 4-bit mixed graph is faster than 8-bit, but its first-page text lost 42 characters
versus 2 for 8-bit. Speed does not decide this; the 50-image accuracy gate does. Until
it passes, `auto` resolves to `int8`.

These are measurements on one page and projections elsewhere, not claims about the
50-image set. Requirement 2 and 5 set the gates; if a stage misses its gate the change
is rejected and V1 stays the default.

### Batching estimate (deferred, not adopted)

On four CPUs one image already occupies every core, so a second image in the same
batch takes about twice as long. The reference implementation measured batch 1 vs 2 at
the same RPS (0.83 vs 0.81) with worse p50 (4.9 s vs 5.4 s) and +0.76 GB RAM. With
int8 the memory cost is smaller, but the compute argument does not change. Expected
V2 win: **0–5% RPS**, at the cost of roughly doubling p95 and adding 0.5–0.8 GiB. It
will be measured once and almost certainly left off; it only becomes interesting if
the organizer allocates 8 or more CPUs.

## Error Handling

| Failure | Behaviour |
|---|---|
| `quant.json` missing while `PRECISION=int8` or `int4` | startup error with the missing path |
| quantized file missing while `PRECISION=auto` | startup error; `ALLOW_FP32_FALLBACK=1` downgrades with a warning |
| quantizer converts zero nodes | build fails with the graph name |
| mixed 4-bit graph has the wrong node mix | build fails; expected 48 × 4-bit and 1 × 8-bit |
| surgery cannot find an expected `Transpose` | build fails with graph and layer |
| `onnx.checker` rejects an edited graph | build fails |
| unknown `POSTPROCESS` / `PRECISION` value | startup error listing valid values |
| quantized model raises at inference | 500 JSON, request logged; the server does not crash |
| accuracy gate missed | change rejected, fp32 stays default |

## Testing Strategy

- **Unit, no weights:** post-processing cleanup cases; precision selection with a fake
  `quant.json`; surgery on the tiny decoder graph (delete transposes, compare logits
  before/after within `1e-4`); quantizer smoke on the tiny model where block size
  allows, skipped otherwise.
- **Integration, real export:** build fp32, int8, and int4 files; run one image through
  each precision and compare token agreement and text; assert `quant.json` matches the
  files on disk and the mixed file has 48 four-bit plus one eight-bit `MatMulNBits`.
- **Accuracy:** 50 FUNSD images against the existing fp32 references, per-image and
  mean, for both post-processing modes and all three precisions.
- **Performance:** sequential p50/p95, open-loop 0.10/0.15/0.20/0.25/0.30, peak
  `VmHWM`, and `bench.stress` on the winning config.
- **Metrics:** timing-key presence and formula tests, `steps + 1 == tokens` without
  EOS, per-stage aggregation with missing fields, and an overhead check against an
  uninstrumented run.
- **Contract:** the existing API tests plus the 11 MB and 64 MP paths run unchanged.
