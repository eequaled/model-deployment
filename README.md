# Florence-2 OCR on CPU, ONNX Runtime only

Deployment of `microsoft/Florence-2-base` behind a **FastAPI** HTTP API, running on
CPU with no PyTorch in the serving image. The model is exported from the original
weights by `export/` and served by `server/` through ONNX Runtime.

Target constraints: **4 GB RAM, CPU only, PyTorch forbidden at serving time.**

* `journal.md` — development log, experiments, numbers, and dead ends.
* `export/export_onnx.py` — turns the original checkpoint into five ONNX graphs.
* `export/quantize_onnx.py` — builds the FP32, INT8, and mixed INT4 variants and
  writes `quant.json`.
* `export/optimize_decoder.py` — decoder graph surgery for the cached step loop.
* `server/` — **FastAPI + uvicorn** app plus the ONNX Runtime decoding loop.
* `bench/` — CER accuracy scoring, latency/throughput measurement, the prefill
  probe, the precision sweep, and old-vs-new graph parity.
* `runall.sh` — one long evidence-collecting run (export, parity, tests, server,
  bench, Docker).
* `Dockerfile` — two stages: PyTorch only to export, then a runtime with none.

## Status at this commit

| Item | State |
|---|---|
| Serving stack | **FastAPI + uvicorn**; one inference worker, bounded queue, fast 503s |
| Model graphs & quantization | 5 graphs; `quant.json` provides `fp32`, `int8`, and `int4` variants |
| Active precision default | `PRECISION=int8`: **INT8 step graph (104 MB)** + FP32 encoders/prefill; INT4 (79 MB) available via `PRECISION=int4` for max speed |
| Hardware adaptation | `has_avx_vnni()` at startup; only activates with `PRECISION=auto` (FP32 → QUInt8 encoder on VNNI CPUs) |
| In-memory embedding table | 255 per-token ONNX `Gather` calls eliminated; direct $O(1)$ NumPy table slicing |
| Decoder graph surgery | Cross-attention keys transposed once in prefill; 6 `Transpose` nodes removed from step |
| Post-processing | `POSTPROCESS=baseline` replicates transformers tokenization whitespace cleanup |
| Unit + integration tests | **52 passed, 6 skipped** without weights; **58 passed** with a real export; 0 errors |
| Step 1 latency (4 CPUs, 50 docs) | sequential p50 **3.43 s**, p95 **3.68 s**, max 3.81 s |
| Decode throughput | p50 **162.3 decode tokens/s** (mean 162.2 tok/s, up to 180+ tok/s on dense pages) |
| Total throughput | p50 **72.7 total tokens/s** |
| Step 2 throughput (4 CPUs) | **0.35 req/s sustained at 100% success** (measured 0.20/0.25/0.30/0.35 all clean) |
| Memory envelope | peak `VmHWM` **2.39 GiB** under load & overload spike (safe inside 4 GB limit) |
| Accuracy vs local proxy | 50 FUNSD docs: **0.7991 mean accuracy** (well above the 0.70 challenge floor; `PRECISION=int8` is the 0.8753 accuracy-first fallback) |

## Build and run

```bash
docker build -t florence2-ocr .
docker run --rm -p 8000:8000 --memory=4g --memory-swap=4g --cpus=4 florence2-ocr
```

The image exports the model during the build. Nothing pre-converted is downloaded;
the only input is the original `microsoft/Florence-2-base` checkpoint, pinned to a
revision recorded in `export/export_onnx.py` and in the generated `meta.json`.

```bash
curl -F "image=@samples/example.png" http://localhost:8000/ocr
curl http://localhost:8000/health
```

**Build note.** The exporter stage installs CPU PyTorch **and CPU torchvision
together** from the CPU index, then `timm`/`einops`/`onnx`:

```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.5.1" "torchvision==0.20.1" \
    && pip install "transformers==4.48.3" "onnx==1.17.0" \
       "onnxruntime==1.31.1" "onnx_ir" "ml_dtypes" "safetensors" \
       "pillow" "numpy" "huggingface_hub" "timm" "einops"
```

Installing CPU `torch` alone lets `timm` pull a mismatched PyPI `torchvision`, and
the export dies with `RuntimeError: operator torchvision::nms does not exist`. The
pair has to come from the same index, and it is pinned so a future build cannot
silently resolve a torch that the export code has never been tested against.

## How the model runs

Five graphs, because the pieces have genuinely different lifetimes:

| Graph | Called | Why separate |
|---|---|---|
| `vision_encoder.onnx` | once per request | the 768×768 DaViT pass dominates prefill and never changes |
| `embed_tokens.onnx` / `embed_tokens.npy` | once per prompt, once per token | a 51k×768 lookup is cheaper as its own table than folded into decoding; the server slices the `.npy` table in memory and falls back to the ONNX graph when the file is absent |
| `encoder_model.onnx` | once per request | fuses the 577 image tokens with the 10 prompt tokens |
| `decoder_prefill.onnx` | once per request | returns logits plus the whole attention cache |
| `decoder_step.onnx` | once per token | takes the previous token's embedding, returns logits plus 12 updated self-attention tensors |

The cross-attention key/value tensors are built in prefill and reused unchanged for
the entire request, which is why the step graph accepts 24 cached tensors and
returns only 12: it has nothing new to say about the image.

Decoding is greedy (`argmax`) with a cap of `MAX_NEW_TOKENS` (default 256, from
`meta.json`) and stops at EOS. Greedy is deliberate: the reference is the fp32
original, and sampling would add variance without improving CER.

A request flows through FastAPI (`/ocr`) into a fixed `ThreadPoolExecutor`, then
through `FlorenceOCR.generate`: PIL preprocessing → `vision_encoder` → `encoder_model`
→ `decoder_prefill` → the `decoder_step` loop → tokenizer cleanup. The JSON response
is `{"text": "...", "timings": {...}}`; the timings include vision, text encoder,
prefill, decode, embedding, total milliseconds, step count, and decode/total tokens
per second.

## Serving stack: FastAPI on ONNX Runtime

The HTTP layer is **FastAPI** with **uvicorn**, not a prebuilt ML serving server.
That keeps the submission inside the challenge rule (“no prebuilt ML serving
servers”) and leaves the process model explicit:

* `GET /health` returns `{"status": "ok"}` only after the engine has loaded and the
  2-token warm-up has touched all five sessions. While loading it returns 503.
* `POST /ocr` accepts multipart field `image`, returns `{"text": ..., "timings": ...}`.
  Missing/empty/unreadable images return 4xx JSON; uploads above 10 MB return 413;
  decoded images above 40 MP return 413 before `image.load()` can allocate the bitmap.
* Inference is blocking CPU work, so it runs in a `ThreadPoolExecutor` via
  `loop.run_in_executor`; the asyncio event loop stays free to answer `/health` and
  reject overload immediately.
* `OCR_WORKERS` controls the executor size. The shipped default is **1** because one
  image already saturates the four CPUs. `QUEUE_LIMIT` adds waiting slots behind the
  worker; when the semaphore is full, `/ocr` returns a fast 503 instead of growing an
  unbounded queue that would turn latency into an OOM.
* `server/serve.py` refuses to start if `torch` is already imported, and uvicorn is
  pinned to one process (`workers=1`). Scaling is a thread-count decision here, not
  a process-count decision, because each process would load its own weight set.

## Optimization inventory

Every optimization in the shipped build, what it does, and where it lives. Numbers
are from the 50-document FUNSD sweep at `--cpus=4` unless noted.

| Optimization | What it is / how it works | Where | Effect |
|---|---|---|---|
| FastAPI + uvicorn serving | Async HTTP surface with a thread pool for blocking ORT calls, a bounded semaphore queue, JSON errors, and a warm-up gate before `/health` | `server/app.py`, `server/serve.py` | responsive health/503s; flat memory under overload |
| Five-graph split | Vision, embedding lookup, text encoder, decoder prefill, and decoder step are separate ONNX files with separate lifetimes | `export/export_onnx.py` | each graph gets the shapes and precision it actually needs |
| Per-graph ORT sessions | Each graph loads with `ORT_ENABLE_ALL`; intra-op threads come from the cgroup CPU quota, inter-op is 1, execution is sequential | `server/ocr_engine.py` | no cross-graph fusion that would defeat the precision split |
| Cross-attention KV cache | Prefill computes the image cross-attention once; every step reuses 24 cached tensors and returns only the 12 self-attention updates | graph I/O contract | avoids recomputing 587 positions of cross-attention per token |
| Decoder graph surgery | Prefill emits cross-attention keys already transposed (`perm=[0,2,3,1]`); the step graph drops the six per-token `Transpose_1` nodes | `export/optimize_decoder.py` | removes 6 transposes from every one of ~255 steps; logits match within 1e-4 |
| One-position prefill | The decoder prefix is the start token only (or the last prompt token), not all 10 prompt positions | `export/`, `server/ocr_engine.py` | 10× less decoder compute for the first token; matches `model.generate()` |
| `prefill_mode` in `meta.json` | The winning prefix is recorded at export time and can be overridden with `PREFILL_MODE` | `export/export_onnx.py` | the correct prefix is evidence, not folklore |
| In-memory embedding table | `embed_tokens.npy` holds the 51,289×768 float32 table; the step loop slices it with NumPy instead of calling the ONNX `Gather` session | `export/quantize_onnx.py`, `server/ocr_engine.py` | embed time across 255 steps 14.2 ms → 4.0 ms; exact match |
| Mixed INT4 decoder step | 48 four-bit `MatMulNBits` transformer nodes + one eight-bit LM head; block 32, symmetric, `accuracy_level=4` | `export/quantize_onnx.py` | 79 MB; 162.3 decode tok/s vs 135.2 INT8 and 54.6 FP32 |
| INT8 decoder step fallback | Weight-only eight-bit `MatMulNBits` for the whole step graph | `export/quantize_onnx.py` | 104 MB; 135.2 tok/s; 0.8753 mean accuracy |
| QUInt8 text encoder | Dynamic per-channel unsigned-8-bit `MatMul` quantization of `encoder_model` | `export/quantize_onnx.py` | 46 MB; saves ~15 ms/page on Zen 2, ~110 ms/page on AVX-VNNI; accuracy trade |
| FP32 vision encoder | DaViT vision tower stays FP32 | `quant.json` | avoids optical feature drift; INT8 did not pay off on Zen 2 |
| FP32 decoder prefill | Prefill stays FP32 | `quant.json` | ~46–50 ms once per request; INT8 unpacking is slower and the KV cache stays clean |
| FP32 embedding table | The lookup table stays float32 | `quant.json` | no matmul to accelerate; quantizing adds dequantization work |
| AVX-VNNI detection | Reads `/proc/cpuinfo` flags (Docker/WSL2/Docker Desktop on Windows) or `platform.processor()` on bare-metal Windows | `server/ocr_engine.py` | `auto` upgrades only the text encoder on VNNI CPUs; explicit `PRECISION` never is rewritten |
| ORT thread sizing | `ORT_THREADS=0` sizes the intra-op pool from the cgroup CPU quota, divided by workers | `server/ocr_engine.py` | 4 threads at `--cpus=4` instead of oversubscribing host cores |
| Spinning off + memory arena | `session.intra_op.allow_spinning=0`; CPU memory arena on by default | `server/ocr_engine.py` | less scheduler noise under a CPU quota |
| Bounded queue + fast 503 | `asyncio.Semaphore(WORKERS + QUEUE_LIMIT)`; full queue rejects immediately | `server/app.py` | 6.1 ms worst-case refusal under a 10-client spike; no OOM |
| Warm-up before health | A 2-token generation touches all five sessions before `/health` returns 200 | `server/ocr_engine.py`, `server/app.py` | no cold first request; ORT kernel plans are resident |
| Prompt ids baked into `meta.json` | The processor replaces `<OCR>` with a sentence; the exporter records those ids | `export/export_onnx.py` | removes per-request tokenization and a silent-error class |
| Greedy `argmax`, stop at EOS | No beam search, no sampling | `server/ocr_engine.py` | matches the fp32 reference; avoids 3× beam state |
| Baseline post-processing | `clean_up_tokenization_spaces` matching transformers’ default | `server/ocr_engine.py` | 50-image mean 0.8769 vs 0.8767 for `pure_text` |
| Preprocessing parity | RGB, bicubic 768×768, rescale, ImageNet normalize, CHW; no EXIF rotation | `server/ocr_engine.py` | byte-exact with `CLIPImageProcessor` |
| Multi-stage torch-free build | Export stage carries PyTorch; runtime copies only graphs and installs ORT | `Dockerfile` | rule compliance; `torch`/`torchvision` absent at runtime |
| Per-stage metrics | Every request records vision, text encoder, prefill, decode, embed, steps, tokens, and token rates | `server/ocr_engine.py` | makes every optimization measurable |
| Benchmark harness | Accuracy, sequential/closed/open load, stress, precision sweep, graph parity, prefill probe | `bench/` | evidence instead of guesses |

## Quantization by graph

`quant.json` exposes three variants. The default `auto` resolves to the mixed INT4
step plus FP32 vision/prefill/embedding; the text encoder is FP32 unless the host
CPU has AVX-VNNI.

| Graph | Default | Scheme / file | Size | Called | Why this precision |
|---|---|---|---:|---|---|
| `vision_encoder` | **FP32** | `vision_encoder.onnx` | 366 MB | once | DaViT convolutions and patch projections are sensitive to small activation shifts; INT8 did not pay off on Zen 2 and a weight-only INT8 vision variant scored 0.7163 mean on 50 documents (rejected) |
| `embed_tokens` | **FP32** | `embed_tokens.onnx` / `embed_tokens.npy` | 158 MB each | once per prompt + per token | pure lookup, no matmul; quantizing adds dequantization without accelerating anything. The `.npy` path slices rows directly |
| `encoder_model` | **FP32** on non-VNNI; **QUInt8** on VNNI | `encoder_model.onnx` / `encoder_model.int8.onnx` | 173 MB / 46 MB | once | dynamic per-channel QUInt8 MatMul. On Zen 2 it saves only ~15 ms and drops accuracy (0.8753 → 0.8035 with INT8 step); on AVX-VNNI the same 8-bit dot products run in hardware, so `auto` upgrades |
| `decoder_prefill` | **FP32** | `decoder_prefill.onnx` | 388 MB | once | ~46–50 ms once per request; weight-only INT8 unpacking is slower on AVX2, and FP32 keeps the initial KV cache clean |
| `decoder_step` | **Mixed INT4** | `decoder_step.int4.onnx` | 79 MB | ~200–256 times | M=1 GEMV is memory-bandwidth bound; 4-bit weights halve the stream versus 8-bit and quarter it versus FP32. The LM head stays 8-bit because 4-bit rounding flips rare tokens over 250 steps |

### How the decoder quantization works

`MatMulNBits` is **weight-only** quantization in the sense that the weights are stored
at 4 or 8 bits and the operator fuses dequantization with the matmul. Weights are
stored in blocks of 32 with a symmetric scale per block, so each small group has its
own scale instead of one scale for the whole matrix. `accuracy_level=4` additionally
permits ORT to quantize the activation block to int8 internally for the int8
dot-product kernel; with `accuracy_level=0` the activations stay fp32. Because the
step graph runs at M=1, it is limited by how many bytes the CPU must stream from RAM,
not by arithmetic throughput. Shrinking the weights is therefore close to a direct
speedup:

| Step variant | File size | RAM traffic per 255-token page | Decode throughput | 50-image mean accuracy |
|---|---:|---:|---:|---:|
| FP32 | 360 MB | ~92 GB | 54.6 tok/s | 0.8769 |
| INT8 `MatMulNBits` | 104 MB | ~26.5 GB | 135.2 tok/s | 0.8753 |
| Mixed INT4 (48×4-bit + 1×8-bit) | 79 MB | ~20 GB | 162.3 tok/s | 0.7991 |

The mixed graph is built in two passes: 4-bit with the LM head excluded, then 8-bit
on the only remaining constant-weight `MatMul` (`/language_model/lm_head/MatMul`).
The LM head projects 768 hidden values to 51,289 vocabulary logits; 4-bit rounding
there caused rare-token flips over the autoregressive loop, while the transformer
layers tolerated 4-bit weights. The final graph must contain exactly 48 four-bit
nodes and one eight-bit node; the exporter checks that invariant.

The INT4 step is a **calculated trade**, not a free win: 0.7991 mean is comfortably
above the challenge's 0.70 mean floor, but below the earlier internal 0.90 mean /
0.75 per-image gate. `PRECISION=int8` loads the 0.8753 variant (FP32 encoder + INT8
step) when accuracy matters more than the last ~27 decode tokens/s.

### How the text-encoder quantization works

`encoder_model` uses **dynamic** quantization: weights are quantized per channel to
unsigned 8-bit (`QUInt8`), and activations are quantized on the fly during the
MatMul. The first attempt used per-channel `QInt8` and collapsed BART's token
stream; `QUInt8` preserved the text. Even so, the quantization noise costs accuracy:

* FP32 encoder + INT8 step: **0.8753** mean on 50 FUNSD documents.
* QUInt8 encoder + INT8 step: **0.8035** mean.
* QUInt8 encoder + INT4 step: **0.7922** mean.

On Zen 2 the QUInt8 encoder saves only ~15 ms because 8-bit dot products are
emulated with 16-bit unpack/multiply. On a CPU with AVX-VNNI / AVX-512 VNNI the same
kernel runs in hardware and can save ~110 ms per page (the reference implementation
measured 180 ms → 70 ms on an i5-13400F). That is why the upgrade is conditional on
the CPU flag rather than unconditional.

### Why FP32 for vision, prefill, and embedding

* **Vision encoder:** compute-bound matrix-matrix work at M=577. Dynamic INT8 with
  `ConvInteger` failed to load on the CPU EP here; a weight-only INT8 variant that
  did load was not enough faster on Zen 2 to justify its accuracy cost and scored
  0.7163 mean on the 50-image set. FP32 is the safer default.
* **Decoder prefill:** one pass per request, ~46–50 ms, dominated by GEMM at M=577.
  INT8 weight unpacking costs more than the bandwidth it saves on AVX2, and FP32
  keeps the cross-attention KV cache numerically clean for the 200+ steps that reuse
  it.
* **Embedding table:** a 51,289×768 lookup, not a matmul. Quantizing it would add a
  dequantization step to every token for zero arithmetic gain; the in-memory `.npy`
  slice is already ~0.016 ms per token.

## Hardware adaptation: AVX-VNNI and the Intel/AMD split

All headline numbers in this README were measured on an **AMD Ryzen 5 3600 (Zen 2,
AVX2, no AVX-VNNI)**. That matters for quantization:

* **Zen 2 / older Intel without VNNI:** 8-bit integer dot products in GEMM are
  emulated with 16-bit unpack and multiply instructions. The speedup is small and
  the quantization noise is not. `auto` therefore keeps `encoder_model` in FP32 and
  uses INT4 only for the memory-bound decoder step.
* **Modern Intel (12th/13th/14th Gen, Xeon Scalable) and AMD Zen 4 / EPYC Genoa:**
  AVX-VNNI or AVX-512 VNNI executes 8-bit dot products natively (`vpdpbusd`). On
  those CPUs `auto` upgrades `encoder_model` to `encoder_model.int8.onnx`, which is
  the ~110 ms-per-page win. The upgrade is not free in accuracy: QUInt8 encoder +
  INT4 step measured **0.7922** mean versus **0.7991** with the FP32 encoder on the
  same 50 documents. It stays above the 0.70 floor, and `PRECISION=int8` keeps the
  FP32 encoder plus the 0.8753 INT8-step variant if accuracy matters more than
  decode speed. The decoder step stays mixed INT4 everywhere because it is
  memory-bandwidth bound and does not need VNNI.

`has_avx_vnni()` reads the Linux `/proc/cpuinfo` flags (`avx_vnni` or
`avx512_vnni`). Docker, WSL2, and Docker Desktop on Windows all expose the host CPU
flags there, so the container sees the real silicon. On bare-metal Windows the
function falls back to `platform.processor()` and recognizes 12th/13th/14th Gen,
Core Ultra, Xeon, and EPYC Genoa names. The Windows fallback is a heuristic; set
`PRECISION` explicitly if the host name is ambiguous.

The adaptation is an **`auto` policy only**. `PRECISION=fp32|int8|int4` loads exactly
the variant it names, even on a VNNI CPU. `ALLOW_FP32_FALLBACK=1` lets a missing
quantized file fall back to the FP32 variant instead of failing startup.

## Prefill mode (the one decision that changes output)

The decoder sees a *prefix* before the first generated token, and that prefix is a
real choice, not a spelling. `export/export_onnx.py` records the choice as
`prefill_mode` in `meta.json`; the prefill graph is dynamic in decoder length, so
switching modes is a setting, not a re-export. `PREFILL_MODE` overrides it at
runtime.

| Mode | Prefix | Verdict |
|---|---|---|
| `start` | `decoder_start_token_id` only (1 position) | **default**; matches `model.generate()` |
| `last` | final prompt position only (1 position) | also matches `model.generate()` on the probe image |
| `prompt` | all 10 prompt positions | wrong: emits EOS immediately |
| `prompt_start` | 10 prompt + 1 start (11 positions) | wrong: emits EOS immediately |

Measured on `82837252.png`, `--max-new 64`, `.venv-onnx` (transformers 4.48.3):

```
reference generate(): [2, 0, 5341, 13675, 3935, ...]
start   63 of 64 tokens match, CER 0.0195
last    63 of 64 tokens match, CER 0.0195
prompt   0 of 64 tokens match, CER 1.0000
prompt_start 0 of 64 tokens match, CER 1.0000
```

The single mismatching token is the 64th (cap boundary). `bench/prefill_probe.py`
reruns this comparison against `generate()` for any image.

## Things that are easy to get wrong

**The prompt is not `<OCR>`.** The processor replaces that tag with a sentence and
tokenizes the sentence. The exporter records the resulting ids in `meta.json`; the
server loads them and never tokenizes the tag itself.

**Encoder length is fixed per request.** The step graph broadcasts its attention
mask against the cached cross-attention tensors, so a mask of the wrong length is a
hard error. With a constant OCR prompt this is 577 image tokens + 10 prompt tokens.
Use the same mask for prefill and every step.

**Preprocessing must match exactly.** Resize to 768×768 with bicubic resampling,
rescale to [0, 1], normalize with ImageNet mean/std, arrange as CHW float32. A
different resampling filter or a different channel order produces plausible-looking
but degraded text, not an obvious failure.

**The step graph has no encoder-feature input.** Tracing removes it because the
cached cross-attention makes the values unused — only the mask length survives.
Feeding a padded mask is the failure mode to avoid.

**Decode like `pure_text`.** `<OCR>` is decoded with special tokens left in, then
only the literal `<s>` and `</s>` strings are removed. Do not switch to
`skip_special_tokens=True` or `.strip()`: that also drops special tokens in the
middle of the text and whitespace the reference keeps. See `decode_pure_text` in
`server/ocr_engine.py`.

**No EXIF rotation.** `CLIPImageProcessor` does not rotate by EXIF, so the server
does not either. Honouring the tag looks helpful but moves the pixels away from the
reference the accuracy is measured against.

## Design rationale

### Why five graphs instead of one or three

Florence-2 decomposes naturally into five execution stages with distinct lifecycles
and arithmetic intensities:

* `vision_encoder.onnx`: once per request, compute-bound DaViT GEMM.
* `embed_tokens.onnx`: a lookup table; keeping it separate avoids embedding a
  158 MB matrix inside the decoding hot loop.
* `encoder_model.onnx`: once per request, fuses image tokens with the task prompt.
* `decoder_prefill.onnx`: once per request, builds the initial cross-attention KV
  cache.
* `decoder_step.onnx`: 200–256 times per document, memory-bandwidth-bound GEMV.

Splitting prefill and step avoids the 12 cross-cache passthrough copies incurred by
merged `If` models (measured at 671.8 ms/page in naive implementations) and allows a
different precision strategy per stage. The trade-off is five session loads and a
small amount of Python glue; the measured per-stage timings show the glue is ~2–3%
of the request.

### Why INT4 only in the decoder step

INT4 weights need dynamic bit-unpacking. At M=577 (vision, text encoder, prefill)
the unpacking overhead in software eats the bandwidth savings; on AVX2 without VNNI
it is slower than FP32. At M=1 (decoder step) the kernel is memory-bound, so
streaming 4-bit weights is close to a direct speedup. That is the boundary: **INT4
for decode only, never for prefill or the encoders.**

The LM head is the exception inside the step graph: it stays 8-bit because 4-bit
rounding on 51k vocabulary logits caused rare-token flips over the autoregressive
loop. The mixed graph is therefore 48 four-bit transformer nodes + one eight-bit LM
head.

### Why FP16 is rejected

On x86 CPUs without AVX-512-FP16, ONNX Runtime's CPU provider inserts
`Cast(fp16 → fp32)` before operations and `Cast(fp32 → fp16)` after. FP16 models run
slower than FP32 on these CPUs, and the memory saving is irrelevant at the sizes
that matter here. There is no FP16 variant in `quant.json`.

### Why greedy, no beam, no batching, no cross-request cache

* **Greedy:** the reference is the fp32 original; beam search evaluates three times
  as many decoder states and diverges from the greedy reference. The measured
  decoder advantage comes from doing one row per image, not three.
* **No batching:** the five graphs have no batch axis; adding one needs a re-export,
  and one image already saturates the four CPUs. A second worker was measured:
  identical 0.35 RPS, +77% p50, +73% p95, and 2.90 GiB peak versus 2.50 GiB.
* **No cross-request cache:** the challenge rules forbid response caching; every
  request must run inference.

### Why FastAPI instead of a prebuilt ML server

The rules forbid prebuilt ML serving servers (Triton, TorchServe, etc.). FastAPI is
a web framework, not a model server: it gives us multipart parsing, JSON errors,
async upload handling, and a lifespan hook for warm-up, while the inference process
stays a plain ONNX Runtime `InferenceSession` pool. Uvicorn is pinned to one process
because each process would load its own ~1.5 GB weight set; concurrency is handled by
the bounded thread pool and the queue semaphore instead.

### Why ONNX Runtime 1.31.1

ORT 1.20.1 only shipped the 4-bit `MatMul4BitsQuantizer`; its CPU `MatMulNBits`
kernel rejects 8-bit weights. ORT 1.22 added 8-bit support, so the exporter and
runtime pin 1.31.1. The exporter also needs `onnx_ir` and `ml_dtypes`. The five
graphs use opset 17, which is inside the supported range.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `/models` | where the exported graphs live |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | listen address |
| `PRECISION` | `int8` | `auto`, `fp32`, `int8`, or `int4`; default is accuracy-first INT8 step + FP32 encoders |
| `POSTPROCESS` | `baseline` | `baseline` (transformers whitespace cleanup) or `pure_text` |
| `ALLOW_FP32_FALLBACK` | `0` | set to `1` to fall back to FP32 if a quantized file is missing |
| `ORT_THREADS` | `0` | intra-op threads; `0` sizes from cgroup CPU quota (or lets ORT decide on bare metal) |
| `OCR_WORKERS` | `1` | threads running inference |
| `QUEUE_LIMIT` | `1` | additional requests allowed to wait before fast `503`s |
| `MAX_UPLOAD_BYTES` | `10485760` | uploads above this get `413` |
| `MAX_IMAGE_PIXELS` | `40000000` | decoded pixel cap (40 MP) checked before `image.load()`, returns `413` |
| `MAX_NEW_TOKENS` | from `meta.json` | hard cap on generated tokens |
| `PREFILL_MODE` | from `meta.json` | override the decoder prefix (`start`, `last`, `prompt`, `prompt_start`) |
| `WARMUP` | `1` | run a 2-token warm-up pass across all 5 sessions before `/health` reports `200` |
| `ORT_ALLOW_SPINNING` | `0` | `session.intra_op.allow_spinning`; off wins under a CPU quota |
| `ORT_MEM_ARENA` | `1` | set to `0` to disable ORT's CPU memory arena |
| `KEEP_ALIVE` | `30` | uvicorn keep-alive seconds |

## Reproducing the model without Docker

```bash
python -m venv .venv-export && . .venv-export/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install "transformers==4.48.3" "onnx==1.17.0" "onnxruntime==1.31.1" \
    "onnx_ir" "ml_dtypes" safetensors pillow numpy \
    huggingface_hub timm einops
python export/export_onnx.py --out models
python export/quantize_onnx.py --models models
python export/optimize_decoder.py --models models

python -m venv .venv-serve && . .venv-serve/bin/activate
pip install "onnxruntime==1.31.1" "tokenizers==0.20.3" fastapi uvicorn \
    "pillow" numpy python-multipart
MODEL_DIR=models python -m server.serve
```

`meta.json` in the output directory records the prompt ids, image size and
normalization, encoder length, prefill mode, and token limits — the server reads
all of them from there. `quant.json` records the variant files and quantization
schemes. `embed_tokens.npy` is the in-memory lookup table.

## Tests

```bash
# graph wiring tests use stand-in graphs and need no weights
pytest tests -q

# with a real export: graph I/O, metadata, encoder length, cache contract
MODEL_DIR=models pytest tests -q
```

The stand-in-graph tests run without weights (**52 passed, 6 skipped**); with a real
export the `needs_model` tests run too (**58 passed**). Bad input, decompression
bombs, cache wiring, `pure_text` decoding, EXIF-free preprocessing, precision
manifest resolution, the AVX-VNNI auto-only upgrade, and the load-test
percentile/sustained-rate math all have tests that need no weights.

## Measuring

```bash
# accuracy against reference text, using the challenge's max(0, 1 - CER) rule
python -m bench.accuracy --url http://localhost:8000 --images samples --refs refs.json

# generate fp32 PyTorch references offline across a directory of images
python -m bench.generate_refs --images ~/florence-lab/funsd/dataset/testing_data/images \
    --out ~/florence-lab/references-funsd-fp32.json --limit 50

# sequential p50/p95/p99, closed-loop concurrency sweep, or open-loop arrival rates
python -m bench.load --url http://localhost:8000 --images samples --mode all --timeout 30
python -m bench.load --url http://localhost:8000 --images samples --mode open --rates 0.08,0.1,0.12,0.15 --seconds 120 --timeout 30

# Step 3 resilience: malformed inputs, 64 MP header bomb, overload spike (503), and recovery
python -m bench.stress --url http://localhost:8000 --images samples

# precision sweep: FP32, INT8, and mixed INT4 on the same images
bash bench/precision_sweep.sh

# concurrency probe: batch 1 vs 2 vs 4
python bench/batch_probe.py --url http://localhost:8000 --images samples

# old graphs vs new graphs, token by token
python bench/parity.py --old models-old --new models-new --images samples --max-new 256 --no-torch

# which decoder prefix reproduces model.generate()
python bench/prefill_probe.py --image samples/example.png --max-new 64
```

## Measured numbers

All on CPU inside `docker run --memory=4g --memory-swap=4g --cpus=4` with the
defaults baked into the image: one worker, intra-op threads sized from the cgroup
quota (4 here), spinning off, one waiting slot, `PRECISION=auto` (mixed INT4 step +
FP32 encoders, in-memory embed table), `POSTPROCESS=baseline`. The host is an AMD
Ryzen 5 3600 (Zen 2, no AVX-VNNI).

| Measurement | Result |
|---|---|
| Sequential latency (50 FUNSD images) | **p50 3.43 s**, p95 **3.68 s**, max 3.81 s, 0 errors |
| Per-stage p50 | vision **1,592.3 ms**, text encoder **193.7 ms**, prefill **49.1 ms**, decode **1,527.1 ms**, embed **4.2 ms** |
| Decode throughput | p50 **162.3 tok/s**, mean **162.2** (up to 180+ on dense pages) |
| Total throughput | p50 **72.7 tok/s**, mean **69.3** |
| Open-loop sustained rate | **0.20/0.25/0.30/0.35 req/s all at 100% success** |
| Accuracy vs local references | **0.7991 mean** (CER-scored, 50 FUNSD images; fp32 proxy 0.8769) |
| Accuracy-first fallback | `PRECISION=int8`: **0.8753 mean**, 135.2 tok/s, p50 3.73 s |
| Two workers x 2 threads | same 0.35 RPS, +77% p50 (5.57 s vs 3.14 s), 2.90 GiB peak; rejected |
| Overload spike (10 clients) | fast 503s, max 6.1 ms, 0 500s, recovery clean |
| Container build | torch absent at runtime; `quant.json` + 10 model files verified |
| v1 baseline (tag `v1-fp32`) | p50 8.29 s, 9.35 s p95, 0.15 RPS, 3.36 GiB peak, 0.939 accuracy |

The organizer's baseline is the authoritative accuracy reference; the local numbers
above come from local fp32 references on FUNSD and are a regression canary, not a
score. V1 (fp32, all five graphs FP32) was tagged `v1-fp32` before the V2 work.

## References

* ONNX format, operators, and opsets: https://onnx.ai/onnx/
* ONNX Runtime thread management: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
* ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
* FastAPI: https://fastapi.tiangolo.com/
* Uvicorn: https://www.uvicorn.org/
* NVIDIA, "Mastering LLM Techniques: Inference Optimization": https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
* Florence-2 paper: https://arxiv.org/abs/2311.06242 · model card: https://huggingface.co/microsoft/Florence-2-base

## Known issues

* Official accuracy against the organizer's baseline is not available; the local
  canary is a regression signal, not a score.
* The organizer's CPU count is not fixed yet. Every container number above is at
  `--cpus=4`; the thread count is sized from the cgroup quota at startup, so a
  different quota changes the pool size without a rebuild.
* The INT4 default (0.7991 mean) is above the challenge's 0.70 mean floor but below
  the earlier internal 0.90 mean / 0.75 per-image gate. If the hidden set is denser
  than FUNSD, `PRECISION=int8` (0.8753 mean) is the accuracy-first fallback.
* One worker peaks at 2.39 GiB of the 4 GiB limit in the c10 stress test; a 0.35 RPS
  re-check measured 2.50 GiB. A second worker x 2 threads measured the same 0.35 RPS
  at +77% p50 and 2.90 GiB peak, so throughput cannot be raised by adding workers on
  this CPU count.
* The AVX-VNNI upgrade is automatic only for `PRECISION=auto`. On VNNI the QUInt8
  encoder costs about 0.007 mean accuracy (0.7991 → 0.7922 with the INT4 step);
  `PRECISION=int8` keeps the FP32 encoder and 0.8753 mean accuracy. The bare-metal
  Windows fallback is a processor-name heuristic; set `PRECISION` explicitly if the
  host name is ambiguous.
* The 40 MP decoded-pixel cap is a memory-safety trade-off. A valid upload under
  10 MB that decodes to more than 40 MP would get a 413; FUNSD-like pages are 1–2 MP.

Numbers from every change, including the ones that were reverted, are in
`journal.md`.
