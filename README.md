# Florence-2 OCR on CPU, ONNX Runtime only

Deployment of `microsoft/Florence-2-base` behind a small HTTP API, running on CPU
with no PyTorch in the serving image. The model is exported from the original
weights by `export/` and served by `server/`.

Target constraints: **4 GB RAM, CPU only, PyTorch forbidden at serving time.**

* `journal.md` — development log, experiments, numbers, and dead ends.
* `export/export_onnx.py` — turns the original checkpoint into five ONNX graphs.
* `server/` — FastAPI app plus the ONNX Runtime decoding loop.
* `bench/` — CER accuracy scoring, latency/throughput measurement, the prefill
  probe, and old-vs-new graph parity.
* `runall.sh` — one long evidence-collecting run (export, parity, tests, server,
  bench, Docker).
* `Dockerfile` — two stages: PyTorch only to export, then a runtime with none.

## Status at this commit

| Item | State |
|---|---|
| Export from pinned weights | verified: 5 graphs, **1,377.9 MiB**, `577 + 10 = 587` encoder length |
| Token parity, old vs new graphs | **PASS**: 5/5 images token-identical (225/154/201/256/256) |
| Prefill prefix | `start` matches `model.generate()`; `prompt`/`prompt_start` do not |
| Unit + graph tests | **27 passed, 6 skipped** without weights; **33 passed** with a real export; 0 errors |
| Server (v3 graphs) | ready in 6 s; OCR 6.7 s total on 12 threads; `VmHWM ≈ 3.07 GiB` |
| Docker under `--memory=4g --cpus=4` | build and run verified: healthy in 4 s, OCR 10.3 s, 3.12 GiB / 4 GiB, torch absent |
| Official accuracy vs organizer baseline | not available yet |

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
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install "transformers==4.48.3" "onnx==1.17.0" "safetensors" \
       "pillow" "numpy" "huggingface_hub" "timm" "einops"
```

Installing CPU `torch` alone lets `timm` pull a mismatched PyPI `torchvision`, and
the export dies with `RuntimeError: operator torchvision::nms does not exist`. The
pair has to come from the same index.

## How the model runs

Five graphs, because the pieces have genuinely different lifetimes:

| Graph | Called | Why separate |
|---|---|---|
| `vision_encoder.onnx` | once per request | the 768×768 DaViT pass dominates prefill and never changes |
| `embed_tokens.onnx` | once per prompt, once per token | a 51k×768 lookup is cheaper as its own graph than folded into decoding |
| `encoder_model.onnx` | once per request | fuses the 577 image tokens with the 10 prompt tokens |
| `decoder_prefill.onnx` | once per request | returns logits plus the whole attention cache |
| `decoder_step.onnx` | once per token | takes the previous token's embedding, returns logits plus 12 updated self-attention tensors |

The cross-attention key/value tensors are built in prefill and reused unchanged for
the entire request, which is why the step graph accepts 24 cached tensors and
returns only 12: it has nothing new to say about the image.

Decoding is greedy (`argmax`) with a cap of `MAX_NEW_TOKENS` (default 256, from
`meta.json`) and stops at EOS. Greedy is deliberate: the reference is the fp32
original, and sampling would add variance without improving CER.

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

## Optimizations in the current build

| Optimization | Where | Why |
|---|---|---|
| Prefill on one decoder position | `export/`, `server/ocr_engine.py` | 10x less decoder compute for an identical next token |
| `prefill_mode` as a recorded setting | `export/`, `server/ocr_engine.py` | the correct prefix is evidence, not folklore |
| Split graphs with per-graph ORT optimization | `export/`, `server/ocr_engine.py` | ORT optimizes each graph for the shapes it actually sees |
| Cached decoding, cross-attention computed once | graph I/O contract | avoids recomputing 587 positions of cross-attention per token |
| Prompt ids baked into `meta.json` | `export/export_onnx.py` | removes per-request tokenization and a class of silent error |
| Greedy `argmax`, no beam | `server/ocr_engine.py` | matches the fp32 baseline; beam search would diverge from it |
| Stop at EOS | `server/ocr_engine.py` | most images finish well below the cap |
| Bounded queue with immediate `503` | `server/app.py` | keeps memory flat under overload instead of OOMing |
| Eager encoder-length check | `server/ocr_engine.py` | turns a mid-decode broadcast error into a clear startup failure |
| `pure_text` decode matching the processor | `server/ocr_engine.py` | keeps the served text on the reference's post-processing path |
| No EXIF rotation | `server/ocr_engine.py` | matches `CLIPImageProcessor` byte for byte |
| Multi-stage build, torch-free runtime | `Dockerfile` | rule compliance, and a smaller image than a single-stage one |

Tried and reverted: folding the embedding lookup into the step graph duplicated the
150.3 MiB embedding matrix inside `decoder_step.onnx` (1,528.1 MiB total) for a
negligible per-token saving, and broke the step graph's I/O contract.

Not adopted, with reasons:

| Idea | Why not |
|---|---|
| Dynamic INT8 on the language graphs | ~2x faster decode, but token drift (7/64 and 9/64 agreement on two images); accuracy floor is absolute |
| INT8 on the vision tower | `ConvInteger` on DaViT convs is not supported by the CPU EP here, and it fails to load |
| Beam search | the reference is greedy; different tokens means worse CER |
| Batching | no batch axis in the graphs; would cost every request to help only under load, and memory is tight |
| Cross-request KV reuse | the rules forbid response caching; every request must run inference |
| `OMP_NUM_THREADS` tuning | ONNX Runtime's intra-op pool governs this, so tune `ORT_THREADS` instead |
| Multiple uvicorn workers | each process loads its own ~1.4 GiB of weights; one worker already reaches ~3.07 GiB peak |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `/models` | where the exported graphs live |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | listen address |
| `ORT_THREADS` | `0` | intra-op threads, `0` lets ONNX Runtime decide |
| `OCR_WORKERS` | `1` | threads running inference |
| `QUEUE_LIMIT` | `8` | additional requests allowed to wait before fast `503`s |
| `MAX_UPLOAD_BYTES` | `10485760` | uploads above this get `413` |
| `MAX_NEW_TOKENS` | from `meta.json` | hard cap on generated tokens |
| `PREFILL_MODE` | from `meta.json` | override the decoder prefix (`start`, `last`, `prompt`, `prompt_start`) |
| `KEEP_ALIVE` | `30` | uvicorn keep-alive seconds |

## Reproducing the model without Docker

```bash
python -m venv .venv-export && . .venv-export/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install "transformers==4.48.3" "onnx==1.17.0" safetensors pillow numpy \
    huggingface_hub timm einops
python export/export_onnx.py --out models

python -m venv .venv-serve && . .venv-serve/bin/activate
pip install "onnxruntime==1.20.1" "tokenizers==0.20.3" fastapi uvicorn \
    "pillow" numpy python-multipart
MODEL_DIR=models python -m server.serve
```

`meta.json` in the output directory records the prompt ids, image size and
normalization, encoder length, prefill mode, and token limits — the server reads
all of them from there.

## Tests

```bash
# graph wiring tests use stand-in graphs and need no weights
pytest tests -q

# with a real export: graph I/O, metadata, encoder length, cache contract
MODEL_DIR=models pytest tests -q
```

The stand-in-graph tests run without weights (27 passed, 6 skipped); with a real
export the `needs_model` tests run too (33 passed). Bad input, decompression bombs,
cache wiring, `pure_text` decoding, EXIF-free preprocessing, and the load-test
percentile/sustained-rate math all have tests that need no weights.

## Measuring

```bash
# accuracy against reference text, using the challenge's max(0, 1 - CER) rule
python -m bench.accuracy --url http://localhost:8000 --images samples --refs refs.json

# sequential p50/p95/p99, then concurrency sweep; each step also reports
# success_rate and sustained_rps (successes only, zero below 99% success)
python -m bench.load --url http://localhost:8000 --images samples --mode all

# old graphs vs new graphs, token by token
python bench/parity.py --old models-old --new models-new --images samples --max-new 256 --no-torch

# which decoder prefix reproduces model.generate()
python bench/prefill_probe.py --image samples/example.png --max-new 64
```

## Measured numbers

All on CPU, fp32, 12 ONNX Runtime threads unless stated otherwise.

| Measurement | Result |
|---|---|
| Export size | 1,377.9 MiB (vision 349.3, embed 150.3, encoder 165.3, prefill 370.0, step 343.0) |
| Graph parity vs fp32 PyTorch | vision/embed/encoder ~1e-6 relative L2; decoder logits ~2e-5 max abs |
| Token parity, old vs new export | 5/5 images identical (225/154/201/256/256 tokens) |
| Single OCR, 82837252 | encoder 2.07 s, decode 4.64 s, total 6.71 s |
| Server peak RSS (`VmHWM`) | 3.07 GiB, one worker |
| Sequential latency (150 requests) | p50 7.21 s, p95 8.58 s, p99 13.25 s, 0 errors |
| Concurrency 1/2/4/8 | p50 6.86 / 14.48 / 21.95 / 37.37 s, 0 errors or 503s |
| Accuracy canary vs local references | mean 0.9413 (CER-scored, 5 images) |
| Container under `--memory=4g --cpus=4` | healthy in 4 s; OCR 10.33 s (encoder 3.21 s, decode 7.12 s); 3.12 GiB / 4 GiB; `torch` and `torchvision` absent |

The organizer's baseline is the authoritative accuracy reference; the local numbers
above come from `model.generate()` on FUNSD and are a regression canary, not a
score.

## References

* ONNX format, operators, and opsets: https://onnx.ai/onnx/
* ONNX Runtime thread management: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
* ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
* NVIDIA, "Mastering LLM Techniques: Inference Optimization": https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
* Florence-2 paper: https://arxiv.org/abs/2311.06242 · model card: https://huggingface.co/microsoft/Florence-2-base

## Known issues

* Official accuracy against the organizer's baseline is not available; the local
  `0.9413` canary is a regression signal, not a score.
* RPS was not measured. The concurrency numbers above are p50 latency at a fixed
  concurrency, not the throughput metric the challenge scores.
* The challenge tests the latest commit on `main`. This work currently lives on
  `challenge-hardening` / `firstproto` with a PR to `dev`, so `main` must be
  fast-forwarded before submission.

Numbers from every change, including the ones that were reverted, are in
`journal.md`.
