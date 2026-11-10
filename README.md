# Florence-2 OCR on CPU, ONNX Runtime only

Deployment of `microsoft/Florence-2-base` behind a small HTTP API, running on CPU
with no PyTorch in the serving image. The model is exported from the original
weights by `export/` and served by `server/`.

* `journal.md` — development log, experiments, numbers, and dead ends.
* `export/export_onnx.py` — turns the original checkpoint into five ONNX graphs.
* `server/` — FastAPI app plus the ONNX Runtime decoding loop.
* `bench/` — CER accuracy scoring and latency/throughput measurement.
* `Dockerfile` — two stages: PyTorch only to export, then a runtime with none.

## Build and run

```bash
docker build -t florence2-ocr .
docker run --rm -p 8000:8000 --memory=4g --memory-swap=4g florence2-ocr
```

The image exports the model during the build. Nothing pre-converted is downloaded;
the only input is the original `microsoft/Florence-2-base` checkpoint, pinned to a
revision recorded in `export/export_onnx.py` and in the generated `meta.json`.

```bash
curl -F "image=@samples/example.png" http://localhost:8000/ocr
curl http://localhost:8000/health
```

## How the model runs

Five graphs, because the pieces have genuinely different lifetimes:

| Graph | Called | Why separate |
|---|---|---|
| `vision_encoder.onnx` | once per request | the 768×768 DaViT pass dominates prefill and never changes |
| `embed_tokens.onnx` | once per prompt, once per token | a 51k×768 lookup is cheaper as its own graph than folded into decoding |
| `encoder_model.onnx` | once per request | fuses the 577 image tokens with the 10 prompt tokens |
| `decoder_prefill.onnx` | once per request | returns logits plus the whole attention cache |
| `decoder_step.onnx` | once per token | takes token ids, returns logits plus 12 updated self-attention tensors |

The cross-attention key/value tensors are built in prefill and reused unchanged for
the entire request, which is why the step graph accepts 24 cached tensors and
returns only 12: it has nothing new to say about the image.

Decoding is greedy (`argmax`) with a cap of `MAX_NEW_TOKENS` (default 256, from
`meta.json`) and stops at EOS. Greedy is deliberate: the reference is the fp32
original, and sampling would add variance without improving CER.

Prefill runs on **one** decoder position, not ten. In a causal decoder only the last
prompt position can predict the next token, and its hidden state has already
attended to everything before it, so encoding ten positions and keeping one row of
logits buys nothing. This is the same trick TensorRT-LLM and vLLM use: a prefill
that outputs one set of logits and a KV cache, then a token-at-a-time decode loop.

## Things that are easy to get wrong

**The prompt is not `<OCR>`.** The processor replaces that tag with a sentence and
tokenizes the sentence. The exporter records the resulting ids in `meta.json`; the
server loads them and never tokenizes the tag itself.

**Encoder length is fixed per request.** The step graph broadcasts its attention
mask against the cached cross-attention tensors, so a mask of the wrong length is a
hard error. With a constant OCR prompt this is 577 image tokens + 10 prompt tokens.

**Prefill position.** The prefill graph expects the embedding of the *last prompt
token* (`decoder_input_embeds`, shape `(1,1,768)`). The server slices it from the
prompt embeddings it already computed.

**Preprocessing must match exactly.** Resize to 768×768 with bicubic resampling,
rescale to [0, 1], normalize with ImageNet mean/std, arrange as CHW float32. A
different resampling filter or a different channel order produces plausible-looking
but degraded text, not an obvious failure.

## Optimizations in the current build

Each is a deliberate choice with a stated reason; the measured effect of each lives
in `journal.md`, because that is where the before/after numbers are kept.

| Optimization | Where | Why |
|---|---|---|
| Prefill on one position, not the whole prompt | `server/ocr_engine.py` | 10x less decoder compute for an identical next token |
| Embedding lookup folded into the step graph | `export/export_onnx.py` | one session call and one array allocation less per token |
| Split graphs with per-graph ORT optimization | `export/`, `server/ocr_engine.py` | ORT optimizes each graph for the shapes it actually sees |
| Cached decoding, cross-attention computed once | graph I/O contract | avoids recomputing 587 positions of cross-attention per token |
| Oracle-free prompt ids (`meta.json`) | `export/export_onnx.py` | removes per-request tokenization and a whole class of silent error |
| Greedy `argmax`, no beam | `server/ocr_engine.py` | matches the fp32 baseline; beam search would diverge from it |
| Stop at EOS | `server/ocr_engine.py` | most images finish well below the cap |
| Bounded queue with immediate `503` | `server/app.py` | keeps memory flat under overload instead of OOMing |
| Eager encoder-length check | `server/ocr_engine.py` | turns a mid-decode broadcast error into a clear startup failure |
| Multi-stage build, torch-free runtime | `Dockerfile` | rule compliance, and a smaller image than a single-stage one |

Not adopted, with reasons:

| Idea | Why not |
|---|---|
| Dynamic INT8 on the language graphs | token drift (7/64 and 9/64 agreement on two images); accuracy floor is absolute |
| INT8 on the vision tower | `ConvInteger` on DaViT convs is not supported by the CPU EP here, and it fails to load |
| Beam search | the reference is greedy; different tokens means worse CER |
| Batching | no batch axis in the graphs; would cost every request to help only under load, and memory is tight |
| Cross-request KV reuse | the rules forbid response caching; every request must run inference |
| `OpenMP`/`OMP_NUM_THREADS` tuning | ONNX Runtime's intra-op pool governs this, so tune `ORT_THREADS` instead |
| Multiple uvicorn workers | each process loads its own ~1.4 GiB of weights; measure `VmHWM` before trying two |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `/models` | where the exported graphs live |
| `ORT_THREADS` | `0` | intra-op threads, `0` lets ONNX Runtime decide |
| `OCR_WORKERS` | `1` | threads running inference |
| `QUEUE_LIMIT` | `8` | additional requests allowed to wait before fast `503`s |
| `MAX_UPLOAD_BYTES` | `10485760` | uploads above this get `413` |
| `MAX_NEW_TOKENS` | from `meta.json` | hard cap on generated tokens |

## Reproducing the model without Docker

```bash
python -m venv .venv-export && . .venv-export/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install "transformers==4.48.3" safetensors pillow numpy huggingface_hub
python export/export_onnx.py --out models
pip install "onnxruntime==1.20.1" "tokenizers==0.20.3" fastapi uvicorn \
    "pillow" numpy python-multipart
MODEL_DIR=models python -m server.serve
```

## Measuring

```bash
# accuracy against reference text, using the challenge's max(0, 1 - CER) rule
python -m bench.accuracy --url http://localhost:8000 --images samples --refs refs.json

# sequential p50/p95/p99, then concurrency sweep
python -m bench.load --url http://localhost:8000 --images samples --mode all
```

Numbers from every change, including the ones that were reverted, are in
`journal.md`.
