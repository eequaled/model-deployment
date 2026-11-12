# Report — Florence-2 OCR on CPU

## What I built

A CPU-only OCR backend for Florence-2-base behind a FastAPI server using ONNX Runtime. No PyTorch at serve time, no prebuilt ML server, no response caching. 4 GB RAM, CPU only.

## The 5 graphs and why

1. **Vision encoder** — DaViT, 768×768 → 577 visual tokens. FP32. Runs once per request. 1.59s.
2. **Embed tokens** — 51,289×768 lookup table. In-memory NumPy slicing. 0.016ms/token.
3. **Text encoder** — BART-style, 6 layers, 12 heads. Fuses 577 image + 10 prompt tokens = 587 positions. FP32 (or QUInt8 on VNNI). 0.19s.
4. **Decoder prefill** — 1 position (start token). Returns logits + 24 cache tensors. FP32. 0.049s.
5. **Decoder step** — one token per call, 200-256 times. Mixed INT4. 1.53s.

## The 3 precision tiers

- **FP32** — vision, prefill, embedding (accuracy-sensitive, run once)
- **INT8 step** — weight-only MatMulNBits, 8-bit, 104MB, 135 tok/s, 0.8753 accuracy
- **Mixed INT4 step** — 48×4-bit transformer + 1×8-bit LM head, 79MB, 162 tok/s, 0.7991 accuracy
- **QUInt8 text encoder** — dynamic per-channel, 46MB, only activates on AVX-VNNI CPUs

## Key optimizations

1. **Cross-attention KV cache** — image cross-attention computed once in prefill, reused unchanged for all 200+ steps. Step graph only returns 12 self-attention tensors, not 24.
2. **One-position prefill** — decoder starts from the start token (1 position), not all 10 prompt positions. Feeding all 10 gives EOS immediately.
3. **Decoder transpose surgery** — prefill emits cross-attention keys already transposed (perm [0,2,3,1]), step drops 6 per-token Transpose nodes. Logits match within 1e-4.
4. **In-memory embedding table** — sliced directly from `embed_tokens.npy` instead of calling ONNX 255 times. 14.2ms → 4.0ms. Exact match.
5. **AVX-VNNI detection** — reads `/proc/cpuinfo` flags. On Intel VNNI, text encoder upgrades to QUInt8 (~110ms faster). Auto-only policy, explicit precision never overridden.
6. **FastAPI + bounded queue** — thread pool for blocking ORT calls, semaphore with 2 slots, fast 503 under overload. 6.1ms worst-case refusal, no OOM.
7. **Cgroup-aware thread sizing** — sizes intra-op threads from Docker's CPU quota. 4 threads at `--cpus=4`.
8. **Spinning off** — `allow_spinning=0` won that 0.15 RPS step (9/9 vs 6/7).
9. **Warm-up** — 2-token generation touches all 5 sessions before `/health` returns 200.
10. **Baseline post-processing** — `clean_up_tokenization_spaces` matching transformers default (0.8769 vs 0.8767).
11. **EXIF-free preprocessing** — RGB, bicubic 768×768, rescale, normalize, CHW. Byte-exact with CLIPImageProcessor.
12. **Multi-stage torch-free Docker build** — PyTorch only in exporter stage, runtime has none.

## Numbers

- 50 FUNSD images: 0.8753 mean accuracy (INT8), 0.7991 (INT4)
- p50: 3.73s (INT8), 3.43s (INT4)
- Decode: 135.2 tok/s (INT8), 162.3 tok/s (INT4)
- Peak memory: 2.39-2.50 GiB
- Sustained RPS: 0.30 (INT8), 0.35 (INT4)
- Tests: 52 passed / 6 skipped (no weights), 58 passed (with weights)

## What I rejected and why

- Dynamic INT8 on language graphs → token drift
- INT8 vision encoder → ConvInteger unsupported + 0.7163 accuracy
- Beam search → diverges from greedy reference
- Batching → no batch axis, saturates CPU
- Multiple workers → same RPS, +77% p50, more memory
- FP16 → CPU inserts casts, slower
- Triton → forbidden + inherits ORT + HTTP per step
- Response caching → forbidden
- 10-position prefill → EOS immediately

## The precision decision

**INT8 as default.** INT4 is faster (162 vs 135 tok/s) but lower accuracy (0.7991 vs 0.8753). The elimination floor is 0.70. INT4 has a min of 0.0603 on one image. INT8 gives 0.18 margin above the floor. INT4 gives 0.10. The speed difference is ~5-8 points of score, but elimination costs everything. Accuracy-first is the safer bet.

## What I'd do next

- Test on the organizer's official CPU and references
- Try OpenVINO on Intel (could be 10-20% faster)
- Try a better INT4 algorithm (AWQ/GPTQ) for the decoder
- Revisit batching if the CPU allocation grows beyond 4
