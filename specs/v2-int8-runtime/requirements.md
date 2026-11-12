# Requirements Document

## Introduction

Version 2 of the Florence-2 CPU OCR service. V1 is already submission-safe: 7.60 s per
full page, p50 8.29 s / p95 9.35 s over 50 real documents, 0.15 req/s sustained at
100% success, 3.36 GiB peak of 4 GiB, and 0.939 mean accuracy against local fp32
references. V2 keeps that API and safety envelope but attacks the two things that
still cost points: token decode time and the accuracy gap caused by post-processing.

The work is measured, not assumed. Every optimization is adopted only if it survives
the 50-image accuracy set and the official container limits. V1 stays available as a
tag and as a runtime fallback while V2 is evaluated.

## Requirements

### Requirement 1

**User Story:** As the challenge submitter, I want the served text to match the
organizer's baseline post-processing, so that whitespace differences do not silently
cost accuracy points.

#### Acceptance Criteria

1. WHEN `POSTPROCESS=pure_text` THEN the server SHALL reproduce V1 text byte for byte.
2. WHEN `POSTPROCESS=baseline` THEN the server SHALL apply the same whitespace cleanup
   as transformers' `clean_up_tokenization_spaces=True` before returning text.
3. WHEN scored against the 50 local fp32 references THEN the default post-processing
   mode SHALL achieve a mean `max(0, 1 - CER)` of at least 0.95.
4. IF a post-processing mode lowers the 50-image mean below the other mode THEN it
   SHALL NOT become the image default.

### Requirement 2

**User Story:** As the challenge submitter, I want lower per-request latency, so that
Step 1 scores improve without approaching the 30 s cutoff.

#### Acceptance Criteria

1. WHEN serving 50 real FUNSD images at `--cpus=4` THEN p50 latency SHALL be at most
   5.0 s and p95 at most 6.5 s.
2. WHEN any single request is served THEN total latency SHALL remain below 30 s.
3. WHEN the engine reports timings THEN it SHALL break the request into encoder,
   prefill, and decode time so a regression can be localized.

### Requirement 3

**User Story:** As the challenge submitter, I want higher sustained throughput, so that
Step 2 scores improve at the same success rate.

#### Acceptance Criteria

1. WHEN measured open-loop at `--cpus=4` THEN the server SHALL sustain at least
   0.25 req/s with a success rate of at least 99%.
2. WHEN arrival rate exceeds capacity THEN excess requests SHALL receive a fast 503
   rather than a 30 s timeout.
3. WHEN throughput is measured THEN the harness SHALL report `target_rps`,
   `attempted_rps`, `rps`, and `sustained_rps` using the existing 99% rule.

### Requirement 4

**User Story:** As the challenge submitter, I want lower memory use, so that the image
stays safely inside the 4 GB limit under load and leaves room for future concurrency.

#### Acceptance Criteria

1. WHEN one request runs at `--memory=4g --cpus=4` THEN peak `VmHWM` SHALL be at most
   3.0 GiB.
2. WHEN the overload spike from `bench.stress` runs THEN the container SHALL NOT be
   OOM-killed and SHALL NOT return any 500.
3. WHEN quantized graphs are loaded THEN the runtime SHALL report the per-graph
   precision at startup.

### Requirement 5

**User Story:** As the challenge submitter, I want quantization to preserve accuracy,
so that a speed win cannot turn into a failed accuracy floor.

#### Acceptance Criteria

1. WHEN encoders are quantized THEN only `MatMul` nodes SHALL be converted, using
   dynamic per-call activation quantization and int8 weights.
2. WHEN decoder graphs are quantized THEN `decoder_step` SHALL use weight-only
   `MatMulNBits` with 8-bit weights, block size 32, symmetric, `accuracy_level=4`, on
   ONNX Runtime 1.31.1 or newer, and `decoder_prefill` SHALL remain fp32. `decoder_step`
   SHALL NOT use 4-bit weights unless the mixed variant in 5.6 passes its gate, and
   `decoder_prefill` SHALL NOT use 4-bit weights at all.
3. WHEN `embed_tokens` is considered THEN it SHALL remain fp32, because a gather has
   no MatMul to quantize and the table is read sparsely.
4. WHEN quantized graphs are scored on the 50 local references THEN mean accuracy
   SHALL be at least 0.90 and no single image SHALL fall below 0.75.
5. IF the quantized build misses either bound THEN the image default SHALL remain
   fp32 and the experiment SHALL be recorded as rejected.
6. WHEN the mixed 4-bit step variant is evaluated THEN it SHALL quantize the
   transformer weights to 4 bits and keep the LM head at 8 bits, both with block size
   32, symmetric, `accuracy_level=4`; it SHALL meet the 5.4 accuracy gate and improve
   decode tokens per second by at least 10% over the 8-bit step graph before it can
   become the default.
7. fp16 weights or activations SHALL NOT be used in any served graph, because the CPU
   execution provider inserts fp16↔fp32 casts and does not accelerate them.

### Requirement 6

**User Story:** As the challenge submitter, I want the decoder step graph to stop
recomputing constant work, so that every generated token is cheaper.

#### Acceptance Criteria

1. WHEN the prefill graph emits cross-attention keys THEN they SHALL already be stored
   transposed as `(batch, heads, head_dim, enc_seq)`.
2. WHEN the step graph consumes a cross-attention key THEN it SHALL NOT contain a
   `Transpose` node for that tensor.
3. WHEN graph surgery is applied THEN `onnx.checker` SHALL accept the result and a
   numerical test on the tiny model SHALL show logits equal to the unsurgered graph
   within `1e-4`.
4. WHEN surgery is applied to both fp32 and int8 graphs THEN the same test SHALL pass
   for both.

### Requirement 7

**User Story:** As the challenge submitter, I want a reproducible build, so that the
organizer can rebuild the exact image from the original weights.

#### Acceptance Criteria

1. WHEN the image is built THEN quantization and graph surgery SHALL run in the
   exporter stage from the original `microsoft/Florence-2-base` checkpoint.
2. WHEN the build finishes THEN the runtime image SHALL contain no `torch` or
   `torchvision`, and no pre-quantized weight downloaded from outside the build.
3. WHEN the build writes model files THEN it SHALL also write `quant.json` recording
   the fp32, int8, and int4 variants, each graph's file, scheme, bit width, block size,
   and the ONNX Runtime version.
4. WHEN `PRECISION` is `fp32`, `int8`, or `int4` THEN the server SHALL load that
   variant's files; `fp32` SHALL behave like V1.

### Requirement 8

**User Story:** As the challenge submitter, I want the public API unchanged, so that
V2 cannot break the scored contract.

#### Acceptance Criteria

1. WHEN `GET /health` is called after startup THEN it SHALL return 200
   `{"status":"ok"}`.
2. WHEN `POST /ocr` receives a valid multipart `image` THEN it SHALL return 200
   `{"text": ...}`.
3. WHEN the `image` field is missing, empty, or not a readable image THEN the server
   SHALL return a 4xx JSON error.
4. WHEN the upload exceeds 10 MB or decodes above 40 MP THEN the server SHALL return
   413 JSON.
5. WHEN the queue is full THEN the server SHALL return 503 JSON without waiting.

### Requirement 9

**User Story:** As the challenge submitter, I want request execution to stay
sequential until the optimizations are proven, so that memory and latency stay
predictable.

#### Acceptance Criteria

1. WHEN V2 ships THEN `OCR_WORKERS` SHALL default to 1 and `QUEUE_LIMIT` to 1.
2. WHEN batching is evaluated THEN it SHALL be a separate experiment after all other
   tasks are complete.
3. IF batch size 2 or 4 does not raise sustained RPS by at least 15% with p95 below
   30 s and peak below 3.8 GiB THEN batching SHALL remain disabled.

### Requirement 10

**User Story:** As the challenge submitter, I want every claimed win backed by a
measurement, so that the journal and README stay truthful.

#### Acceptance Criteria

1. WHEN a configuration is adopted THEN its 50-image accuracy, p50/p95, sustained RPS,
   tokens per second, per-stage timings, and peak memory SHALL be recorded in
   `journal.md`.
2. WHEN a configuration is rejected THEN the measured reason SHALL be recorded.
3. WHEN V2 is finished THEN `README.md` SHALL show the new numbers, the precision
   defaults, and the graph-split rationale.

### Requirement 11

**User Story:** As the challenge submitter, I want tokens per second and per-stage
timings exposed per request and aggregated by the bench, so that a speed change can be
attributed to the graph that caused it.

#### Acceptance Criteria

1. WHEN `POST /ocr` returns 200 THEN `timings` SHALL include `vision_ms`,
   `text_encoder_ms`, `prefill_ms`, `decode_ms`, `total_ms`, `steps`, `tokens`,
   `decode_tokens_per_s`, and `total_tokens_per_s`.
2. WHEN timings are defined THEN `decode_tokens_per_s` SHALL equal
   `steps * 1000 / decode_ms`, and `total_tokens_per_s` SHALL equal
   `tokens * 1000 / total_ms`.
3. WHEN instrumentation is enabled THEN its overhead SHALL be less than 0.5% of p50
   latency.
4. WHEN `bench.load` runs in latency or open-loop mode THEN it SHALL report mean,
   p50, and p95 of `decode_tokens_per_s` and of each per-stage timing.
5. WHEN a config is recorded in the journal THEN its tokens-per-second and per-stage
   breakdown SHALL be included next to its p50 and RPS.
