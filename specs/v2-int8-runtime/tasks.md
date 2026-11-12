# Implementation Plan

- [x] 1. Snapshot the V1 baseline
  - Tagged commit `5de8519` as `v1-fp32` and created branch `v2-int8-runtime`
  - Recorded comparison row: 50-image p50 8.29 s / p95 9.35 s, 0.15 RPS sustained at
    100%, 3.36 GiB peak, 0.939 mean accuracy, 54.55 decode tok/s
  - _Requirements: 10.1_

- [x] 2. Add baseline-exact post-processing
  - [x] 2.1 Implement `decode_text(mode)` with `pure_text` and `baseline` in
    `server/ocr_engine.py`; the baseline mode applies the transformers
    `clean_up_tokenization_spaces` replacements after removing `<s>`/`</s>`
    - Added unit tests in `tests/test_postprocess.py` covering spaces before
      `. , ? !`, contractions, and mode validation
    - _Requirements: 1.1, 1.2_
  - [x] 2.2 Score all 50 FUNSD images under both modes against
    `references-funsd-50.json`; record per-image worst and mean
    - Measured on 50 images: `pure_text` mean acc 0.8767, `baseline` mean acc 0.8769,
      both min acc 0.1299; `baseline` is slightly better on CER (0.1231 vs 0.1233)
    - Wrote `bench/postprocess_eval.py`
    - _Requirements: 1.3, 1.4_

- [x] 3. Build the quantized graphs
  - [x] 3.0 Resolve the ONNX Runtime version for weight-only quantization
    - Resolved: **1.31.1** installed and verified against the existing fp32 graphs
      (same text, 7.0 s load). 1.20.1 shipped only 4-bit; 1.22 added 8-bit
    - The exporter and runtime both pin 1.31.1; the exporter also installs `onnx_ir`
      and `ml_dtypes`
    - _Requirements: 5.2, 7.1, 7.3_
  - [x] 3.1 Write `export/quantize_onnx.py`: dynamic int8 on `MatMul` only for
    `vision_encoder` and `encoder_model`, `embed_tokens` copied as fp32
    - Fail if a graph converts zero nodes
    - _Requirements: 5.1, 5.3_
  - [x] 3.2 Extend the script with `MatMulNBitsQuantizer(bits=8, block_size=32,
    is_symmetric=True, accuracy_level=4)` for `decoder_step`, written as
    `decoder_step.int8.onnx`; copy `decoder_prefill` as fp32
    - Print source and target size per graph
    - _Requirements: 5.2_
  - [x] 3.3 Emit `quant.json` with the three variants, file names, schemes, bits,
    block size, and ORT version; add a test that every listed file exists and matches
    the on-disk size
    - _Requirements: 5.3, 7.3_
  - [x] 3.4 Build the mixed 4-bit step variant: pass one 4-bit excluding
    `/language_model/lm_head/MatMul`, pass two 8-bit on the remaining MatMul; write
    `decoder_step.int4.onnx`
    - Assert 48 four-bit `MatMulNBits` nodes and one eight-bit LM-head node
    - _Requirements: 5.2, 5.6_

- [ ] 4. Remove the per-step cross-key transposes
  - [ ] 4.1 Write `export/optimize_decoder.py` for the split graphs: prefill emits
    `present.{i}.encoder.key` already transposed, step drops the six matching
    `Transpose` nodes and uses the past input directly
    - Apply to the fp32, int8, and int4 step graphs; update `ValueInfo` shapes; run
      `onnx.checker` on every edited model
    - _Requirements: 6.1, 6.2_
  - [ ] 4.2 Add a tiny-model test that compares step logits before and after surgery
    within `1e-4`, for fp32, int8, and int4 files
    - _Requirements: 6.3, 6.4_

- [ ] 5. Teach the engine to choose precision and post-processing
  - [ ] 5.1 Add `load_precision(models_dir, requested)` reading `quant.json`, with
    `auto`, `fp32`, `int8`, and `int4`; fail on a missing file unless
    `ALLOW_FP32_FALLBACK=1`; `auto` resolves to `int8` until the 4-bit gate passes
    - Unit tests with a fake model directory for each mode and the fallback
    - _Requirements: 7.1, 7.4_
  - [ ] 5.2 Read `POSTPROCESS` in `server/app.py`, pass it to the engine, and reject
    unknown values at startup with a message listing the valid ones
    - _Requirements: 1.1, 1.2_
  - [ ] 5.3 Instrument `generate()` with `vision_ms`, `embed_ms`,
    `text_encoder_ms`, `prefill_ms`, `decode_ms`, `total_ms`, `steps`, `tokens`,
    `decode_tokens_per_s`, and `total_tokens_per_s`, keeping the V1 keys
    - _Requirements: 2.3, 11.1, 11.2_
  - [ ] 5.4 Add tests asserting the timing keys, the two rate formulas, and that
    `steps + 1` equals `tokens` for a run without EOS
    - Measure instrumentation overhead on one page and assert it is under 0.5%
    - _Requirements: 11.2, 11.3_

- [ ] 6. Add the measurement harnesses
  - [ ] 6.1 Write `bench/precision_sweep.sh`: one container per `PRECISION`
    (`fp32`, `int8`, `int4`), running accuracy, sequential latency, open-loop rates,
    and `VmHWM`; JSON per config
    - _Requirements: 3.3, 10.1_
  - [ ] 6.2 Write `bench/postprocess_eval.py` to score both modes against the same
    references and print per-image CER and mean
    - _Requirements: 1.3_
  - [ ] 6.3 Extend `bench/load.py` to aggregate `decode_tokens_per_s`,
    `total_tokens_per_s`, and each per-stage timing as mean/p50/p95, skipping
    fields a V1 server does not send
    - Add a unit test with synthetic timing rows, including a missing-field case
    - _Requirements: 11.4_

- [ ] 7. Measure and decide the defaults
  - [ ] 7.1 Run the 50-image accuracy for fp32, int8, and int4; apply the 0.90 mean /
    0.75 per-image gate to each. Record the per-image worst case and the CER delta
    versus fp32 for every variant
    - _Requirements: 5.4, 5.5, 5.6_
  - [ ] 7.2 Run 50-image sequential latency for the winning precision; apply the
    5.0 s p50 / 6.5 s p95 gate and confirm every request is under 30 s
    - _Requirements: 2.1, 2.2_
  - [ ] 7.3 Run open-loop at 0.10/0.15/0.20/0.25/0.30; apply the 0.25 RPS at 99%
    gate and verify fast 503s above capacity
    - _Requirements: 3.1, 3.2_
  - [ ] 7.4 Read peak `VmHWM` under the spike and the sustained step; apply the
    3.0 GiB gate
    - _Requirements: 4.1, 4.2_
  - [ ] 7.5 Bake the winning `PRECISION` and `POSTPROCESS` into the Dockerfile and
    `server/app.py` defaults; leave `PRECISION=fp32` available as the fallback
    - _Requirements: 7.4, 10.3_
  - [ ] 7.6 Record the winning config's tokens per second (mean, p50, p95) and its
    per-stage breakdown next to p50 and RPS
    - _Requirements: 10.1, 11.5_

- [ ] 8. Estimate the batching win, then leave it off
  - [ ] 8.1 Write `bench/batch_probe.py` and run batch 1/2/4 at the winning
    precision under `--cpus=4`, reporting RPS, p95, and peak memory
  - [ ] 8.2 Keep `OCR_WORKERS=1`, `QUEUE_LIMIT=1`, batch 1 unless RPS rises at
    least 15% with p95 under 30 s and peak under 3.8 GiB; record the numbers either
    way
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 9. Verify the submission
  - [ ] 9.1 Build the image from a clean cache and assert no `torch`/`torchvision`,
    `quant.json` present, and all nine model files load (five fp32, three int8, one
    int4)
    - _Requirements: 7.1, 7.2, 7.3_
  - [ ] 9.2 Run the API contract tests and `bench.stress` against the winning
    container; confirm health, 400s, 413s, fast 503s, and no 500
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 4.2_
  - [ ] 9.3 Update `README.md` and `journal.md` with the measured V2 rows, the
    precision defaults, the tokens-per-second and per-stage table, and the
    five-graph versus three-module rationale
    - _Requirements: 10.1, 10.2, 10.3, 11.5_
  - [ ] 9.4 Run the full test suite with and without `MODEL_DIR`, then commit and
    push `main`
    - _Requirements: 10.1_

- [ ] 10. Optional probes, only after every gate above passes
  - [ ] 10.1 ORT 1.31.1 is already the resolved pin. Record the 1.20.1 → 1.31.1
    before/after from the upgrade (7.0 s load, same text) and do not re-open the
    version unless a measured regression appears.
    - _Requirements: 2.1_
  - [ ] 10.2 A/B two workers with two threads each against one worker with four;
    keep only if sustained RPS improves at least 10% with p95 under 30 s
    - _Requirements: 3.1, 9.1_
  - [ ] 10.3 Score 3-beam search on the 50 images; adopt only if mean accuracy rises
    at least 2 points and p95 stays under 10 s
    - _Requirements: 5.4_
  - [ ] 10.4 If the mixed 4-bit step graph misses the accuracy gate, walk the fallback
    ladder: 8-bit attention + 4-bit FFN + 8-bit LM head; then 4-bit transformer +
    fp32 LM head; then 8-bit all. Keep only a rung that stays at 0.90 mean / 0.75
    per-image and beats 8-bit decode tok/s by at least 10%
    - Record every rejected rung and its accuracy delta
    - _Requirements: 5.4, 5.5, 5.6, 10.1, 11.5_
