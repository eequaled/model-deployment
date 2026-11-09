# Development journal

## Starting point

Goal: serve OCR with Florence-2-base on CPU, meet the challenge's accuracy and resource limits, and understand each optimization well enough to explain it.

Development machine: AMD Ryzen 5 3600, Nobara Linux x86-64, approximately 16 GiB RAM. The machine has a GPU, but the challenge prohibits using it for development, export, or inference. Final testing must enforce the 4 GB container memory limit and the organizer's CPU limit once announced.

## Approach

1. Obtain the organizer's starter repository, sample images, reference outputs, and baseline settings when available. Run the original model on CPU for offline inspection and reference comparison; PyTorch is not allowed in the final server.
2. Study the image preprocessing, prompt tokens, vision encoder, encoder-decoder, and autoregressive decoding. Record tensor shapes, dtypes, and timings as they are measured.
3. Develop a reproducible conversion from the original weights and a server using an allowed runtime. Do not use pre-converted models or response caching.
4. Establish accuracy, sequential latency, throughput, and memory measurements before optimizing. Change one thing at a time and record results, including unsuccessful attempts.
5. Test the API, container limits, overload behavior, and recovery before submission.

## Open questions and actions

- Starter repository, sample images, references, baseline settings, CPU limit, and evaluation hardware: awaiting organizer details.
- `khiro45` collaborator invitation: sent, as reported by the participant. Confirm acceptance separately if needed.
- Confirm with the organizer how the individual-work policy applies to AI assistance; keep solution code and conversion scripts individually authored.

## Offline setup and first model run

The participant verified Docker works, then created isolated environments outside the repository. The working environment uses Python 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, and Transformers 4.40.2. CUDA was unavailable in this environment and `pip check` reported no broken requirements. The original model revision is `microsoft/Florence-2-base` at `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac`.

An initial Python 3.14 / Transformers 5.22.2 attempt exposed a processor/tokenizer mismatch. Loading the native model class separately produced many missing and unexpected keys, so that model was not used for inference. The original repository's remote code failed under 5.22.2 with a missing configuration attribute. The participant inspected the pinned model code before enabling remote code in the offline environment.

On Transformers 4.40.2, the model loader reported a `flash_attn` dependency despite the guarded import not being used on the CPU. The participant installed a local stub in the experimental environment to pass the import check, not the actual CUDA package. This is an environment-specific workaround that must be documented and reviewed for reproducibility; it is not part of the final server. The original model then loaded on CPU in FP32 with no missing, unexpected, or mismatched keys reported. Cached load took 4.1 s and peak process RSS was 1,720 MiB; neither is a container measurement.

For an initial smoke test, the participant downloaded FUNSD locally outside the repository and used `testing_data/images/82837252.png` (754 × 1000). With `<OCR>`, greedy decoding (`num_beams=1`), and `max_new_tokens=256`, the reported inputs were `input_ids` of shape `(1, 10)` (int64), `attention_mask` of shape `(1, 10)`, and `pixel_values` of shape `(1, 3, 768, 768)` (float32). One cold generation took 9.24 s and returned 221 tokens and plausible text, with spacing and recognition errors. This is not an end-to-end server latency benchmark, a p50 measurement, or an official accuracy score. The organizer's generation settings and references remain unknown.

## Experiments

| Date | Change | Accuracy | p50 latency | Max RPS | Kept? |
|---|---|---|---|---|---|
| — | Offline CPU model smoke test; one cold generation: 9.24 s | Not measured | Not measured | Not measured | Exploratory |
| — | Server baseline | Not measured | Not measured | Not measured | Pending |
