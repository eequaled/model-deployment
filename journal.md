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

### Five-image provisional references

The participant generated local FP32 references for five FUNSD test images, after one warm-up, using greedy decoding, `use_cache=True`, a 256-new-token cap, and 12 PyTorch threads. The local JSON records image hashes, output text, token IDs, settings, and package versions. Single generation times were 8.36, 6.81, 7.52, 9.33, and 9.07 seconds; returned sequence lengths were 221, 155, 205, 257, and 257. All ended in EOS, but the two 257-token sequences may have reached the cap with a forced EOS; this must be checked before assuming they are complete. A follow-up at 512 new tokens found `forced_eos_token_id=2` and `eos_token_id=2`: `82251504.png` returned 323 tokens in 11.71 s, with text differing from the 256-token run; `93106788.png` returned 513 tokens in 16.45 s, also with different text, and may still have hit the limit. The 256-token results for both should not be assumed complete. These are exploratory timings, not p50, end-to-end latency, or official accuracy measurements. Warnings included `resume_download` deprecation and `early_stopping=True` with one beam.

### Conversion experiments

The conversion environment is separate from the reference environment: Python 3.12, PyTorch 2.15.0+cpu, Transformers 4.48.3, ONNX 1.17.0, ONNX Runtime 1.20.1, and Optimum 1.24.0. The pinned checkpoint loaded with no missing or unexpected keys and without the earlier `flash_attn` import problem.

A public Florence-2 export script was reviewed. It is real but targets a fine-tuned checkpoint and reimplements part of the image path, so it is treated only as a reference design. Our own export wraps the model's existing image-encoding method and fixes the input at 768 × 768, because that is the only size the processor produces and dynamic shapes were the most likely source of tracing errors.

The first vision-encoder graph exported in about 37 s to a 349 MiB FP32 file and ran under the CPU execution provider. Against PyTorch on one fixed random input and three FUNSD images, the largest absolute difference was 8.2 × 10⁻⁵ to 4.1 × 10⁻⁴, the relative L2 difference was about 4 × 10⁻⁶, and cosine similarity rounded to 1.0. The difference was stable across repeated runs. This establishes numerical equivalence for the vision path only; the language encoder, decoder, cache handling, and tokenizer remain unverified.

The token-embedding and language-encoder graphs matched PyTorch to a relative L2 difference near 10⁻⁶, including a deliberately padded second length, so the encoder's declared dynamic sequence length is real rather than nominal. A one-off test then settled how the decoder treats its inputs on cached steps: replacing the encoder features with same-shape zeros left the logits bit-identical, while a zero-length tensor changed them by more than 3 logits. The cached cross-attention therefore ignores the feature values but still consumes their length.

Two decoder graphs were exported: a prefill graph that emits logits plus all 24 cache tensors, and a step graph that takes the 24 cached tensors and emits 12 updated self-attention ones, keeping the cross-attention cache from prefill. Prefill and three chained steps matched PyTorch logits to about 2 × 10⁻⁵ with identical argmax at each step; the cache tensors themselves matched to about 3 × 10⁻⁶. Tracing removed the unused encoder-feature input from the step graph automatically.

One constraint came out of an attempted second encoder length: the step graph broadcasts the encoder attention mask against the cached cross-attention tensors, so its mask length must equal the prefill encoder length. For `<OCR>` that length is 587; a different prompt length simply requires that prefill and steps agree. Full generation against the existing references, and a PyTorch-free preprocessing and tokenizer path, remain untested.

## Experiments

| Date | Change | Accuracy | p50 latency | Max RPS | Kept? |
|---|---|---|---|---|---|
| — | Offline CPU model smoke test; one cold generation: 9.24 s | Not measured | Not measured | Not measured | Exploratory |
| — | Vision-encoder ONNX export, fixed 768 × 768, FP32 | Not applicable | Not measured | Not measured | Parity verified, kept |
| — | Server baseline | Not measured | Not measured | Not measured | Pending |
