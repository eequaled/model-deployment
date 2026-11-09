# AI Inference Challenge: Deploy & Optimize Florence-2 on a CPU

> Load it. Understand it. Serve it. Make it fast. Survive the stress test.

Values marked **[TBA]** will be announced by the organizer on Day 1.

---

## 1. What is this competition?

Every participant takes the same AI model, **Florence-2-base** by Microsoft, and turns it into a backend service that reads text from images (OCR).

Your job is to **load the model, optimize it, and deploy it behind your own backend server** so it handles **as many requests per second as possible, as fast as possible, on limited resources**: 4 GB of RAM, a few CPU threads, and no GPU.

The organizer runs every server on the same machine, with the same limits and the same hidden test images. The leaderboard ranks:

- **Inference time**: how fast your server answers a single request.
- **Throughput**: how many requests per second your server can handle.
- **Resilience**: on the final day, how well your server survives the organizer's stress tests.

But speed alone is not enough. This competition is also about learning. Every participant must understand how a model runs, how the model they deployed works, and why every optimization they made works. **Anyone who cannot explain their own work is eliminated, however fast their server is.**

---

## 2. The model

| | |
|---|---|
| Model | `microsoft/Florence-2-base` ([Hugging Face](https://huggingface.co/microsoft/Florence-2-base)) |
| Size | ~230M parameters, 464 MB weights file (stored in fp16) |
| License | MIT |
| Task you serve | `<OCR>`: image in, text out |

Florence-2 is a vision-language model. A vision encoder turns the image into visual tokens, then a transformer encoder-decoder generates the answer **one token at a time** until it decides to stop. That token-by-token loop is where most of the time goes, and where most of your optimization work will happen.

**Starter repo:** [TBA]. It contains:

- `samples/`: sample images, the same kind as the hidden test set
- `references/`: the original model's output for each sample
- `baseline.py`: reference PyTorch inference, for understanding and checking only. It can **not** be your server (see 4.4).

---

## 3. Objectives

### Main objective

Deploy Florence-2-base behind your own backend server and make it **handle the most requests per second with the lowest inference time** under the official resource limits, while keeping accuracy **at 70% or above**.

### Side objectives (mandatory for everyone)

Every participant must build real knowledge in:

1. **How a model runs at inference time**: tensors, shapes, dtypes (fp32 / fp16 / int8), computation graphs, operators, and what is different from training.
2. **Florence-2 architecture**: the vision encoder, the encoder-decoder transformer, the tokenizer, autoregressive generation, greedy vs. beam search, the KV cache, and the exact inputs and outputs of each part.
3. **Model optimization**: model formats and export (ONNX), graph optimizations, operator fusion, quantization (dynamic vs. static), and the speed vs. accuracy trade-off.
4. **Backend & systems**: CPU cores vs. threads, concurrency vs. parallelism, async vs. threads vs. processes, batching, queues and back-pressure, memory limits, latency vs. throughput.
5. **Benchmarking**: p50 / p95 / p99 latency, requests per second, warm-up, load testing, and how to compare two versions fairly.

Go as deep as you want on the technical side. **A simple optimization you fully understand beats a complex one you copied.**

---

## 4. Rules

### 4.1 Resources

| Resource | Limit |
|---|---|
| RAM | **4 GB** for your whole server: model, runtime, buffers, everything |
| CPU | **[TBA] threads** |
| GPU | **None** |

The organizer enforces the limits with Docker:

```bash
docker run --rm -p 8000:8000 --memory=4g --memory-swap=4g --cpus=[TBA] <your-image>
```

If your server goes over 4 GB, the container is killed (OOM). That counts as a crash.

### 4.2 Hardware

- You may develop and optimize on **Intel, AMD, and Apple CPUs only**.
- No GPU, NPU, or TPU, local or remote. **Third-party GPU clouds (Colab, Kaggle, RunPod, etc.) are not allowed.**
- Final evaluation runs on the organizer's machine: **[TBA: CPU model + architecture]**. If you develop on Apple Silicon (ARM), make sure your Docker image also builds and runs on that architecture.

### 4.3 What you can use

- **Any programming language**: Python, Rust, C++, Go, TypeScript/NestJS, Java…
- **Any web framework or server you set up yourself**: FastAPI, NestJS, Axum/Actix, Drogon, Gin…
- **Any inference runtime or library called from your own code**: ONNX Runtime, OpenVINO Runtime, your own kernels…
- **Any optimization technique**: export/conversion, graph optimization, quantization, pruning, operator fusion, thread tuning, batching, KV cache, decoding changes, multiple processes, faster preprocessing…

### 4.4 What you cannot use

- ⚠️ **No PyTorch in the backend server. Balak.**
  Your server must not import or link PyTorch (`torch`, `torchvision`, LibTorch, or anything that loads them, such as `transformers` models). PyTorch is allowed **offline only**: to load the original weights, inspect them, and export or convert them. The simplest proof is that `torch` is not installed in your final Docker image.
- **No prebuilt ML serving servers**: NVIDIA Triton Inference Server, TorchServe, TensorFlow Serving, OpenVINO Model Server, BentoML, Ray Serve, KServe, vLLM, TGI, or anything else that does the serving for you.
  > Rule of thumb: a **library you call** from your server is OK. A **server you drop your model into** is not.
- **No pre-converted or pre-optimized Florence-2 from the internet**, for example the ready-made ONNX exports on Hugging Face (`onnx-community/Florence-2-base`) or any pre-quantized weights. Start from the original `microsoft/Florence-2-base` weights and do the conversion yourself.
- **No replacing the model.** What you serve must be derived from the original weights. No other OCR engine (Tesseract, PaddleOCR…), no other model, no external APIs.
- **No response caching.** Every request must run inference. Caching inside a single request (like the KV cache) is fine.

### 4.5 Accuracy threshold: 70%

- Accuracy is measured on a **hidden test set** of the same kind as the sample images.
- The reference is the output of the **original model**: PyTorch, fp32, with the settings from `baseline.py`.
- Per image: `accuracy = max(0, 1 − CER)`, where `CER = edit_distance(your_text, reference_text) / len(reference_text)`.
- Your accuracy is the average over all test images.
- **70% or above: you are ranked. Below 70%: you are eliminated**, however fast your server is.

> Quantization, changing the decoding strategy, and lowering the token limit can all cost accuracy. Measure after every change.

---

## 5. API contract

Every server must expose exactly this API, so the organizer's tools can test everyone the same way.

**Port:** `8000` inside the container.

### `GET /health`

- Returns `200` with `{"status": "ok"}` **only when the model is loaded and ready**.
- Must return `200` within **120 seconds** of the container starting.

### `POST /ocr`

- **Request:** `multipart/form-data` with one field, `image` (JPEG or PNG).
- **Response:** `200` with JSON:

  ```json
  { "text": "the text read from the image" }
  ```

- Valid images up to **10 MB** must be answered. You may reject larger files with `413`.
- Bad input (missing field, not an image, corrupted file) must get a `4xx` JSON error. **Never crash.**

```bash
curl -F "image=@samples/example.jpg" http://localhost:8000/ocr
```

Any extra endpoints are up to you.

---

## 6. Leaderboard & scoring

### How the organizer runs your server

```bash
git clone <your-repo> && cd <your-repo>
docker build -t <your-name> .
docker run --rm -p 8000:8000 --memory=4g --memory-swap=4g --cpus=[TBA] <your-name>
```

The organizer always tests the **latest commit on your `main` branch**. Leaderboard runs happen regularly and are announced in the daily meets.

### Step 0: Gates (pass / fail)

To be ranked, your server must:

- build and start with the commands above;
- answer `/health` within 120 s;
- follow the API contract;
- reach **70% accuracy or more**;
- contain no PyTorch and no forbidden serving framework.

### Step 1: Inference time (30 points)

Requests are sent **one at a time** (no concurrency) over the test set.

- Metric: **p50 latency** (p95 is shown too).
- `score = 30 × best_p50 / your_p50`

### Step 2: Throughput (40 points)

The organizer's load generator raises the number of concurrent requests step by step, 2 minutes per step.

- A request is successful if it returns `200` with a valid response **within 30 s**.
- Metric: the **highest requests/second sustained for a full step with at least 99% success**.
- `score = 40 × your_rps / best_rps`

### Step 3: Stress test, final day (30 points)

The organizer attacks your server with scenarios that stay **secret until the day**. Expect things like:

- sudden traffic spikes far beyond normal load;
- long sustained load (memory leaks show up here);
- large or unusual images;
- invalid requests mixed in with valid ones.

You earn points for every scenario where your server:

- **stays up**: no crash, no OOM kill, no manual restart;
- **keeps answering correctly**: accuracy stays at 70% or above;
- **recovers**: latency goes back to normal once the load drops.

> Under overload, answering `503` quickly is better than crashing. A crash in the stress test costs you that scenario's points, not your place in the competition.

### Final ranking

`final score = Step 1 + Step 2 + Step 3` (out of 100). Tie-breaker: higher accuracy.

---

## 7. High-level map: what you have to do

1. **Load the model.**
   Clone the starter repo, load `microsoft/Florence-2-base` in PyTorch on your machine (offline), run `<OCR>` on the samples, and check that your outputs match `references/`.

2. **Understand how it works.**
   What goes in: the image becomes a pixel tensor, and the task prompt becomes tokens. What happens inside: vision encoder → encoder → decoder loop, one token at a time. What comes out: token IDs → text. Know the shape and dtype of every tensor, and profile where the time is spent.

3. **Choose a backend and find a way to run the model in it, without PyTorch.**
   Pick your language and framework. Then work out how to run the model there: export it (e.g. to ONNX) and use a runtime, or write your own. Preprocessing (resize, normalize) and tokenization/decoding must also live in your backend.

4. **Deploy.**
   Run the model inside your server, implement `/health` and `/ocr` following the API contract, wrap everything in a `Dockerfile`, and run it with the official limits (`--memory=4g --cpus=[TBA]`).

5. **Set up your evaluation loop before you optimize anything.**
   Write a script that:
   - computes **accuracy** against `references/`;
   - measures **p50 / p95 latency** with sequential requests;
   - measures **requests per second** under concurrency with a load-testing tool (k6, Locust, wrk, hey, oha…).

   Run it after **every** change and log the numbers in your `journal.md`. If you don't measure, you're guessing.

6. **Research how to optimize the model and the backend.**
   - Model side: graph optimizations, quantization, decoding strategy, KV cache, token limits…
   - Server side: concurrency model, batching, threads vs. processes, memory, request queue, preprocessing speed…

   Change one thing at a time → measure → keep it or revert it.

7. **Harden for the stress test.**
   Timeouts, image size limits, a bounded request queue, fast `503`s under overload, no memory leaks.

---

## 8. Repository & submission

- Create a dedicated **GitHub repository** for your work (private or public).
- ⚠️ **Before the end of Day 1, invite `khiro45` as a collaborator** on your repo (*Settings → Collaborators → Add people → `khiro45`*). **No invite by the end of Day 1 = not registered in the competition.**
- Your `main` branch is your submission. The last commit before the deadline counts.

### Required structure

```
your-repo/
├── journal.md      # mandatory development journal (see section 9)
├── README.md       # how to build, run, and reproduce your optimized model
├── Dockerfile      # the official way the organizer runs your server
├── export/         # scripts that turn the original weights into your optimized model
├── server/         # your backend code
└── bench/          # your accuracy + latency + load-test scripts
```

### About model files

- GitHub rejects files over 100 MB, so don't commit big weights directly.
- Preferred: generate your optimized model **at Docker build time** from the original weights. A multi-stage build lets you use PyTorch in the build stage only and ship a final image without it.
- Alternative: Git LFS or GitHub Releases.
- Either way, the organizer must be able to **reproduce your optimized model from the original weights** using your `export/` scripts.

---

## 9. Daily meets & development journal

- **Daily meets, every day:** regular check-ins where participants share progress, numbers, blockers, and insights. Come with what you tried, what it changed, and what's next.
- **Mandatory GitHub journal:** inside your repository you **MUST** maintain a file named `journal.md`. It is a critical part of your evaluation and must log your complete development journey:
  - your overall approach and strategy;
  - problems, bugs, and technical hurdles you hit;
  - how you debugged and solved them;
  - architecture ideas, experiments, and thoughts;
  - **every optimization, with its numbers before and after**;
  - key things you learned about model inference, optimization, and backend systems.

Keep an experiment log in your journal, like this:

| Date | Change | Accuracy | p50 latency | Max RPS | Kept? |
|---|---|---|---|---|---|
| Day 2 | Baseline: first working server | x% | x.x s | x.x | ✅ |
| Day 3 | (your next change) | x% | x.x s | x.x | ✅ / ❌ |

During the daily meets and at the final review, you will be asked to explain your choices using your journal.

---

## 10. Collaboration policy

This is an **individual competition**: everyone submits their own server and is ranked alone. Collaboration is allowed, but only to help each other learn.

### ✅ Allowed

- Discussing concepts, papers, docs, and tutorials together (ONNX, quantization, batching, async…).
- Explaining things to each other (how a decoder generates tokens, what a KV cache is…).
- Helping someone debug an installation or an error message.
- Sharing public learning resources and general tool recommendations (e.g. which load-testing tool you like).

### ❌ Not allowed

- Sharing or copying server code, export/conversion scripts, or optimized model files.
- Submitting a server or model built (fully or partly) by someone else.
- Building a solution "together" and submitting it twice.
- Using pre-converted or pre-optimized Florence-2 models downloaded from the internet.

---

## 11. Elimination conditions

You are eliminated if:

- your accuracy is **below 70%**;
- your server uses **PyTorch** or a **prebuilt ML serving server**;
- you used a **GPU** or a **third-party GPU cloud**;
- you downloaded a **pre-converted / pre-optimized** Florence-2, or **replaced the model**;
- you **copied** code, scripts, or model files from someone else, or shared yours;
- you did **not invite `khiro45`** to your repo by the end of Day 1;
- your repo has **no `journal.md`**;
- you **cannot explain your own work**.

---

## 12. FAQ

**Can I use Python in my server?**
Yes, as long as PyTorch is not in it.

**Are ONNX Runtime / OpenVINO allowed?**
Yes. Runtimes and libraries you call from your own server are allowed. Ready-made model servers (Triton, TorchServe…) are not.

**Can I use `transformers` in my server?**
Not its models, because they need PyTorch. Libraries that only handle tokenization and don't depend on PyTorch (e.g. Hugging Face `tokenizers`) are fine.

**Can I change the generation settings (greedy instead of beam search, fewer max tokens)?**
Yes, as long as your accuracy stays at 70% or above.

**Can I cache responses for images I've already seen?**
No. Every request must run inference.

**Can I use a GPU just to export the model faster?**
No. No GPU at any point.

**What happens if my server crashes during the stress test?**
You lose the points for that scenario. You are not eliminated for it.

---

## 13. Recommended learning resources

- **ONNX documentation**: the model format most of you will export to: graphs, operators, opsets.
  https://onnx.ai/onnx/
- **NVIDIA Technical Blog: "Mastering LLM Techniques: Inference Optimization"**: written for GPUs and LLMs, but the core ideas (token-by-token decoding, KV cache, batching, quantization) apply directly to Florence-2's decoder on a CPU.
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
- **ONNX Runtime docs: thread management & quantization**: how to tune threads for a CPU and how to quantize a model.
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- **Florence-2 paper & model card**: how the model is built, what it was trained on, and how to call it.
  https://arxiv.org/abs/2311.06242
  https://huggingface.co/microsoft/Florence-2-base

---

Good luck, and remember: **measure everything, understand everything, and don't you dare put PyTorch in your server.**
