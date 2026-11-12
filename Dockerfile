# Multi-stage so the shipped image has ONNX Runtime and no PyTorch at all.
#
# Stage 1 carries PyTorch, Transformers and Optimum purely to run export/export_onnx.py
# against the original microsoft/Florence-2-base weights. Stage 2 copies only the
# resulting graphs, so "torch is not installed in the final image" is a property of
# the build rather than a promise in a README.

FROM python:3.12-slim AS exporter

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    HF_HOME=/hf-cache \
    CUDA_VISIBLE_DEVICES="" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels only: the challenge forbids a GPU at every stage, including export.
# torch and torchvision must be installed together from the CPU index: installing
# torch alone lets timm pull a PyPI torchvision built against a different torch,
# which fails at import with "operator torchvision::nms does not exist".
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1" \
    && pip install "transformers==4.48.3" "onnx==1.17.0" "onnxruntime==1.31.1" \
       "onnx_ir" "ml_dtypes" "safetensors" \
       "pillow" "numpy" "huggingface_hub" "timm" "einops"

WORKDIR /build
COPY export/export_onnx.py export/export_onnx.py
RUN python export/export_onnx.py --out /models

FROM python:3.12-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ORT_ALLOW_SPINNING=0 \
    MODEL_DIR=/models \
    PORT=8000

RUN pip install "onnxruntime==1.31.1" "tokenizers==0.20.3" "fastapi==0.115.6" \
    "uvicorn==0.34.0" "pillow==11.0.0" "numpy==1.26.4" "python-multipart==0.0.20"

WORKDIR /app
COPY server/ server/
COPY --from=exporter /models /models

# Fail the build, not the evaluation, if PyTorch ever leaks into the runtime stage.
RUN python -c "import importlib.util as u, sys; \
bad=[m for m in ('torch','torchvision') if u.find_spec(m)]; \
sys.exit('torch present in runtime image: '+str(bad)) if bad else print('runtime is torch-free')"

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python", "-m", "server.serve"]
