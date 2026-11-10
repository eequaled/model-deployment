"""HTTP surface: /health, /ocr, and a bounded queue so overload turns into fast 503s.

Concurrency shape and why:

* Inference sessions are not free-threaded in a way that helps us, and each request
  allocates a cache of its own. So a fixed pool of worker threads owns all the
  requests, and a semaphore caps how many can be in flight at once.
* When the queue is full the server rejects immediately instead of queueing
  without limit. An unbounded queue is how a latency problem turns into a memory
  problem and then into the OOM crash the rules treat as a failure.
* The upload is read before it is validated, but only up to the documented limit;
  the body is never trusted to be an image until PIL accepts it.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .ocr_engine import FlorenceOCR

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
THREADS = int(os.getenv("ORT_THREADS", "0"))
WORKERS = int(os.getenv("OCR_WORKERS", "1"))
QUEUE_LIMIT = int(os.getenv("QUEUE_LIMIT", "8"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "0"))
PREFILL_MODE = os.getenv("PREFILL_MODE", "")


class State:
    engine: FlorenceOCR | None = None
    pool: ThreadPoolExecutor | None = None
    slots: asyncio.Semaphore | None = None


state = State()


def build_engine() -> FlorenceOCR:
    """Constructed once at startup. Tests patch this to avoid needing weights."""
    return FlorenceOCR(MODEL_DIR, threads=THREADS, max_new_tokens=MAX_NEW_TOKENS,
                       prefill_mode=PREFILL_MODE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.engine = build_engine()
    state.pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="ocr")
    state.slots = asyncio.Semaphore(WORKERS + QUEUE_LIMIT)
    logger.info("ready: workers=%d queue=%d threads=%s", WORKERS, QUEUE_LIMIT, THREADS or "auto")
    try:
        yield
    finally:
        if state.pool is not None:
            state.pool.shutdown(wait=False)


app = FastAPI(title="florence2-ocr", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    if state.engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


def _decode_image(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


async def _infer(payload: bytes) -> tuple[str, dict[str, float]]:
    assert state.engine is not None and state.pool is not None

    def work() -> tuple[str, dict[str, float]]:
        image = _decode_image(payload)
        return state.engine.generate(image)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(state.pool, work)


@app.post("/ocr")
async def ocr(request: Request, image: UploadFile | None = File(default=None)) -> JSONResponse:
    if state.engine is None or state.slots is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    if image is None:
        return JSONResponse({"error": "expected multipart field 'image'"}, status_code=400)

    # Reject oversized bodies from the declared header before reading them, so a
    # 500 MB upload never becomes 500 MB of resident memory.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD + 4096:
        return JSONResponse({"error": f"upload exceeds {MAX_UPLOAD} bytes"}, status_code=413)

    payload = await image.read(MAX_UPLOAD + 1)
    if not payload:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    if len(payload) > MAX_UPLOAD:
        return JSONResponse({"error": f"image larger than {MAX_UPLOAD} bytes"}, status_code=413)

    if state.slots.locked():
        return JSONResponse({"error": "busy"}, status_code=503)

    async with state.slots:
        try:
            text, timings = await _infer(payload)
        except Image.DecompressionBombError as exc:
            # A small file can still decode to an enormous bitmap. That is bad
            # input, not a server fault: refuse it before it costs us the heap.
            return JSONResponse(
                {"error": f"image exceeds the safe pixel limit: {exc}"}, status_code=413
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return JSONResponse({"error": f"not a readable image: {exc}"}, status_code=400)
        except Exception:  # never let one bad request kill the worker
            logger.exception("inference failed")
            return JSONResponse({"error": "inference failed"}, status_code=500)

    return JSONResponse({"text": text, "timings": timings})


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "florence2-ocr", "endpoints": "GET /health, POST /ocr"}
