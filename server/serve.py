"""Answer the API with ONNX Runtime only; PyTorch appears in the build stage, never here."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEVICE_PORT = int(os.getenv("PORT", "8000"))
DEVICE_HOST = os.getenv("HOST", "0.0.0.0")
DEVICE_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))


def main() -> int:
    import uvicorn

    if "torch" in sys.modules:
        print("refusing to start: torch is imported in the serving process", file=sys.stderr)
        return 2
    missing = [name for name in ("vision_encoder", "embed_tokens", "encoder_model",
                                 "decoder_prefill", "decoder_step", "meta.json", "tokenizer.json")
               if not (DEVICE_MODEL_DIR / (name if "." in name else f"{name}.onnx")).exists()]
    if missing:
        print(f"model directory {DEVICE_MODEL_DIR} is missing: {missing}", file=sys.stderr)
        return 2

    uvicorn.run(
        "server.app:app",
        host=DEVICE_HOST,
        port=DEVICE_PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        workers=1,
        timeout_keep_alive=int(os.getenv("KEEP_ALIVE", "30")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
