#!/usr/bin/env bash
# Worker x intra-op-thread sweep under the official container limits.
#
# The five graphs share one FlorenceOCR instance, and every worker thread calls the
# same sessions. So the question is not "how many processes" but "how many callers
# share the intra-op pool, and how big is that pool". This script answers it with
# the same measurements the challenge scores: sequential p50, closed-loop
# concurrency, and open-loop arrival rates.
#
#   IMAGE=florence2-ocr:opt ./bench/sweep.sh
#
# Env overrides: IMAGE, CPUS (default 4), IMAGES (a directory of sample images),
# SECONDS_PER_STEP (default 45), CONFIGS ("workers:threads:spinning:queue ...").
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAB="${FLORENCE_LAB:-$HOME/florence-lab}"
PY="${PYTHON:-$LAB/.venv-onnx/bin/python}"
IMAGES="${IMAGES:-$LAB/funsd/dataset/testing_data/images}"
OUT="${OUT:-$REPO/results/sweep}"
IMAGE="${IMAGE:-florence2-ocr:opt}"
CPUS="${CPUS:-4}"
SECONDS_PER_STEP="${SECONDS_PER_STEP:-45}"
CONFIGS="${CONFIGS:-1:4:1:1 1:4:0:1 2:2:0:1 2:4:0:1 4:1:0:1}"

mkdir -p "$OUT"
cd "$REPO" || exit 1

for cfg in $CONFIGS; do
  W="${cfg%%:*}"; rest="${cfg#*:}"
  T="${rest%%:*}"; rest="${rest#*:}"
  S="${rest%%:*}"; Q="${rest#*:}"
  TAG="w${W}_t${T}_s${S}_q${Q}"
  LOG="$OUT/$TAG.log"
  echo "=== $TAG ===" | tee "$LOG"

  docker rm -f ocr-sweep >/dev/null 2>&1
  if ! docker run -d --rm --name ocr-sweep -p 8000:8000 \
      --memory=4g --memory-swap=4g --cpus="$CPUS" \
      -e OCR_WORKERS="$W" -e ORT_THREADS="$T" \
      -e ORT_ALLOW_SPINNING="$S" -e QUEUE_LIMIT="$Q" \
      "$IMAGE" >/dev/null; then
    echo "start failed" | tee -a "$LOG"
    continue
  fi

  healthy=0
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then healthy=1; break; fi
    sleep 2
  done
  if [ "$healthy" != 1 ]; then
    echo "never healthy" | tee -a "$LOG"
    docker logs --tail 20 ocr-sweep >>"$LOG" 2>&1
    docker stop ocr-sweep >/dev/null 2>&1
    continue
  fi

  {
    echo "--- latency (step 1) ---"
    "$PY" -m bench.load --url http://127.0.0.1:8000 --images "$IMAGES" \
      --mode latency --repeats 1 --timeout 30
    echo "--- closed loop ---"
    "$PY" -m bench.load --url http://127.0.0.1:8000 --images "$IMAGES" \
      --mode throughput --clients 2,4 --seconds "$SECONDS_PER_STEP" --timeout 30
    echo "--- open loop (step 2) ---"
    "$PY" -m bench.load --url http://127.0.0.1:8000 --images "$IMAGES" \
      --mode open --rates 0.1,0.15 --seconds "$SECONDS_PER_STEP" --timeout 30
  } 2>&1 | tee -a "$LOG"

  echo "--- memory ---" | tee -a "$LOG"
  docker exec ocr-sweep sh -c "grep -E 'VmHWM|VmRSS' /proc/1/status" 2>&1 | tee -a "$LOG"
  docker stats --no-stream --format '{{.MemUsage}} pids={{.PIDs}}' ocr-sweep 2>&1 | tee -a "$LOG"
  docker stop ocr-sweep >/dev/null 2>&1
  docker rm -f ocr-sweep >/dev/null 2>&1
done

echo "sweep done: $OUT"
