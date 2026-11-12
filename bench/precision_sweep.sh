#!/usr/bin/env bash
# Sweep precision variants (fp32, int8, int4) under container limits (--memory=4g --cpus=4)
# measuring accuracy, latency, and peak memory (VmHWM).
set -euo pipefail

IMAGE="${IMAGE:-florence2-ocr:final}"
MEM="${MEM:-4g}"
CPUS="${CPUS:-4}"
RESULTS_DIR="${RESULTS_DIR:-results/precision}"
REFS="${REFS:-/home/eequaled/florence-lab/references-funsd-50.json}"
IMAGES_DIR="${IMAGES_DIR:-/home/eequaled/florence-lab/funsd/dataset/testing_data/images}"
PORT=8000

mkdir -p "$RESULTS_DIR"

for prec in fp32 int8 int4; do
  cname="florence2-bench-$prec"
  echo "=== Running precision sweep for $prec ==="
  docker rm -f "$cname" >/dev/null 2>&1 || true

  docker run -d --name "$cname" \
    --memory="$MEM" --cpus="$CPUS" \
    -p "$PORT:8000" \
    -e PRECISION="$prec" \
    -e POSTPROCESS="baseline" \
    -e ORT_ALLOW_SPINNING="0" \
    "$IMAGE" >/dev/null

  # Wait for health
  echo "Waiting for $cname to be healthy..."
  for i in $(seq 1 40); do
    if curl -s -f "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  # Measure accuracy
  echo "Scoring accuracy on $cname..."
  PYTHONPATH="$PWD" /home/eequaled/florence-lab/.venv-onnx/bin/python -m bench.accuracy \
    --url "http://127.0.0.1:$PORT" \
    --images "$IMAGES_DIR" \
    --refs "$REFS" \
    --timeout 30 \
    > "$RESULTS_DIR/acc_$prec.json"

  # Measure sequential latency
  echo "Measuring latency on $cname..."
  PYTHONPATH="$PWD" /home/eequaled/florence-lab/.venv-onnx/bin/python -m bench.load \
    --url "http://127.0.0.1:$PORT" \
    --images "$IMAGES_DIR" \
    --mode latency \
    --repeats 1 \
    --timeout 30 \
    > "$RESULTS_DIR/latency_$prec.json"

  # Measure peak memory
  pid=$(docker inspect --format '{{.State.Pid}}' "$cname")
  vm_hwm=$(grep VmHWM "/proc/$pid/status" | awk '{print $2, $3}')
  echo "Peak memory (VmHWM) for $prec: $vm_hwm"
  echo "{\"precision\": \"$prec\", \"vm_hwm\": \"$vm_hwm\"}" > "$RESULTS_DIR/mem_$prec.json"

  docker rm -f "$cname" >/dev/null 2>&1 || true
done

echo "Sweep complete. Results in $RESULTS_DIR"
