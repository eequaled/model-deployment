#!/usr/bin/env bash
# One long evidence-collecting run: export, parity, API, unit tests, bench, docker.
#
# Everything is logged to results/ so a single file can be handed back afterwards.
# Nothing here stops the whole run on the first failure: the point is to come back
# to a complete picture, not to a stack trace.
#
#   ./runall.sh              # everything
#   ./runall.sh --no-docker  # skip the docker stages
#   ./runall.sh --docker-only
#
# Exit status: 0 if every stage that ran succeeded, 1 otherwise.

set -u
cd "$(dirname "$0")" || exit 1

LAB="${FLORENCE_LAB:-$HOME/florence-lab}"
NEW="$LAB/onnx-export-v2"
OLD="$LAB/onnx-export"
IMAGES="$LAB/funsd/dataset/testing_data/images"
LOG_DIR="results"
MASTER="$LOG_DIR/runall.log"
PORT="${PORT:-8000}"
RUN_DOCKER=1
ONLY_DOCKER=0

for arg in "$@"; do
  case "$arg" in
    --no-docker) RUN_DOCKER=0 ;;
    --docker-only) ONLY_DOCKER=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

FAILED=0
SERVERS=()
cleanup() {
  for pid in "${SERVERS[@]:-}"; do
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null
  done
}
trap cleanup EXIT

say() {  # say <stage> <message>
  local line
  line="==> [$1] $2"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$MASTER"
}

# run <stage> <logfile> <command...>
run() {
  local stage="$1" log="$2"; shift 2
  say "$stage" "running: $*"
  if "$@" >>"$log" 2>&1; then
    say "$stage" "PASS (log: $log)"
  else
    local rc=$?
    say "$stage" "FAIL rc=$rc (log: $log)"
    printf '%s\n' "--- last 25 lines of $log ---" >>"$MASTER"
    tail -25 "$log" >>"$MASTER"
    FAILED=1
  fi
}

# expect <stage> <logfile> <want_exit> <command...>
expect() {
  local stage="$1" log="$2" want="$3"; shift 3
  say "$stage" "running (expecting exit $want): $*"
  "$@" >>"$log" 2>&1
  local rc=$?
  if [ "$rc" = "$want" ]; then
    say "$stage" "PASS (exit $rc as expected)"
  else
    say "$stage" "FAIL (exit $rc, wanted $want; log: $log)"
    tail -10 "$log" >>"$MASTER"
    FAILED=1
  fi
}

# wait_health <base_url> <seconds>
wait_health() {
  local url="$1" limit="${2:-180}" waited=0
  while [ "$waited" -lt "$limit" ]; do
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      say health "ready after ~${waited}s"
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  say health "NOT ready after ${limit}s"
  FAILED=1
  return 1
}

printf 'runall started %s\n' "$(date -Is)" >"$MASTER"
say env "repo=$(pwd) lab=$LAB python=$(command -v python || echo none)"

if [ "$ONLY_DOCKER" = 0 ]; then

  say versions "recording environment"
  {
    echo "date: $(date -Is)"
    python -V
    python -c 'import platform; print(platform.platform())'
    python -c 'import torch, transformers, onnx, onnxruntime
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
print("cuda", torch.version.cuda)'
    pip list 2>/dev/null | grep -Ei "torch|transformers|onnx|tokenizers|optimum|numpy|pillow|fastapi|uvicorn"
    nproc
    free -m
  } >"$LOG_DIR/versions.log" 2>&1
  say versions "PASS (log: $LOG_DIR/versions.log)"

  # 0. decisions that change what the numbers mean -------------------------
  run decide "$LOG_DIR/decide.log" python bench/prefill_probe.py --json-out "$LOG_DIR/decide.json"
  PREFILL_MODE="${PREFILL_MODE:-prompt}"
  if [ -f "$LOG_DIR/decide.json" ]; then
    PREFILL_MODE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("best_prefix","prompt"))' "$LOG_DIR/decide.json" 2>/dev/null || echo prompt)"
  fi
  if [ -f "$LOG_DIR/decide.log" ]; then
    grep -E "^(PREFIX|ONE_POSITION|ACTION)" "$LOG_DIR/decide.log" >>"$MASTER" 2>/dev/null
  fi

  # 1. export from the original weights ------------------------------------
  run export "$LOG_DIR/export.log" python export/export_onnx.py --out "$NEW" --prefill-mode "$PREFILL_MODE"
  if [ -f "$NEW/meta.json" ]; then
    cp "$NEW/meta.json" "$LOG_DIR/meta.json"
    python - "$NEW/meta.json" >>"$MASTER" 2>&1 <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print("meta: image_tokens=%s prompt_len=%s encoder_length=%s layers=%s prefill_mode=%s max_new=%s" % (
    meta.get("image_tokens"), len(meta.get("prompt_ids", [])),
    meta.get("encoder_length"), meta.get("decoder_layers"),
    meta.get("prefill_mode"), meta.get("max_new_tokens")))
print("prompt_text:", meta.get("prompt_text"))
PY
    du -sh "$NEW" >>"$MASTER" 2>&1
  fi

  # 2. token-level parity, old graphs vs new graphs (same PT loop) ----------
  if [ -d "$OLD" ]; then
    run parity "$LOG_DIR/parity.log" python bench/parity.py \
      --old "$OLD" --new "$NEW" --images "$IMAGES" --max-new 256
    grep -E '^(\{"check"|PARITY)' "$LOG_DIR/parity.log" >>"$MASTER" 2>/dev/null
  else
    say parity "SKIP (no previous export at $OLD to compare against)"
  fi

  # 3. unit tests ----------------------------------------------------------
  run tests "$LOG_DIR/pytest.log" python -m pytest tests -q
  run graphtests "$LOG_DIR/pytest_graphs.log" env MODEL_DIR="$NEW" python -m pytest tests/test_graphs.py -q

  # 4. the server, on the new graphs --------------------------------------
  say serve "starting on port $PORT (log: $LOG_DIR/server.log)"
  MODEL_DIR="$NEW" PORT="$PORT" python -m server.serve >"$LOG_DIR/server.log" 2>&1 &
  SERVERS+=($!)
  if wait_health "http://127.0.0.1:$PORT" 180; then
    BASE="http://127.0.0.1:$PORT"

    run first_request "$LOG_DIR/first_request.log" bash -c \
      "curl -sS -F 'image=@$IMAGES/82837252.png' $BASE/ocr"

    # crash-safety and status codes
    expect bad_file "$LOG_DIR/robustness.log" 0 bash -c \
      "curl -sS -o /dev/null -w '%{http_code}' -F 'image=@/etc/hostname' $BASE/ocr | grep -q 400"
    expect no_field "$LOG_DIR/robustness.log" 0 bash -c \
      "curl -sS -X POST $BASE/ocr | grep -q error"
    head -c 11534336 /dev/urandom >"$LOG_DIR/big.bin" 2>/dev/null
    expect too_big "$LOG_DIR/robustness.log" 0 bash -c \
      "curl -sS -o /dev/null -w '%{http_code}' -F 'image=@$LOG_DIR/big.bin' $BASE/ocr | grep -q 413"
    expect still_alive "$LOG_DIR/robustness.log" 0 bash -c \
      "curl -fsS $BASE/health | grep -q ok"

    # accuracy against the local provisional references
    if [ -f "$LAB/references-greedy-fp32.json" ]; then
      python - "$LAB/references-greedy-fp32.json" "$LOG_DIR/refs.json" <<'PY' 2>/dev/null
import json, sys
data = json.load(open(sys.argv[1]))
out = {i["filename"]: i["text"] for i in data.get("images", [])}
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
PY
      run accuracy "$LOG_DIR/accuracy.log" python -m bench.accuracy \
        --url "$BASE" --images "$IMAGES" --refs "$LOG_DIR/refs.json" --json-out "$LOG_DIR/accuracy.json"
      grep -E "mean accuracy|above 70" "$LOG_DIR/accuracy.log" >>"$MASTER" 2>/dev/null
    else
      say accuracy "SKIP (no local references)"
    fi

    # latency + concurrency sweeps
    run bench "$LOG_DIR/bench.log" python -m bench.load \
      --url "$BASE" --images "$IMAGES" --mode all --repeats 3 --clients 1,2,4,8 --seconds 20 \
      --json-out "$LOG_DIR/bench.json"
    grep -E '^\{"mode"' "$LOG_DIR/bench.log" >>"$MASTER" 2>/dev/null

    # memory with the model warm
    if [ -f "/proc/$(pgrep -f 'server.serve' | head -1)/status" ]; then
      {
        echo "--- server process memory (warm) ---"
        grep -E "VmRSS|VmHWM" "/proc/$(pgrep -f 'server.serve' | head -1)/status"
      } >>"$MASTER" 2>&1
    fi
  else
    tail -20 "$LOG_DIR/server.log" >>"$MASTER"
  fi
  kill "${SERVERS[-1]}" 2>/dev/null
fi

# 5. docker, under the official limits -------------------------------------
if [ "$RUN_DOCKER" = 1 ]; then
  IMAGE="florence2-ocr:runall"
  say docker "building $IMAGE (log: $LOG_DIR/docker_build.log)"
  if docker build -t "$IMAGE" . >>"$LOG_DIR/docker_build.log" 2>&1; then
    say docker "build PASS"
    say docker "running with --memory=4g --memory-swap=4g"
    docker run --rm --name runall-ocr -p "$PORT:8000" \
      --memory=4g --memory-swap=4g "$IMAGE" >"$LOG_DIR/docker_run.log" 2>&1 &
    SERVERS+=($!)
    START=$(date +%s)
    if wait_health "http://127.0.0.1:$PORT" 150; then
      say docker "cold start to healthy: $(( $(date +%s) - START ))s"
      run docker_request "$LOG_DIR/docker_request.log" bash -c \
        "curl -sS -F 'image=@$IMAGES/82837252.png' http://127.0.0.1:$PORT/ocr"
      run docker_bench "$LOG_DIR/docker_bench.log" python -m bench.load \
        --url "http://127.0.0.1:$PORT" --images "$IMAGES" --mode latency --repeats 2
      grep -E '^\{"mode"' "$LOG_DIR/docker_bench.log" >>"$MASTER" 2>/dev/null
      docker stats --no-stream runall-ocr >>"$MASTER" 2>&1
      docker exec runall-ocr python -c "import importlib.util as u; print('torch present:', bool(u.find_spec('torch')))" \
        >>"$MASTER" 2>&1
    else
      say docker "never became healthy under the 4 GB limit"
      docker logs --tail 40 runall-ocr >>"$MASTER" 2>&1
    fi
    docker stop runall-ocr >/dev/null 2>&1
  else
    say docker "build FAILED"
    FAILED=1
  fi
fi

say done "finished $(date -Is), failures=$FAILED"
say done "paste back: $MASTER"
exit "$FAILED"
