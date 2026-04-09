#!/usr/bin/env bash
# Full benchmark sweep — supports multiple hardware targets.
# Usage: bash run_sweep.sh [TARGET [PHASE]]
#   TARGET: dgx-spark-gb10 (default) | rtx-pro-6000
#   PHASE:  all (default) | nemotron | qwen
#
# Prerequisites:
#   - .env already configured for the target hardware (cp .env.<TARGET>.nemotron-nano .env)
#   - OPENWEBUI_API_KEY and OPENWEBUI_KB_ID set in .env
#   - vLLM container healthy before running
#   - uv installed

set -euo pipefail
cd "$(dirname "$0")"

TARGET=${1:-dgx-spark-gb10}
PHASE=${2:-all}

case "$TARGET" in
  dgx-spark-gb10)
    GPU_MEM=0.70
    NEMOTRON_MAX_LEN=8192
    QWEN_MAX_LEN=32768   # raised from 8192 — Qwen3.5 think chains overflow at 8192
    ;;
  rtx-pro-6000)
    GPU_MEM=0.55
    NEMOTRON_MAX_LEN=16384
    QWEN_MAX_LEN=16384
    ;;
  *)
    echo "Unknown target: $TARGET. Valid values: dgx-spark-gb10, rtx-pro-6000" >&2
    exit 1
    ;;
esac

HOST="http://localhost:3000"
LOCUST="./locustfile.py"

# ── helpers ──────────────────────────────────────────────────────────────────

wait_healthy() {
    local svc=${1:-vllm}
    echo "[sweep] waiting for $svc to be healthy..."
    until [ "$(docker inspect --format='{{.State.Health.Status}}' "$svc" 2>/dev/null)" = "healthy" ]; do
        sleep 15
    done
    echo "[sweep] $svc is healthy"
}

run_locust() {
    local tag=$1 users=$2 ramp=$3 outdir=$4
    local prefix="${tag}_u${users}"
    # mixed = no tag filter (all four tasks); nothink/think = filtered
    local tag_args=(); [[ "$tag" != "mixed" ]] && tag_args=(-T "$tag")
    echo "[sweep] $(date '+%H:%M:%S') ▶ $prefix (tag=$tag, users=$users)"
    uv run "$LOCUST" --host "$HOST" \
        --headless -u "$users" -r "$ramp" --run-time 5m --reset-stats \
        --csv="$outdir/$prefix" "${tag_args[@]}" \
        >"$outdir/${prefix}.stdout.log" 2>>"$outdir/${prefix}.stderr.log"
    echo "[sweep] $(date '+%H:%M:%S') ✓ $prefix done"
}

sweep() {
    local outdir=$1
    mkdir -p "$outdir"
    for tag in nothink think mixed; do
        run_locust "$tag" 10  2 "$outdir"
        run_locust "$tag" 20  2 "$outdir"
        run_locust "$tag" 30  5 "$outdir"
        run_locust "$tag" 50  5 "$outdir"
        run_locust "$tag" 100 10 "$outdir"
    done
}

switch_model() {
    local model=$1 parser_plugin=$2 parser=$3 fp4=$4 max_len=$5
    echo "[sweep] switching vLLM to $model (MAX_MODEL_LEN=$max_len)"
    echo "[sweep] NOTE: vLLM must unload the current model and load the new one."
    echo "[sweep]       This typically takes 10–20 min (download + warm-up). Do not interrupt."
    sed -i "s|^VLLM_MODEL=.*|VLLM_MODEL=$model|"                                             .env
    sed -i "s|^OPENWEBUI_MODEL=.*|OPENWEBUI_MODEL=$model|"                                   .env
    sed -i "s|^VLLM_REASONING_PARSER_PLUGIN=.*|VLLM_REASONING_PARSER_PLUGIN=$parser_plugin|" .env
    sed -i "s|^VLLM_REASONING_PARSER=.*|VLLM_REASONING_PARSER=$parser|"                      .env
    sed -i "s|^VLLM_USE_FLASHINFER_MOE_FP4=.*|VLLM_USE_FLASHINFER_MOE_FP4=$fp4|"            .env
    sed -i "s|^VLLM_GPU_MEMORY_UTILIZATION=.*|VLLM_GPU_MEMORY_UTILIZATION=$GPU_MEM|"         .env
    sed -i "s|^VLLM_MAX_MODEL_LEN=.*|VLLM_MAX_MODEL_LEN=$max_len|"                           .env
    sed -i "s|^VLLM_KV_CACHE_DTYPE=.*|VLLM_KV_CACHE_DTYPE=fp8|"                             .env
    docker compose up -d vllm
    wait_healthy vllm
    echo "[sweep] restarting open-webui so it picks up the new model"
    docker compose restart open-webui
    wait_healthy open-webui
}

# ── Phase 1: Nemotron ─────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "nemotron" ]]; then
    echo "=== [$TARGET] Phase 1: Nemotron-3-Nano-30B-A3B-NVFP4 ==="
    wait_healthy vllm
    sweep "results/$TARGET/nemotron-nano"
fi

# ── Phase 2: Qwen3.5-35B-A3B-FP8 ────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "qwen" ]]; then
    echo "=== [$TARGET] Phase 2: Qwen3.5-35B-A3B-FP8 ==="
    switch_model \
        "Qwen/Qwen3.5-35B-A3B-FP8" \
        "/vllm_plugins/noop.py" \
        "qwen3" \
        "0" \
        "$QWEN_MAX_LEN"
    sweep "results/$TARGET/qwen3.5-35b-fp8"

    # ── Phase 3: restore Nemotron ─────────────────────────────────────────────
    echo "=== [$TARGET] Phase 3: restoring Nemotron ==="
    switch_model \
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
        "/vllm_plugins/nano_v3_reasoning_parser.py" \
        "nano_v3" \
        "1" \
        "$NEMOTRON_MAX_LEN"
fi

echo "=== sweep complete ==="
