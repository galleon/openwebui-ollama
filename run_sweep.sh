#!/usr/bin/env bash
# Full benchmark sweep — supports multiple hardware targets.
# Usage: bash run_sweep.sh [TARGET [PHASE]]
#   TARGET: dgx-spark-gb10 (default) | rtx-pro-6000 | l40s
#   PHASE:  all (default) | nemotron | qwen
#
# Prerequisites:
#   - .env already configured for the target hardware (cp .env.<TARGET>.* .env)
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
    USER_LADDER=(10 20 30 50 100)
    RAMP_LADDER=(2  2  5  5  10)
    ;;
  rtx-pro-6000)
    GPU_MEM=0.55
    NEMOTRON_MAX_LEN=16384
    QWEN_MAX_LEN=16384
    USER_LADDER=(10 20 30 50 100)
    RAMP_LADDER=(2  2  5  5  10)
    ;;
  l40s)
    GPU_MEM=0.90
    NEMOTRON_MAX_LEN=8192
    QWEN_MAX_LEN=8192    # KV budget ~8 GB on L40S — keep context short
    USER_LADDER=(5 10 15 20 30)
    RAMP_LADDER=(1  2  2  2   5)
    ;;
  *)
    echo "Unknown target: $TARGET. Valid values: dgx-spark-gb10, rtx-pro-6000, l40s" >&2
    exit 1
    ;;
esac

WEBUI_PORT=$(grep '^WEBUI_PORT=' .env | cut -d= -f2)
HOST="http://localhost:${WEBUI_PORT:-3000}"
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

smoke_test() {
    # One-shot check: send a minimal completion through the full stack.
    # Model readiness is already guaranteed by wait_healthy (healthcheck hits
    # /v1/models, not just /health).  This catches config problems like a wrong
    # model name or a missing API key before wasting benchmark time.
    local model api_key http_code
    model=$(grep '^OPENWEBUI_MODEL=' .env | cut -d= -f2)
    api_key=$(grep '^OPENWEBUI_API_KEY=' .env | cut -d= -f2)
    echo "[sweep] smoke-testing $HOST with model $model ..."
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$HOST/api/chat/completions" \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"stream\":false}" \
        --max-time 60 2>/dev/null || echo "000")
    if [[ "$http_code" != "200" ]]; then
        echo "[sweep] ERROR: smoke test got HTTP $http_code (model=$model) — aborting sweep" >&2
        exit 1
    fi
    echo "[sweep] smoke test passed (HTTP 200)"
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
        for i in "${!USER_LADDER[@]}"; do
            run_locust "$tag" "${USER_LADDER[$i]}" "${RAMP_LADDER[$i]}" "$outdir"
        done
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
    smoke_test
}

# ── Phase 1: Nemotron ─────────────────────────────────────────────────────────

if [[ "$TARGET" != "l40s" ]] && [[ "$PHASE" == "all" || "$PHASE" == "nemotron" ]]; then
    echo "=== [$TARGET] Phase 1: Nemotron-3-Nano-30B-A3B-NVFP4 ==="
    wait_healthy vllm
    smoke_test
    sweep "results/$TARGET/nemotron-nano"
fi

# ── Phase 2: Qwen3.5 ─────────────────────────────────────────────────────────

if [[ "$PHASE" == "all" || "$PHASE" == "qwen" ]]; then
    if [[ "$TARGET" == "l40s" ]]; then
        QWEN_MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
        QWEN_PARSER="qwen3"
        QWEN_FP4="0"
        QWEN_OUTDIR="results/$TARGET/qwen3.5-35b-fp8"
    else
        QWEN_MODEL="AxionML/Qwen3.5-35B-A3B-NVFP4"
        QWEN_PARSER="qwen3"
        QWEN_FP4="1"
        QWEN_OUTDIR="results/$TARGET/qwen3.5-35b-nvfp4"
    fi

    echo "=== [$TARGET] Phase 2: $QWEN_MODEL ==="
    if [[ "$TARGET" == "l40s" ]]; then
        wait_healthy vllm  # l40s starts already configured for Qwen FP8
        smoke_test
    else
        switch_model "$QWEN_MODEL" "/vllm_plugins/noop.py" "$QWEN_PARSER" "$QWEN_FP4" "$QWEN_MAX_LEN"
    fi
    sweep "$QWEN_OUTDIR"

    # ── Phase 3: restore Nemotron (non-l40s only) ─────────────────────────────
    if [[ "$TARGET" != "l40s" ]]; then
        echo "=== [$TARGET] Phase 3: restoring Nemotron ==="
        switch_model \
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" \
            "/vllm_plugins/nano_v3_reasoning_parser.py" \
            "nano_v3" \
            "1" \
            "$NEMOTRON_MAX_LEN"
    fi
fi

echo "=== sweep complete ==="
