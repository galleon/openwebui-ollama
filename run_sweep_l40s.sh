#!/usr/bin/env bash
# L40S parameter sweep — Qwen3.5-35B-A3B-FP8 only
#
# Sweeps a 2-D grid of vLLM parameters:
#   VLLM_MAX_MODEL_LEN  × VLLM_MAX_NUM_SEQS
#
# For each combo the script pauses and asks you to restart vLLM with the
# right settings, then runs locust at every user level (nothink always;
# think + mixed when MAX_MODEL_LEN is large enough for reasoning chains).
# CSVs are saved under results/l40s/qwen3.5-35b-fp8/mlen<L>_seqs<S>/
#
# Usage:
#   bash run_sweep_l40s.sh [--dry-run]
#
# Prerequisites:
#   - OPENWEBUI_API_KEY and OPENWEBUI_KB_ID exported (or in .env)
#   - vLLM already running and reconfigured before confirming each combo
#   - uv installed

set -euo pipefail
cd "$(dirname "$0")"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Load .env for API key / KB ID if not already in environment ───────────────
if [[ -f .env ]]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | grep -E '^(OPENWEBUI_API_KEY|OPENWEBUI_KB_ID|OPENWEBUI_MODEL)=' | xargs)
fi

# ── Parameter grid ────────────────────────────────────────────────────────────
# Edit these arrays to change the sweep.
#
# MAX_MODEL_LEN choices:
#   8192   — GB10 baseline; think chains overflow the window (0 completions)
#   32768  — allows short-to-medium reasoning chains
#   65536  — long chains, but tight on 48 GB (< 1 GB KV budget at 0.90 util); may OOM
#
# MAX_NUM_SEQS choices:
#   8    — conservative; low memory pressure, high per-request throughput
#   32   — aggressive batching; better aggregate req/s at high concurrency
MAX_MODEL_LENS=(8192 32768 65536)
MAX_NUM_SEQS_LIST=(8 32)

# Minimum MAX_MODEL_LEN for which think/mixed tasks are included.
# Below this threshold Qwen3.5 reasoning chains overflow → 0 completions.
THINK_MIN_LEN=32768

# User levels for the locust sweep.
USER_LEVELS=(10 20 30 50 100)

ramp_for() {
    case $1 in
        10|20) echo 2 ;;
        30|50) echo 5 ;;
        *)     echo 10 ;;
    esac
}

HOST="${OPENWEBUI_HOST:-http://localhost:8888}"
LOCUST="./locustfile.py"
RESULTS_ROOT="results/l40s/qwen3.5-35b-fp8"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[sweep] $(date '+%H:%M:%S') $*"; }

wait_for_ready() {
    # Poll the vLLM health endpoint directly — no Docker required.
    local url="${VLLM_HEALTH_URL:-http://localhost:8000/health}"
    log "Polling $url until vLLM responds healthy..."
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 10
    done
    log "vLLM is responding."
}

run_locust() {
    local tag=$1 users=$2 outdir=$3
    local ramp; ramp=$(ramp_for "$users")
    local prefix="${tag}_u${users}"
    local tag_args=()
    [[ "$tag" != "mixed" ]] && tag_args=(-T "$tag")

    log "▶ $prefix  (tag=$tag, users=$users, ramp=$ramp/s)"

    if $DRY_RUN; then
        log "[dry-run] would run: locust $prefix -> $outdir"
        return
    fi

    mkdir -p "$outdir"
    uv run "$LOCUST" --host "$HOST" \
        --headless -u "$users" -r "$ramp" --run-time 5m --reset-stats \
        --csv="$outdir/$prefix" "${tag_args[@]}" \
        > "$outdir/${prefix}.stdout.log" 2>> "$outdir/${prefix}.stderr.log"

    log "✓ $prefix done"
}

run_combo() {
    local max_len=$1 max_seqs=$2
    local outdir="${RESULTS_ROOT}/mlen${max_len}_seqs${max_seqs}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  COMBO: MAX_MODEL_LEN=$max_len   MAX_NUM_SEQS=$max_seqs"
    echo "  Output: $outdir"
    echo ""
    echo "  Please start/restart vLLM with:"
    echo "    --max-model-len $max_len"
    echo "    --max-num-seqs  $max_seqs"
    echo "    --gpu-memory-utilization 0.90   (recommended for L40S 48 GB GDDR6)"
    echo "    --kv-cache-dtype fp8"
    echo "    --reasoning-parser deepseek_r1  (no plugin needed for Qwen3.5)"
    echo "════════════════════════════════════════════════════════════"

    if $DRY_RUN; then
        log "[dry-run] skipping wait and locust runs"
    else
        read -rp "  Press Enter once vLLM is running and healthy, or 's' to skip this combo: " ans
        [[ "$ans" == "s" ]] && { log "Skipping combo."; return; }
        wait_for_ready
    fi

    # nothink — always run (viable at any MAX_MODEL_LEN)
    log "── nothink sweep ──"
    for u in "${USER_LEVELS[@]}"; do
        run_locust nothink "$u" "$outdir"
    done

    # think + mixed — only when MAX_MODEL_LEN is large enough
    if (( max_len >= THINK_MIN_LEN )); then
        log "── think sweep (MAX_MODEL_LEN=$max_len >= $THINK_MIN_LEN) ──"
        for u in "${USER_LEVELS[@]}"; do
            run_locust think "$u" "$outdir"
        done

        log "── mixed sweep ──"
        for u in "${USER_LEVELS[@]}"; do
            run_locust mixed "$u" "$outdir"
        done
    else
        log "── skipping think/mixed (MAX_MODEL_LEN=$max_len < $THINK_MIN_LEN) ──"
    fi

    log "COMBO done → $outdir"
}

# ── Plan summary ──────────────────────────────────────────────────────────────

think_combos=0
for L in "${MAX_MODEL_LENS[@]}"; do
    (( L >= THINK_MIN_LEN )) && (( think_combos++ )) || true
done
nothink_only_combos=$(( ${#MAX_MODEL_LENS[@]} - think_combos ))
total_combos=$(( ${#MAX_MODEL_LENS[@]} * ${#MAX_NUM_SEQS_LIST[@]} ))

total_runs=$(( (nothink_only_combos * ${#MAX_NUM_SEQS_LIST[@]} * ${#USER_LEVELS[@]}) + \
               (think_combos       * ${#MAX_NUM_SEQS_LIST[@]} * ${#USER_LEVELS[@]} * 3) ))

echo ""
echo "L40S Qwen3.5-35B-FP8 parameter sweep"
echo "  MAX_MODEL_LENS : ${MAX_MODEL_LENS[*]}"
echo "  MAX_NUM_SEQS   : ${MAX_NUM_SEQS_LIST[*]}"
echo "  User levels    : ${USER_LEVELS[*]}"
echo "  Think min len  : $THINK_MIN_LEN"
echo "  Total combos   : $total_combos"
echo "  Total locust runs: $total_runs  (~5 min each → ~$(( total_runs * 5 )) min locust time)"
echo "  Results under  : $RESULTS_ROOT/mlen<L>_seqs<S>/"
echo ""

if $DRY_RUN; then
    echo "DRY RUN — no network calls, no CSVs written."
    echo ""
fi

# ── Main sweep loop ───────────────────────────────────────────────────────────

START=$(date +%s)

for max_len in "${MAX_MODEL_LENS[@]}"; do
    for max_seqs in "${MAX_NUM_SEQS_LIST[@]}"; do
        run_combo "$max_len" "$max_seqs"
    done
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
log "════════════════════════════════════════"
log "Sweep complete. Elapsed: ${ELAPSED} min"
log "Results: $RESULTS_ROOT/mlen<L>_seqs<S>/"
log "════════════════════════════════════════"
