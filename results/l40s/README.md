# L40S Benchmark — Qwen3.5-35B-A3B-FP8

**Hardware:** NVIDIA L40S — 48 GB GDDR6, 864 GB/s memory bandwidth  
**Model:** `Qwen/Qwen3.5-35B-A3B-FP8`  
**Script:** [`run_sweep_l40s.sh`](../../run_sweep_l40s.sh)

---

## Memory budget

| Component | Size |
|---|---|
| GPU VRAM | 48 GB |
| vLLM allocation (`--gpu-memory-utilization 0.90`) | 43.2 GB |
| Qwen3.5-35B weights (FP8) | ~35 GB |
| KV cache headroom | ~8 GB |

The KV cache budget is tight. `MAX_MODEL_LEN` and `MAX_NUM_SEQS` directly trade off against each other — larger context windows leave less room for concurrent sequences.

---

## Optimal vLLM settings

### Recommended baseline

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --reasoning-parser deepseek_r1 \
  --enable-chunked-prefill \
  --trust-remote-code
```

**Why these values:**

| Flag | Value | Rationale |
|---|---|---|
| `--gpu-memory-utilization` | `0.90` | Maximises KV cache within the 48 GB envelope |
| `--max-model-len` | `32768` | Minimum for Qwen3.5 think mode to produce completions; 8192 overflows |
| `--max-num-seqs` | `8` | Conservative with 8 GB KV budget; raise to 32 once nothink baseline is established |
| `--kv-cache-dtype` | `fp8` | Halves KV memory vs fp16 — essential on this hardware |
| `--reasoning-parser` | `deepseek_r1` | Built-in; no plugin file needed (unlike Nemotron) |
| `--enable-chunked-prefill` | — | Improves throughput under concurrent load |

### Parameter tradeoffs

```
MAX_MODEL_LEN   MAX_NUM_SEQS   Think mode?   Expected ITL
──────────────────────────────────────────────────────────
8192            8 / 32         No (overflow)  ~22 ms
32768           8              Yes            ~22–30 ms
32768           32             Yes            ~25–35 ms (more batching)
65536           8              Yes            Risk of OOM — monitor VRAM
```

> **Think mode minimum:** `MAX_MODEL_LEN=32768` is the floor. At 8192, Qwen3.5 reasoning
> chains exhaust the context window before completing — throughput is effectively 0.

### Notes specific to Qwen3.5-FP8

- **No FP4 kernels** — do NOT set `VLLM_USE_FLASHINFER_MOE_FP4=1`. That flag is for
  Nemotron NVFP4 models only; it will degrade or crash Qwen3.5-FP8.
- **No plugin needed** — `deepseek_r1` is built into vLLM. You do not need
  `nano_v3_reasoning_parser.py` or any `--reasoning-parser-plugin` flag.
- **Nothink via chat_template_kwargs** — thinking is disabled per-request by the benchmark
  client using `"chat_template_kwargs": {"enable_thinking": false}`. No server-side change
  is needed to switch between think and nothink tasks.

---

## Running the sweep

### Prerequisites

```bash
# uv — fast Python package manager (installs locust on first run)
curl -LsSf https://astral.sh/uv/install.sh | sh

# .env must contain your Open WebUI credentials
# (copy from the project root; only two fields are needed here)
cp ../../.env.example .env
# then edit:
#   OPENWEBUI_API_KEY=<your key>
#   OPENWEBUI_KB_ID=<your knowledge base UUID>
#   OPENWEBUI_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
```

### Dry run (preview only)

```bash
bash ../../run_sweep_l40s.sh --dry-run
```

Shows the full plan — all combos, user levels, output paths — without touching any
process or writing any file.

### Full sweep

```bash
bash ../../run_sweep_l40s.sh
```

The script pauses before each parameter combo and prints the exact `vllm serve` flags
to use:

```
════════════════════════════════════════════════════════════
  COMBO: MAX_MODEL_LEN=32768   MAX_NUM_SEQS=8
  Output: results/l40s/qwen3.5-35b-fp8/mlen32768_seqs8

  Please start/restart vLLM with:
    --max-model-len 32768
    --max-num-seqs  8
    --gpu-memory-utilization 0.90
    --kv-cache-dtype fp8
    --reasoning-parser deepseek_r1
════════════════════════════════════════════════════════════
  Press Enter once vLLM is running and healthy, or 's' to skip this combo:
```

Once you press Enter, the script polls `http://localhost:8000/health` until vLLM
responds, then fires the locust user sweep automatically.

### Customising the grid

Edit the arrays at the top of `run_sweep_l40s.sh`:

```bash
MAX_MODEL_LENS=(8192 32768 65536)   # context window sizes to sweep
MAX_NUM_SEQS_LIST=(8 32)            # batch size values to sweep
USER_LEVELS=(10 20 30 50 100)       # concurrent user counts
```

Set `OPENWEBUI_HOST` if Open WebUI is not on `localhost:8888`:

```bash
OPENWEBUI_HOST=http://10.0.0.5:3000 bash ../../run_sweep_l40s.sh
```

Set `VLLM_HEALTH_URL` if vLLM is not on `localhost:8000`:

```bash
VLLM_HEALTH_URL=http://10.0.0.5:8000/health bash ../../run_sweep_l40s.sh
```

---

## Output structure

```
results/l40s/qwen3.5-35b-fp8/
  mlen8192_seqs8/
    nothink_u10_stats.csv
    nothink_u10_stats_history.csv
    nothink_u10.stdout.log
    ...
  mlen32768_seqs8/
    nothink_u10_stats.csv
    think_u10_stats.csv
    mixed_u10_stats.csv
    ...
  mlen32768_seqs32/
    ...
```

Think and mixed subdirectories only appear for `MAX_MODEL_LEN >= 32768`.

---

## Pilot results (10 users, nothink NT RAG, mlen=8192)

Collected during initial stack validation — short run, 5 completions only.

| Metric | Value |
|---|---|
| Throughput | 0.086 req/s |
| NT RAG E2E latency (avg) | 40.6 s |
| NT RAG TTFT (avg) | 38.1 s |
| ITL avg | 22 ms (~45 tok/s) |

The ~38 s TTFT is almost entirely RAG retrieval overhead (embed → hybrid search → rerank),
not model latency. NT PLAIN TTFT is expected to be sub-second at light load.

Compared to the same model on DGX Spark GB10: **~5× faster ITL** (22 ms vs 108 ms),
driven by dedicated GDDR6 bandwidth vs shared unified memory.
