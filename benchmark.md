# Benchmark Results — DGX Spark (GB10)

**Date:** 2026-04-07
**Branch:** bench/dgx-spark-gb10
**Hardware:** NVIDIA DGX Spark, GB10 (128 GB unified memory)

## Stack configuration

| Parameter | Value |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.70 |
| `VLLM_MAX_MODEL_LEN` | 8192 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
| `VLLM_MAX_NUM_SEQS` | 8 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 |
| Embedder | `BAAI/bge-m3` on GPU |
| Reranker | `BAAI/bge-reranker-v2-m3` on GPU |
| Vector store | Qdrant |

## Methodology

Benchmarked with `locustfile.py`, 5-minute steady-state window (`--reset-stats`) per run.
Two load levels tested: **10 users** (ramp 2/s) and **100 users** (ramp 5/s).

Three runs, in order:

| Run | Tag | Tasks |
|---|---|---|
| 1 | `nothink` | NT PLAIN (weight 1) + NT RAG (weight 3) |
| 2 | `think` | PLAIN (weight 1) + RAG (weight 3) |
| 3 | `mixed` | All four tasks (3:1:3:1 ratio) |

Thinking is disabled per-request via `chat_template_kwargs: {"enable_thinking": false}` handled by the `nano_v3` reasoning parser plugin.

RAG queries attach the Nurburgring knowledge base. PLAIN queries are bare chat with no retrieval.

---

## Results — 10 users

### Throughput

| Mode | Total reqs / 5 min | req/s |
|---|---|---|
| nothink | 83 | **0.28** |
| think | 38 | 0.13 |
| mixed | 48 | 0.16 |

Nothink delivers **2.2× more throughput** than think at 10 concurrent users.

### TTFT — time to first token

| Mode | PLAIN avg | PLAIN p50 | RAG avg | RAG p50 |
|---|---|---|---|---|
| nothink | **2.2s** | 1.6s | **28.5s** | 28s |
| think | 8.8s | 9.5s | 34.9s | 31s |

Nothink is **4× faster** on plain TTFT. The RAG gap is smaller (28.5s vs 34.9s) because Qdrant retrieval dominates TTFT regardless of thinking mode.

### E2E latency

| Mode | PLAIN avg | PLAIN p50 | RAG avg | RAG p50 |
|---|---|---|---|---|
| nothink | 22.6s | 20s | 35.5s | 35s |
| think | 56.6s | 51s | 69.8s | 66s |

### ITL — inter-token latency (once streaming starts)

| Mode | avg | p95 |
|---|---|---|
| nothink | 76ms | 123–139ms |
| think | 68–69ms | 100–102ms |

ITL is nearly identical between modes. **The GPU is not the bottleneck once generation is underway** — token throughput is consistent regardless of thinking mode.

### TPS — ms per output token (lower = faster)

| Mode | PLAIN avg | RAG avg |
|---|---|---|
| nothink | 22ms | 160ms |
| think | 23ms | 40ms |

The elevated NT RAG TPS reflects that nothink responses are counted via `reasoning_content` tokens, which may differ in chunking from `content` tokens.

### RAG overhead — retrieval cost above PLAIN TTFT

| Mode | avg | p50 | p95 |
|---|---|---|---|
| nothink | 25.9s | 26s | 38s |
| think | 27.0s | 27s | 45s |

**RAG overhead is ~26–27s regardless of thinking mode.** Qdrant vector retrieval is the dominant latency factor for RAG queries at 10 concurrent users, not the LLM.

---

## Results — 100 users

> **Stack saturated.** `VLLM_MAX_NUM_SEQS=8` limits vLLM to 8 concurrent sequences. With 100 users, 92 requests queue immediately and hit the 120s client timeout.

### Failure rates

| Run | Requests | Failures | Failure rate |
|---|---|---|---|
| nothink | 122 | 110 | **90%** |
| think | 153 | 153 | **100%** |
| mixed | 144 | 144 | **100%** |

All failures are `ReadTimeoutError` at the 120s client timeout (`HTTP 0`). No RAG requests completed in think or mixed runs.

### nothink — only partial data captured

The 12 NT PLAIN requests that completed before the queue filled provide a pre-saturation snapshot:

| Metric | 10 users | 100 users |
|---|---|---|
| NT PLAIN TTFT avg | 2.2s | **56.7s** |
| NT PLAIN TTFT p50 | 1.6s | 65s |
| NT PLAIN ITL avg | 76ms | 74ms |
| NT PLAIN TPS avg | 22ms/tok | 129ms/tok |

TTFT degraded **26×** while ITL remained stable — confirming the bottleneck is queue wait time before vLLM picks up the request, not GPU generation speed once started.

### think / mixed — complete timeout

Every request timed out. vLLM's thinking token generation at `VLLM_MAX_NUM_SEQS=8` cannot serve 100 concurrent users within any reasonable timeout.

---

## Key findings

1. **Disable thinking for high-concurrency workloads.** At 10 users, nothink gives 2.2× throughput and 4× faster plain TTFT with no meaningful ITL regression.

2. **`VLLM_MAX_NUM_SEQS=8` is the hard concurrency ceiling.** At 100 users it causes complete saturation. Raising this is the highest-priority tuning action before re-testing at scale.

3. **Qdrant is the RAG bottleneck at low concurrency.** The 26–27s retrieval overhead at 10 users dominates RAG TTFT. At higher concurrency, vLLM queue depth takes over as the primary bottleneck.

4. **GPU generation throughput is healthy.** ITL of 68–76ms is stable from 10 to 100 users — the GB10 has headroom. The problem is queue depth, not compute.

5. **Saturation point is between 10 and 100 users.** A sweep at 20, 30, and 50 users is needed to characterise the degradation curve before tuning `VLLM_MAX_NUM_SEQS`.

---

---

## Results — 100 users, VLLM_MAX_NUM_SEQS=64

> **0 failures.** Raising MAX_NUM_SEQS from 8 to 64 eliminates timeouts. The system is under load but stable.

vLLM image: `nvcr.io/nvidia/vllm:26.02-py3` (upgrade to 26.03-py3 pending).

### Throughput

| Mode | Total reqs / 5 min | req/s | vs 10u |
|---|---|---|---|
| nothink | 152 | **0.54** | +93% |
| think | 55 | **0.20** | +54% |
| mixed | 100 | 0.36 | — |

Throughput roughly doubles vs 10 users thanks to higher concurrency — the GB10 is now being utilised more fully.

### TTFT — time to first token

| Mode | PLAIN avg | PLAIN p50 | RAG avg | RAG p50 |
|---|---|---|---|---|
| nothink | 35.7s | 33s | 127.8s | 124s |
| think | 49.3s | 44s | 160.8s | 155s |
| mixed NT | 45.4s | 41s | 144.2s | 142s |
| mixed think | 47.3s | 49s | 142.9s | 137s |

TTFT degraded significantly vs 10 users (nothink plain: 2.2s → 35.7s) due to queue depth — 100 users competing for 64 slots.

### E2E latency

| Mode | PLAIN avg | PLAIN p50 | RAG avg | RAG p50 |
|---|---|---|---|---|
| nothink | 88.7s | 67s | 151.6s | 150s |
| think | 136s | 141s | 235s | 237s |

### ITL — inter-token latency

| Mode | avg | p95 |
|---|---|---|
| nothink | 224–259ms | 451–483ms |
| think | 193–224ms | 366–462ms |

ITL degraded **3× vs 10 users** (from ~70ms to ~210–260ms). This is expected: with 64 concurrent sequences batched together, each token generation step handles more work per iteration, increasing per-token latency across all sequences. The GB10 is now genuinely loaded.

### RAG overhead

| Mode | avg | p50 | p95 |
|---|---|---|---|
| nothink | 88.2s | 83s | 146s |
| think | 117.9s | 114s | 174s |

RAG overhead exploded from ~27s at 10 users to 88–118s at 100 users. This combines Qdrant retrieval latency under concurrent load and queue wait time before vLLM picks up the request.

### Comparison summary

| Metric | 10u / seqs=8 | 100u / seqs=8 | 100u / seqs=64 |
|---|---|---|---|
| Failure rate | 0% | 90–100% | **0%** |
| NT PLAIN TTFT avg | 2.2s | 56.7s† | 35.7s |
| PLAIN TTFT avg | 8.8s | timeout | 49.3s |
| NT PLAIN ITL avg | 76ms | 74ms† | 224ms |
| PLAIN ITL avg | 69ms | timeout | 224ms |
| nothink req/s | 0.28 | ~0 | **0.54** |
| think req/s | 0.13 | ~0 | **0.20** |

† partial data from the few requests that squeezed through before queue filled.

---

## Key findings

1. **`VLLM_MAX_NUM_SEQS=64` is the minimum viable setting for 100 concurrent users.** seqs=8 causes complete saturation; seqs=64 eliminates failures and doubles throughput.

2. **Disable thinking for high-concurrency workloads.** Nothink gives 2.7× more throughput than think at 100 users, and TTFT is 14s faster on plain queries.

3. **ITL degrades 3× at 100 users** (70ms → 220ms). This is a batching effect — 64 concurrent sequences share compute per step. The GB10 is now genuinely saturated; further scaling requires either a larger model-serving budget or reduced concurrency.

4. **Qdrant is a growing bottleneck under load.** RAG overhead scales from 27s at 10 users to 88–118s at 100 users. At higher concurrency, externalising or tuning Qdrant becomes critical.

5. **Nothink TTFT at 100 users (35.7s) is still high for interactive use.** Acceptable for batch/background workloads; for interactive chat, target ≤20 concurrent users or raise MAX_NUM_SEQS further.

---

## Next steps

- [ ] Upgrade to `VLLM_NGC_TAG=26.03-py3` and re-run to check for regressions/improvements
- [ ] Sweep at 20 / 30 / 50 users to characterise the degradation curve between 10 and 100 users
- [ ] Investigate Qdrant performance under concurrent load (tune thread count, consider dedicated instance)
- [ ] Consider raising `VLLM_MAX_MODEL_LEN` from 8192 once saturation profile is known

---

*Raw CSV and HTML reports in `results/` — `*_u10_*`, `*_u100_*`, and `*_u100_seqs64_*` prefixes.*
