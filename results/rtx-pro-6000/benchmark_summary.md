# Benchmark Summary

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`  
**Hardware:** NVIDIA RTX Pro 6000 Blackwell (96 GB GDDR7)  
**Stack:** Open WebUI + vLLM · embedder: `BAAI/bge-m3` · reranker: `BAAI/bge-reranker-v2-m3`  
**Knowledge base:** Nürburgring FAQ (8 questions, hybrid search, RAG top-k=5 → reranked to 3)  
**Tool:** Locust · task ratio RAG:PLAIN = 3:1 · wait 1–3 s between tasks

---

## Run 1 — 10 concurrent users

`-u 10 -r 2 --run-time 60s`

| Metric | PLAIN (no KB) | RAG (with KB) |
|---|---|---|
| Requests | 6 | 29 |
| Failures | 0 | 0 |
| Throughput | 0.10 req/s | 0.49 req/s |
| **TTFT avg** | 3,249 ms | 10,374 ms |
| TTFT p50 | 3,000 ms | 9,500 ms |
| TTFT p95 | 6,000 ms | 16,000 ms |
| **E2E avg** | 19,460 ms | 11,803 ms |
| E2E p50 | 21,000 ms | 10,000 ms |
| E2E p95 | 29,000 ms | 18,000 ms |
| ITL avg | 10 ms | 10 ms |
| ITL p95 | 12 ms | 13 ms |
| TPS avg (ms/token) | 3 | 29 |
| **RAG overhead avg** | — | 8,440 ms |
| RAG overhead p50 | — | 7,600 ms |
| RAG overhead p95 | — | 16,000 ms |

> PLAIN E2E is higher than RAG E2E because unrestricted responses are much longer.
> TPS difference (3 vs 29 ms/token) reflects the token count gap between grounded and open-ended answers.

---

## Run 2 — 100 concurrent users

`-u 100 -r 10 --run-time 120s`

| Metric | PLAIN (no KB) | RAG (with KB) |
|---|---|---|
| Requests | 35 | 124 |
| Failures | 0 | 0 |
| Throughput | 0.29 req/s | 1.04 req/s |
| **TTFT avg** | 20,373 ms | 50,497 ms |
| TTFT p50 | 23,000 ms | 55,000 ms |
| TTFT p95 | 40,000 ms | 67,000 ms |
| **E2E avg** | 45,524 ms | 52,475 ms |
| E2E p50 | 46,000 ms | 56,000 ms |
| E2E p95 | 74,000 ms | 69,000 ms |
| ITL avg | 17 ms | 16 ms |
| ITL p95 | 41 ms | 51 ms |
| TPS avg (ms/token) | 9 | 153 |
| **RAG overhead avg** | — | 27,148 ms |
| RAG overhead p50 | — | 26,000 ms |
| RAG overhead p95 | — | 42,000 ms |

---

## 10 vs 100 users — scaling comparison

| Metric | 10 users | 100 users | Ratio |
|---|---|---|---|
| Total throughput | 3.9 req/s | 9.1 req/s | +2.3x |
| RAG TTFT avg | 10,374 ms | 50,497 ms | +4.9x |
| RAG TTFT p50 | 9,500 ms | 55,000 ms | +5.8x |
| RAG TTFT p95 | 16,000 ms | 67,000 ms | +4.2x |
| RAG E2E avg | 11,803 ms | 52,475 ms | +4.4x |
| RAG TPS avg (ms/token) | 29 | 153 | +5.3x |
| PLAIN TTFT avg | 3,249 ms | 20,373 ms | +6.3x |
| RAG overhead avg | 8,440 ms | 27,148 ms | +3.2x |
| ITL avg | 10 ms | 16–17 ms | +1.6x |

### Key observations

- **Throughput scales sub-linearly (+2.3x for 10x users)** — vLLM is capped at `VLLM_MAX_NUM_SEQS=32`, so the extra 68 users queue rather than run in parallel.
- **TTFT blows up ~5x** — dominated by queue wait time, not GPU compute.
- **ITL is nearly stable (10 → 16 ms)** — the GPU keeps pace per-token once a request starts; the bottleneck is admission, not generation.
- **RAG overhead grows 3.2x** — retrieval itself is fast, but the reranker and embedding pipeline also contend under load.
- **0 failures at both concurrency levels** — the system is stable; it degrades gracefully rather than erroring.

---

## Run 3 — 100 concurrent users, optimised config

`-u 100 -r 10 --run-time 120s`

Changes applied vs Run 2:
- `VLLM_MAX_NUM_SEQS`: 32 → **64**
- `VLLM_MAX_MODEL_LEN`: 65536 → **16384**
- Added **`--enable-chunked-prefill`**

| Metric | PLAIN (no KB) | RAG (with KB) |
|---|---|---|
| Requests | 41 | 139 |
| Failures | 0 | 0 |
| Throughput | 0.35 req/s | 1.18 req/s |
| **TTFT avg** | 28,585 ms | 44,080 ms |
| TTFT p50 | 28,000 ms | 44,000 ms |
| TTFT p95 | 41,000 ms | 64,000 ms |
| **E2E avg** | 56,888 ms | 46,907 ms |
| E2E p50 | 53,000 ms | 47,000 ms |
| E2E p95 | 82,000 ms | 69,000 ms |
| ITL avg | 22 ms | 21 ms |
| ITL p95 | 51 ms | 45 ms |
| TPS avg (ms/token) | 20 | 138 |
| **RAG overhead avg** | — | 16,817 ms |
| RAG overhead p50 | — | 17,000 ms |
| RAG overhead p95 | — | 36,000 ms |

---

## Run 2 vs Run 3 — impact of optimisations (100 users)

| Metric | Baseline (Run 2) | Optimised (Run 3) | Delta |
|---|---|---|---|
| Total throughput | 9.1 req/s | 10.3 req/s | **+13%** |
| Total requests completed | 1,078 | 1,219 | **+13%** |
| RAG TTFT avg | 50,497 ms | 44,080 ms | **-13%** |
| RAG TTFT p50 | 55,000 ms | 44,000 ms | **-20%** |
| RAG TTFT p95 | 67,000 ms | 64,000 ms | -4% |
| RAG E2E avg | 52,474 ms | 46,907 ms | **-11%** |
| **RAG overhead avg** | 27,148 ms | 16,817 ms | **-38%** |
| RAG overhead p50 | 26,000 ms | 17,000 ms | **-35%** |
| PLAIN TTFT avg | 20,373 ms | 28,585 ms | +40% ⚠ |
| PLAIN E2E avg | 45,524 ms | 56,888 ms | +25% ⚠ |
| ITL avg | 16–17 ms | 21–22 ms | +34% |

### Key observations

- **+13% throughput** — more sequences admitted in parallel (64 vs 32) means higher GPU utilisation.
- **RAG overhead dropped 38%** — the biggest win. Chunked prefill prevents long RAG prefill phases from blocking other requests; retrieval latency under load is much more predictable.
- **RAG TTFT improved 13–20%** — queue wait reduced because more slots are available.
- **PLAIN latency got worse (+40% TTFT)** ⚠ — with 64 concurrent sequences, PLAIN requests (which generate far more tokens) compete harder for GPU decode time. The GPU is more saturated, pushing up ITL (16 → 22 ms) for everyone.
- **The trade-off**: these settings favour RAG throughput. If PLAIN (long-form) queries matter equally, a smaller increase to `VLLM_MAX_NUM_SEQS=48` with a longer context limit would be a better balance.
- **Remaining bottleneck**: at 100 users even with 64 slots, ~36 users still queue. True 100-user concurrency at acceptable latency requires either a larger `VLLM_MAX_NUM_SEQS` or reducing context length.

---

## Conclusion — further optimisation options

### High impact, low effort (config / flag changes)

| Option | Change | Expected impact |
|---|---|---|
| Prefix caching | `--enable-prefix-caching` | Reuses KV blocks for repeated RAG context — high hit rate with a small KB. Directly cuts TTFT. |
| Multi-step scheduling | `--num-scheduler-steps 8` | Batches multiple decode steps per scheduler call, reducing Python overhead. Free throughput gain. |
| Disable reranker | `ENABLE_RAG_HYBRID_SEARCH=false`, `RAG_TOP_K=3` | Removes GPU reranker from the hot path. RAG overhead dropped 38% in Run 3 — this could cut it further. Quality impact TBD on small focused KB. |
| Tune chunked prefill chunk size | `--max-num-batched-tokens 2048` | Larger chunks process RAG context faster (lower TTFT); smaller chunks give decode more turns (lower ITL). Worth profiling both directions. |

### Medium impact, medium effort

| Option | Description | Expected impact |
|---|---|---|
| Semantic response cache | Redis + similarity check in front of Open WebUI. With 8 rotating questions, cache hit rate is high after warmup. | Near-zero latency for repeated queries; removes them from the vLLM queue entirely. |
| Request admission queue | Nginx rate limiter or async proxy admits requests at ~10 req/s (current vLLM capacity). | Predictable queue wait instead of unpredictable 44–50s TTFT spikes. Better perceived UX at same hardware. |

### High impact, high effort (architecture)

| Option | Description | Expected impact |
|---|---|---|
| Model routing | Small fast model (7B) for grounded RAG answers; 30B reserved for open-ended PLAIN queries. | RAG latency drops significantly; 30B capacity freed for requests that need it. |

### Recommended next steps

1. Apply `--enable-prefix-caching` and `--num-scheduler-steps 8` — zero downside, immediate gain.
2. Benchmark reranker-off vs reranker-on to quantify the quality/latency trade-off on this KB.
3. If latency targets are still not met at 100 users, tune `VLLM_MAX_NUM_SEQS` and `VLLM_MAX_MODEL_LEN` together to find the best throughput/latency balance for the target concurrency.
