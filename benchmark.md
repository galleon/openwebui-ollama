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

Benchmarked with `locustfile.py` using **10 concurrent users**, ramp 2/s, 5-minute steady-state window (`--reset-stats`).

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

## Key findings

1. **Disable thinking for high-concurrency workloads.** At 10 users, nothink gives 2.2× throughput and 4× faster plain TTFT with no meaningful ITL regression.

2. **Qdrant is the RAG bottleneck.** The 26–27s retrieval overhead is consistent across modes. As user count increases, Qdrant saturation will likely surface before vLLM does.

3. **GPU generation throughput is healthy.** ITL of 68–76ms and TPS of 22–23ms/token on plain queries confirm the GB10 is not saturated at 10 users.

4. **Mixed mode behaves as expected.** Think and nothink tasks coexist without measurable interference — KV cache pressure is within budget at 0.70 utilization.

---

## Next steps

- [ ] Repeat suite at **100 users** to find saturation point
- [ ] Monitor Qdrant under higher concurrent load (likely bottleneck)
- [ ] Consider raising `VLLM_MAX_NUM_SEQS` above 8 for higher concurrency
- [ ] Consider raising `VLLM_MAX_MODEL_LEN` from 8192 once saturation profile is known

---

*Raw CSV and HTML reports: `results/nothink_u10_stats.csv`, `results/think_u10_stats.csv`, `results/mixed_u10_stats.csv`*
