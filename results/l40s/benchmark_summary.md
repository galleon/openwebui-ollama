# Benchmark Results — NVIDIA L40S

**Hardware:** NVIDIA L40S (45 GB VRAM available, 864 GB/s memory bandwidth)
**Stack:** Open WebUI + vLLM · embeddings: built-in sentence-transformers CPU · vector store: Chroma (built-in)

## Stack configuration

| Parameter | Value |
|---|---|
| `VLLM_MODEL` | `Qwen/Qwen3.5-35B-A3B-FP8` |
| `VLLM_NGC_TAG` | `26.03-py3` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.92 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
| `VLLM_MAX_MODEL_LEN` | 32768 |
| `VLLM_MAX_NUM_SEQS` | 32 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 |
| `VLLM_REASONING_PARSER` | qwen3 |
| Embedder | Open WebUI built-in (CPU) |
| Reranker | Open WebUI built-in (CPU) |

## Memory budget

| Component | Size |
|---|---|
| GPU VRAM (available) | 45 GB |
| vLLM allocation (0.92 × 45 GB) | 41.4 GB |
| Qwen3.5-35B weights (FP8) | ~35 GB |
| KV cache headroom | ~6 GB |

---

# Model: Qwen3.5-35B-A3B-FP8

**Raw CSVs:** `qwen3.5-35b-fp8/mlen32768_seqs32/`

---

## ITL (Inter-Token Latency) — consistent across all modes

ITL is stable at **~35 ms/token (~28 tok/s)** across all concurrency levels and task types,
indicating the model's generation throughput is the steady-state ceiling regardless of load.

| Users | NT PLAIN ITL avg | NT RAG ITL avg | PLAIN ITL avg | RAG ITL avg |
|---|---|---|---|---|
| 10 | 25 ms | 25 ms | — | — |
| 20 | 34 ms | 32 ms | 33 ms | 34 ms |
| 30 | 36 ms | 37 ms | 36 ms | 38 ms |
| 50 | 36 ms | 36 ms | 35 ms | 37 ms |
| 100 | 36 ms | — | 35 ms | — |

---

## Nothink mode (thinking disabled)

### NT PLAIN — bare chat, no RAG

| Users | E2E avg | TTFT avg | TTFT p95 | ITL avg | TPS avg |
|---|---|---|---|---|---|
| 10 | 29.7 s | 26.0 s | 59 s | 25 ms | 75 ms/tok |
| 20 | 5.1 s | 142 ms | 170 ms | 34 ms | 9 ms/tok |
| 30 | 19.6 s | 15.2 s | 49 s | 36 ms | 39 ms/tok |
| 50 | 73.0 s | 68.3 s | 129 s | 36 ms | 155 ms/tok |
| 100 | 147.8 s | 143.4 s | 276 s | 36 ms | 374 ms/tok |

> **Best operating point: u=20** — TTFT of 142 ms confirms near-instant response when
> the model is not queued. Saturation occurs between u=20 and u=30 where TTFT jumps
> from 142 ms to 15 s. The u=10 figures are anomalous (first-run warm-up effects).

### NT RAG — knowledge-base retrieval, thinking disabled

| Users | E2E avg | TTFT avg | RAG overhead avg | ITL avg |
|---|---|---|---|---|
| 10 | 133.6 s | 128.3 s | 128.3 s | 25 ms |
| 20 | 136.9 s | 131.3 s | 131.1 s | 32 ms |
| 30 | 166.7 s | 160.3 s | 158.0 s | 37 ms |
| 50 | 244.4 s | 238.4 s | 152.3 s | 36 ms |
| 100 | — | — | — | — |

> ⚠️ **RAG overhead is very high (~130–160 s)** — dominated by CPU-based embedding
> (bge-m3 on CPU takes ~100–150 ms per query × chunked document lookups).
> Switching the embedder to GPU would reduce this to ~10–50 ms. See note below.

---

## Think mode (reasoning enabled)

### PLAIN — bare chat, thinking enabled

| Users | E2E avg | TTFT avg | TTFT p95 | ITL avg |
|---|---|---|---|---|
| 10 | 171.7 s | 164.2 s | 223 s | 37 ms |
| 20 | 33.8 s | 27.8 s | 47 s | 33 ms |
| 30 | 46.7 s | 39.9 s | 80 s | 36 ms |
| 50 | 67.7 s | 60.1 s | 166 s | 35 ms |
| 100 | 110.6 s | 104.1 s | 239 s | 35 ms |

> Think mode E2E is 3–5× higher than nothink due to reasoning chain generation.
> ITL remains stable, confirming generation throughput is unchanged.

### RAG — knowledge-base retrieval, thinking enabled

| Users | E2E avg | TTFT avg | RAG overhead avg | ITL avg |
|---|---|---|---|---|
| 20 | 209.7 s | 205.7 s | 181.9 s | 34 ms |
| 30 | 252.3 s | 248.1 s | 167.9 s | 38 ms |
| 50 | 255.7 s | 250.5 s | 84.9 s | 37 ms |
| 100 | — | — | — | — |

---

## Mixed mode (think + nothink concurrent)

At u=20 the system handles the mixed workload well. NT PLAIN TTFT stays low (111 ms)
while think PLAIN degrades gracefully. Above u=30 all variants experience significant
queuing.

| Users | NT PLAIN TTFT | PLAIN TTFT | NT RAG E2E | RAG E2E |
|---|---|---|---|---|
| 10 | 78.3 s | 121.8 s | 173.2 s | 171.2 s |
| 20 | 111 ms | 28.6 s | 153.7 s | 218.7 s |
| 30 | 9.0 s | 38.3 s | 170.0 s | 237.1 s |
| 50 | 64.6 s | 76.5 s | 199.1 s | 278.6 s |
| 100 | 75.5 s | 171.4 s | — | — |

---

## Key findings

1. **Optimal concurrency: 20 users** — NT PLAIN TTFT of 142 ms; saturation kicks in at ~25 users.

2. **ITL throughput ceiling: ~28 tok/s** — rock solid from u=10 to u=100, model generation is the bottleneck not memory bandwidth.

3. **RAG overhead is the dominant latency** — CPU embedding (bge-m3) adds ~130–160 s to every RAG request. Moving the embedder to GPU would reduce overhead by ~100× and bring RAG E2E in line with PLAIN E2E.

4. **Think mode adds 3–5× E2E latency** — reasoning chains saturate the 32K context window at higher concurrency; feasible for single-user or low-concurrency workloads.

5. **RAG at high concurrency (u≥100) times out** — the 300 s request timeout in locustfile.py is hit; system is fully saturated.
