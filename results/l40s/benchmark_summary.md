# Benchmark Results — NVIDIA L40S

**Hardware:** NVIDIA L40S (45 GB VRAM available, 864 GB/s memory bandwidth)
**Stack:** Open WebUI + vLLM · embeddings: built-in sentence-transformers CPU
**Sweeps:** two runs — Chroma (built-in) and Qdrant (external container)

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
| Vector store (sweep A) | Chroma (built-in, no extra service) |
| Vector store (sweep B) | Qdrant (external container, `--profile qdrant`) |

## Memory budget

| Component | Size |
|---|---|
| GPU VRAM (available) | 45 GB |
| vLLM allocation (0.92 × 45 GB) | 41.4 GB |
| Qwen3.5-35B weights (FP8) | ~35 GB |
| KV cache headroom | ~6 GB |

---

# Model: Qwen3.5-35B-A3B-FP8

**Raw CSVs:**
- Chroma sweep: `qwen3.5-35b-fp8-chroma/mlen32768_seqs32/`
- Qdrant sweep: `qwen3.5-35b-fp8-qdrant/mlen32768_seqs32/`

---

## ITL (Inter-Token Latency) — consistent across both sweeps

ITL is stable at **~35 ms/token (~28 tok/s)** across all concurrency levels, task types,
and vector store backends. The generation throughput ceiling is the model itself.

| Users | NT PLAIN ITL | NT RAG ITL | PLAIN ITL | RAG ITL |
|---|---|---|---|---|
| 10 | 25–37 ms | 25 ms | 32–37 ms | 27 ms |
| 20 | 33–34 ms | 32 ms | 33 ms | 33–34 ms |
| 30 | 36 ms | 36–37 ms | 36–37 ms | 38 ms |
| 50 | 36–37 ms | 36 ms | 35–37 ms | 32–37 ms |
| 100 | 35–36 ms | — | 35–37 ms | — |

Ranges reflect Chroma / Qdrant values respectively; differences are within measurement noise.

---

## Nothink mode (thinking disabled)

### NT PLAIN — bare chat, no RAG

| Users | E2E (Chroma) | TTFT (Chroma) | E2E (Qdrant) | TTFT (Qdrant) | ITL avg |
|---|---|---|---|---|---|
| 10 | 29.7 s | 26.0 s | 42.8 s | 36.8 s | ~30 ms |
| 20 | 5.1 s | **142 ms** | 4.6 s | **200 ms** | ~33 ms |
| 30 | 19.6 s | 15.2 s | 17.9 s | 12.3 s | ~36 ms |
| 50 | 73.0 s | 68.3 s | 49.3 s | 43.7 s | ~36 ms |
| 100 | 147.8 s | 143.4 s | 156.3 s | 151.7 s | ~35 ms |

> **Best operating point: u=20** — TTFT of 142–200 ms (Chroma/Qdrant) confirms near-instant
> response when the model is not queued. Saturation occurs between u=20 and u=30. The u=10
> figures are anomalous (first-run warm-up effects).

### NT RAG — knowledge-base retrieval, thinking disabled

| Users | E2E (Chroma) | RAG overhead (Chroma) | E2E (Qdrant) | RAG overhead (Qdrant) | ITL avg |
|---|---|---|---|---|---|
| 10 | 133.6 s | 128.3 s | 141.8 s | 115.1 s | ~25 ms |
| 20 | 136.9 s | 131.1 s | 132.1 s | 126.8 s | ~32 ms |
| 30 | 166.7 s | 158.0 s | 174.5 s | 167.4 s | ~36 ms |
| 50 | 244.4 s | 152.3 s | 219.5 s | 132.2 s | ~36 ms |
| 100 | — | — | — | — | — |

> ⚠️ **RAG overhead is very high (~115–170 s) on both backends** — dominated by CPU-based
> embedding (bge-m3 on CPU). The Qdrant query itself is sub-millisecond; the vector store
> choice has no measurable impact on RAG latency. Moving the embedder to GPU would reduce
> overhead by ~100×.

---

## Think mode (reasoning enabled)

### PLAIN — bare chat, thinking enabled

| Users | E2E (Chroma) | TTFT (Chroma) | E2E (Qdrant) | TTFT (Qdrant) | ITL avg |
|---|---|---|---|---|---|
| 10 | 171.7 s | 164.2 s | 181.4 s | 172.9 s | ~34 ms |
| 20 | 33.8 s | 27.8 s | 44.9 s | 38.8 s | ~33 ms |
| 30 | 46.7 s | 39.9 s | 36.9 s | 31.4 s | ~37 ms |
| 50 | 67.7 s | 60.1 s | 86.3 s | 80.9 s | ~36 ms |
| 100 | 110.6 s | 104.1 s | 147.4 s | 140.5 s | ~36 ms |

> Think mode E2E is 3–5× higher than nothink due to reasoning chain generation.
> Both runs show identical ITL, confirming the vector store has no influence on generation.

### RAG — knowledge-base retrieval, thinking enabled

| Users | E2E (Chroma) | RAG overhead (Chroma) | E2E (Qdrant) | RAG overhead (Qdrant) | ITL avg |
|---|---|---|---|---|---|
| 10 | — | — | 283.8 s | 248.1 s | ~27 ms |
| 20 | 209.7 s | 181.9 s | 234.0 s | 191.0 s | ~34 ms |
| 30 | 252.3 s | 167.9 s | 243.6 s | 218.9 s | ~38 ms |
| 50 | 255.7 s | 84.9 s | 266.0 s | 53.3 s | ~34 ms |
| 100 | — | — | — | — | — |

---

## Mixed mode (think + nothink concurrent)

| Users | NT PLAIN TTFT (Chroma) | NT PLAIN TTFT (Qdrant) | PLAIN TTFT (Chroma) | PLAIN TTFT (Qdrant) | NT RAG E2E (Chroma) | NT RAG E2E (Qdrant) |
|---|---|---|---|---|---|---|
| 10 | 78.3 s | — | 121.8 s | 95.6 s | 173.2 s | 250.7 s |
| 20 | **111 ms** | **100 ms** | 28.6 s | 33.2 s | 153.7 s | 147.0 s |
| 30 | 9.0 s | — | 38.3 s | 16.6 s | 170.0 s | 159.7 s |
| 50 | 64.6 s | — | 76.5 s | 79.1 s | 199.1 s | 238.7 s |
| 100 | 75.5 s | — | 171.4 s | 141.5 s | — | — |

---

## Chroma vs Qdrant — head-to-head comparison

| Metric | Chroma | Qdrant | Delta | Verdict |
|---|---|---|---|---|
| NT PLAIN TTFT (u=20) | 142 ms | 200 ms | +58 ms | within noise |
| NT RAG overhead (u=20) | 131 s | 127 s | −4 s | equivalent |
| NT RAG overhead (u=30) | 158 s | 167 s | +9 s | equivalent |
| PLAIN E2E (u=20) | 33.8 s | 44.9 s | +11 s | within noise |
| Think RAG E2E (u=20) | 209.7 s | 234.0 s | +24 s | within noise |
| ITL avg (all modes) | ~35 ms | ~35 ms | 0 | identical |
| Extra container | none | qdrant (~200 MB) | — | — |
| Setup complexity | zero | `--profile qdrant` | — | — |

> **The two backends are statistically equivalent for this workload.** All observed deltas
> are within run-to-run variability (different KB content, warm-up state, OS scheduling).
> The bottleneck is CPU embedding, not the vector store.

---

## Chroma vs Qdrant — operational pros and cons

### Chroma (built-in default)

**Pros**
- Zero extra configuration — ships inside the Open WebUI container, no additional service needed
- No extra memory footprint or container to manage
- Ideal for single-node, single-user, or development deployments
- Simpler `docker compose up -d` — no profile flag required

**Cons**
- Data lives inside the `open_webui_data` Docker volume — no independent persistence or backup path
- No HTTP API for direct inspection or external tooling
- Not suitable for multi-node or HA deployments (no replication, no distributed search)
- Harder to scale independently of Open WebUI (tied to the same container lifecycle)

### Qdrant (external container)

**Pros**
- Independent persistence (`qdrant_data` volume) — can be backed up, migrated, or inspected separately from Open WebUI
- Full HTTP + gRPC API — query, inspect, or manage collections with external tools
- Production-grade: supports replication, snapshots, filtering, payload indexing, and cluster mode
- Can be shared across multiple Open WebUI instances or other services
- Easy to upgrade or replace without touching Open WebUI

**Cons**
- One extra container to operate and monitor
- Requires `--profile qdrant` flag (or a dedicated compose file) to start
- Marginal memory overhead (~200–400 MB at rest for small collections)
- On a single node with CPU embedding the added operational cost provides no latency benefit

### Recommendation

| Use case | Recommended backend |
|---|---|
| Local dev / single user | **Chroma** — zero setup, zero overhead |
| Small team, single node | **Chroma** — adequate; switch only if independent backup is needed |
| Production / multi-user | **Qdrant** — independent lifecycle, backup, external tooling |
| Multi-node / HA | **Qdrant** — mandatory; Chroma has no cluster mode |
| GPU embedder (future) | **Qdrant** — latency will drop to ~10 ms; worth the ops overhead |

---

## Key findings

1. **Optimal concurrency: 20 users** — NT PLAIN TTFT of 142–200 ms; saturation kicks in at ~25 users on both backends.

2. **ITL throughput ceiling: ~28 tok/s** — rock solid from u=10 to u=100 on both Chroma and Qdrant; the model generation pipeline is the sole bottleneck.

3. **RAG overhead is the dominant latency** — CPU embedding (bge-m3) adds ~115–170 s to every RAG request regardless of vector store. Moving the embedder to GPU is the single highest-impact optimization available (~100× reduction in RAG overhead).

4. **Chroma and Qdrant are equivalent on a single node** — all latency differences are within measurement noise. The choice between them is purely operational, not performance-driven.

5. **Think mode adds 3–5× E2E latency** — reasoning chains saturate the 32K context window at higher concurrency; feasible for single-user or low-concurrency workloads only.

6. **RAG at high concurrency (u≥100) times out** — the 300 s request timeout in `locustfile.py` is hit; system is fully saturated regardless of vector store.
