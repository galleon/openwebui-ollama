# Benchmark Results — L40S

**Hardware:** NVIDIA L40S (48 GB GDDR6, 864 GB/s memory bandwidth)
**Stack:** Open WebUI + vLLM · embedder: `BAAI/bge-m3` · reranker: `BAAI/bge-reranker-v2-m3` · vector store: Qdrant

## Stack configuration

| Parameter | Value |
|---|---|
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
| `VLLM_MAX_MODEL_LEN` | 8192 (pilot) |
| `VLLM_MAX_NUM_SEQS` | 8 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 |

## Memory budget

| Component | Size |
|---|---|
| GPU VRAM | 48 GB |
| vLLM allocation (0.90) | 43.2 GB |
| Qwen3.5-35B weights (FP8) | ~35 GB |
| KV cache headroom | ~8 GB |

---

# Model: Qwen3.5-35B-A3B-FP8

**Model:** `Qwen/Qwen3.5-35B-A3B-FP8`
**Raw CSVs:** `qwen3.5-35b-fp8/`

> **Full sweep pending.** Only pilot data collected to date (10 users, nothink NT RAG, mlen=8192, 5 completions).

## Pilot results (10 users, nothink NT RAG, mlen=8192)

| Metric | Value |
|---|---|
| Throughput | 0.086 req/s |
| NT RAG E2E latency (avg) | 40.6 s |
| NT RAG TTFT (avg) | 38.1 s |
| ITL avg | 22 ms (~45 tok/s) |

The ~38 s TTFT is almost entirely RAG retrieval overhead (embed → hybrid search → rerank), not model latency. NT PLAIN TTFT is expected to be sub-second at light load.

*For sweep setup and vLLM configuration guidance, see [`SWEEP_GUIDE.md`](SWEEP_GUIDE.md).*
