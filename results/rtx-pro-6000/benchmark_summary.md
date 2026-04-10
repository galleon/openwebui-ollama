# Benchmark Results — RTX Pro 6000 Blackwell

**Hardware:** NVIDIA RTX Pro 6000 Blackwell Server Edition (97 GB GDDR7)
**Date:** 2026-04-10
**Stack:** Open WebUI + vLLM · embedder: `BAAI/bge-m3` · reranker: `BAAI/bge-reranker-v2-m3` · vector store: Qdrant

## Stack configuration

| Parameter | Value |
|---|---|
| `VLLM_NGC_TAG` | `26.03-py3` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.55 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
| `VLLM_MAX_MODEL_LEN` | 16384 |
| `VLLM_MAX_NUM_SEQS` | 64 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 |
| Embedder | `BAAI/bge-m3` on GPU |
| Reranker | `BAAI/bge-reranker-v2-m3` on GPU |
| Vector store | Qdrant |

## Methodology

Benchmarked with `locustfile.py`, 5-minute steady-state window (`--reset-stats`) per run.
User levels tested: **10 / 20 / 30 / 50 / 100** users.

Three tag-filtered runs per user level:

| Run | Tag | Tasks |
|---|---|---|
| 1 | `nothink` | NT PLAIN (weight 1) + NT RAG (weight 3) |
| 2 | `think` | PLAIN (weight 1) + RAG (weight 3) |
| 3 | `mixed` | All four tasks (NT+think, 1:3:1:3 ratio) |

RAG queries attach the Nürburgring knowledge base (8 domain-specific questions, hybrid search top-k=5 → reranked to 3).

---

# Model: Nemotron-3-Nano-30B-A3B-NVFP4

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
**Config:** `VLLM_MAX_MODEL_LEN=16384`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_REASONING_PARSER=nano_v3`
**Raw CSVs:** `nemotron-nano/`

## Throughput (completions/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 1.233 | 0.651 | 0.817 |
| 20 | 1.963 | 1.047 | 1.387 |
| 30 | 2.575 | 1.275 | 1.690 |
| 50 | 3.403 | 1.723 | 2.228 |
| 100 | 4.249 | 2.157 | 2.946 |

## TTFT — time to first token (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 0.06 | 3.84 |
| 20 | 0.08 | 4.90 |
| 30 | 0.11 | 5.72 |
| 50 | 0.15 | 7.43 |
| 100 | 2.34 | 14.60 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 5.12 | 9.62 |
| 20 | 5.33 | 12.49 |
| 30 | 6.42 | 15.64 |
| 50 | 9.57 | 18.84 |
| 100 | 16.43 | 35.32 |

## E2E latency (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 9.66 | 4.93 |
| 20 | 13.27 | 6.49 |
| 30 | 16.06 | 7.57 |
| 50 | 20.13 | 9.83 |
| 100 | 30.63 | 17.79 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 20.10 | 10.82 |
| 20 | 25.48 | 14.10 |
| 30 | 30.84 | 17.67 |
| 50 | 38.65 | 21.40 |
| 100 | 50.35 | 38.15 |

## ITL — inter-token latency (avg, ms)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 9.4 | 9.3 |
| 20 | 13.1 | 12.9 |
| 30 | 16.0 | 16.0 |
| 50 | 20.3 | 20.4 |
| 100 | 26.5 | 28.9 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 9.5 | 9.6 |
| 20 | 12.9 | 12.9 |
| 30 | 15.4 | 15.5 |
| 50 | 19.4 | 19.4 |
| 100 | 22.4 | 22.2 |

## RAG overhead (avg TTFT above PLAIN baseline, seconds)

| Users | nothink | think |
|---|---|---|
| 10 | 3.79 | 5.05 |
| 20 | 4.80 | 7.46 |
| 30 | 5.62 | 10.39 |
| 50 | 7.27 | 11.18 |
| 100 | 12.59 | 18.71 |

---

# Model: Qwen3.5-35B-A3B-NVFP4

**Model:** `AxionML/Qwen3.5-35B-A3B-NVFP4`
**Config:** `VLLM_MAX_MODEL_LEN=16384`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_REASONING_PARSER=qwen3`
**Raw CSVs:** `qwen3.5-35b-nvfp4/`

## Throughput (completions/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 0.282 | 0.082 | 0.139 |
| 20 | 0.398 | 0.156 | 0.193 |
| 30 | 0.575 | 0.179 | 0.281 |
| 50 | 0.679 | 0.255 | 0.424 |
| 100 | 0.906 | 0.260 | 0.393 |

## TTFT — time to first token (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 0.43 | 36.64 |
| 20 | 0.07 | 49.91 |
| 30 | 0.09 | 55.63 |
| 50 | 0.74 | 68.08 |
| 100 | 25.72 | 111.34 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 24.82 | 120.71 |
| 20 | 53.21 | 116.88 |
| 30 | 36.44 | 138.02 |
| 50 | 69.61 | 161.45 |
| 100 | 88.86 | 219.28 |

## E2E latency (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 8.17 | 38.87 |
| 20 | 10.26 | 53.00 |
| 30 | 11.05 | 59.11 |
| 50 | 14.92 | 72.39 |
| 100 | 40.84 | 115.99 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 32.34 | 122.77 |
| 20 | 63.65 | 119.58 |
| 30 | 47.82 | 140.56 |
| 50 | 85.93 | 164.89 |
| 100 | 106.95 | 222.78 |

## ITL — inter-token latency (avg, ms)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 13.0 | 13.1 |
| 20 | 17.3 | 17.2 |
| 30 | 19.8 | 20.0 |
| 50 | 24.1 | 24.1 |
| 100 | 26.3 | 26.7 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 13.6 | 13.7 |
| 20 | 16.9 | 17.0 |
| 30 | 25.3 | 17.7 |
| 50 | 22.7 | 22.4 |
| 100 | 24.8 | 23.8 |

## RAG overhead (avg TTFT above PLAIN baseline, seconds)

| Users | nothink | think |
|---|---|---|
| 10 | 35.21 | 72.51 |
| 20 | 49.85 | 55.81 |
| 30 | 55.56 | 89.86 |
| 50 | 67.72 | 66.90 |
| 100 | 81.87 | 79.33 |

---

# Cross-model comparison (RTX Pro 6000)

## Throughput — nothink (completions/s)

| Users | Nemotron-Nano NVFP4 | Qwen3.5-35B NVFP4 |
|---|---|---|
| 10 | 1.233 | 0.282 |
| 20 | 1.963 | 0.398 |
| 30 | 2.575 | 0.575 |
| 50 | 3.403 | 0.679 |
| 100 | 4.249 | 0.906 |

## Throughput — think (completions/s)

| Users | Nemotron-Nano NVFP4 | Qwen3.5-35B NVFP4 |
|---|---|---|
| 10 | 0.651 | 0.082 |
| 20 | 1.047 | 0.156 |
| 30 | 1.275 | 0.179 |
| 50 | 1.723 | 0.255 |
| 100 | 2.157 | 0.260 |

## NT PLAIN TTFT (avg, seconds)

| Users | Nemotron-Nano NVFP4 | Qwen3.5-35B NVFP4 |
|---|---|---|
| 10 | 0.06 | 0.43 |
| 20 | 0.08 | 0.07 |
| 30 | 0.11 | 0.09 |
| 50 | 0.15 | 0.74 |
| 100 | 2.34 | 25.72 |

## ITL nothink — NT PLAIN (avg, ms)

| Users | Nemotron-Nano NVFP4 | Qwen3.5-35B NVFP4 |
|---|---|---|
| 10 | 9.4 | 13.0 |
| 20 | 13.1 | 17.3 |
| 30 | 16.0 | 19.8 |
| 50 | 20.3 | 24.1 |
| 100 | 26.5 | 26.3 |

---

## Summary

**Nemotron-3-Nano-30B-A3B-NVFP4** on the RTX Pro 6000 delivers exceptional throughput: 4.25 completions/s nothink at 100 users, with NT PLAIN TTFT staying sub-second up to 50 users. ITL scales smoothly from 9 ms (u10) to 26 ms (u100) — no stalls. Think mode runs at ~50% of nothink throughput and scales cleanly through 100 users. RAG overhead is 3.8–12.6 s (nothink) and 5–18.7 s (think), well within UX-acceptable range at all tested concurrency levels.

**Qwen3.5-35B-A3B-NVFP4** delivers ~4–8× lower nothink throughput than Nemotron-Nano (0.28–0.91 vs 1.23–4.25 completions/s). Unlike the FP8 variant on GB10, think mode completes at all user levels — NVFP4 MoE kernels bring it within the 5-minute window even at 100 users (0.26 completions/s). NT PLAIN TTFT is competitive at light loads (sub-second up to u30) but spikes at u100 (25.7 s) where queuing dominates. RAG overhead in nothink mode is 35–82 s, driven by long PLAIN outputs queuing ahead of RAG requests. The ITL floor (~13–26 ms) is comparable to Nemotron at equivalent load, confirming effective NVFP4 quantization at the per-token level.

*Raw CSVs: `nemotron-nano/` and `qwen3.5-35b-nvfp4/`*
