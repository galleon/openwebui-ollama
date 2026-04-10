# Benchmark Results — DGX Spark GB10

**Hardware:** NVIDIA DGX Spark, GB10 (128 GB unified memory, Grace CPU + Blackwell GPU)
**Stack:** Open WebUI + vLLM · embedder: `BAAI/bge-m3` · reranker: `BAAI/bge-reranker-v2-m3` · vector store: Qdrant

## Stack configuration

| Parameter | Value |
|---|---|
| `VLLM_NGC_TAG` | `26.03-py3` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.70 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
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

**Date:** 2026-04-08
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
**Config:** `VLLM_MAX_MODEL_LEN=8192`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_REASONING_PARSER=nano_v3`
**Raw CSVs:** `nemotron-nano/`

## Throughput (completions/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 0.072 ¹ | 0.042 | 0.062 |
| 20 | 0.149 | 0.062 | 0.080 |
| 30 | 0.206 | 0.079 | 0.114 |
| 50 | 0.220 | 0.077 | 0.159 |
| 100 | 0.258 | 0.069 | 0.160 |

¹ *First run after model load — TTFT inflated by warm-up. Throughput not significantly affected.*

## TTFT — time to first token (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 17.4 ¹ | 39.9 |
| 20 | 0.4 | 28.3 |
| 30 | 1.1 | 31.1 |
| 50 | 0.6 | 38.4 |
| 100 | 22.0 | 103.7 |

¹ *Warm-up run.*

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 50.1 | 85.8 |
| 20 | 70.2 | 70.9 |
| 30 | 92.3 | 62.8 |
| 50 | 21.2 | 78.8 |
| 100 | 65.3 | 104.3 |

## E2E latency (avg, seconds)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 122.3 | 98.6 |
| 20 | 103.0 | 104.3 |
| 30 | 81.5 | 109.4 |
| 50 | 95.1 | 145.5 |
| 100 | 147.7 | 207.4 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 182.0 | 160.9 |
| 20 | 259.9 | 193.3 |
| 30 | 240.0 | 184.4 |
| 50 | 164.4 | 197.6 |
| 100 | 202.4 | 226.7 |

## ITL — inter-token latency (avg, ms)

### Nothink

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 81.8 | 81.7 |
| 20 | 105.5 | 105.2 |
| 30 | 119.3 | 117.7 |
| 50 | 146.1 | 144.7 |
| 100 | 155.5 | 155.7 |

### Think

| Users | PLAIN | RAG |
|---|---|---|
| 10 | 95.1 | 81.0 |
| 20 | 103.1 | 102.3 |
| 30 | 115.2 | 113.3 |
| 50 | 141.3 | 142.3 |
| 100 | 153.1 | 153.3 |

## RAG overhead (avg TTFT above PLAIN baseline, seconds)

| Users | nothink | think |
|---|---|---|
| 10 | 19.6 ¹ | 73.5 |
| 20 | 28.0 | — ² |
| 30 | 29.7 | 11.3 ³ |
| 50 | 37.8 | 59.7 |
| 100 | 86.3 | 47.4 |

¹ *Warm-up run.*
² *Not recorded — insufficient mixed-type completions in this window.*
³ *Based on 2 completions only; treat as indicative.*

---

# Model: Qwen3.5-35B-A3B-FP8

**Date:** 2026-04-09
**Model:** `Qwen/Qwen3.5-35B-A3B-FP8`
**Config:** `VLLM_MAX_MODEL_LEN=32768`, `VLLM_REASONING_PARSER=qwen3`
**Raw CSVs:** `qwen3.5-35b-fp8/`

> **Think mode:** FP8 throughput on GB10 is too low for any think-mode request to complete within the 5-minute benchmark window — all think and mixed counts are zero. Bottleneck is tokens/second, not context length.
>
> **100-user nothink:** At 100 concurrent users even NT PLAIN requests queue longer than 300 s (zero completions). Saturation reached below u100.

## Throughput (completions/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 0.014 ¹ | — | — |
| 20 | 0.028 | — | — |
| 30 | 0.043 | — | — |
| 50 | 0.058 ² | — | — |
| 100 | — | — | — |

¹ *Warm-up run.*
² *Only NT PLAIN completions at u50; RAG requests exceed the 5-min window.*

## TTFT — time to first token (avg, seconds)

Nothink only — think mode has zero completions.

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 57.9 ¹ | 207.7 |
| 20 | 0.5 | 168.0 |
| 30 | 1.0 | 136.3 |
| 50 | 2.1 | — ² |
| 100 | — | — |

¹ *Warm-up run.*
² *No NT RAG completions at u50.*

## E2E latency (avg, seconds)

Nothink only.

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 148.0 | 275.6 |
| 20 | 107.6 | 247.6 |
| 30 | 141.9 | 186.4 |
| 50 | 87.6 | — |
| 100 | — | — |

## ITL — inter-token latency (avg, ms)

Nothink only.

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 107.5 | 112.6 |
| 20 | 152.2 | 147.3 |
| 30 | 178.9 | 175.2 |
| 50 | 210.9 | — |
| 100 | — | — |

## RAG overhead (avg TTFT above NT PLAIN baseline, seconds)

| Users | nothink |
|---|---|
| 10 | 149.8 ¹ |
| 20 | 167.5 |
| 30 | 135.4 |
| 50 | — |
| 100 | — |

¹ *Warm-up run — NT PLAIN TTFT inflated, so overhead understated.*

---

# Model: Qwen3.5-35B-A3B-NVFP4

**Date:** pending
**Model:** `AxionML/Qwen3.5-35B-A3B-NVFP4`
**Config:** `VLLM_MAX_MODEL_LEN=32768`, `VLLM_REASONING_PARSER=qwen3`, `VLLM_USE_FLASHINFER_MOE_FP4=1`
**Raw CSVs:** `qwen3.5-35b-nvfp4/` *(sweep pending)*

---

*Raw CSVs: `nemotron-nano/`, `qwen3.5-35b-fp8/`, `qwen3.5-35b-nvfp4/`*
