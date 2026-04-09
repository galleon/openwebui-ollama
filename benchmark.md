# Benchmark Results

Results are organized per hardware target and model.
Raw CSVs live under `results/<hardware>/<model>/`.

| Hardware | Results |
|---|---|
| DGX Spark GB10 | this file |
| RTX Pro 6000 Blackwell | [`results/rtx-pro-6000/benchmark_summary.md`](results/rtx-pro-6000/benchmark_summary.md) |

---

# DGX Spark (GB10)

**Hardware:** NVIDIA DGX Spark, GB10 (128 GB unified memory)

## Stack configuration

| Parameter | Value |
|---|---|
| `VLLM_NGC_TAG` | `26.03-py3` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.70 |
| `VLLM_MAX_MODEL_LEN` | 8192 |
| `VLLM_KV_CACHE_DTYPE` | fp8 |
| `VLLM_MAX_NUM_SEQS` | 64 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 1 |
| `--enable-chunked-prefill` | yes |
| Embedder | `BAAI/bge-m3` on GPU |
| Reranker | `BAAI/bge-reranker-v2-m3` on GPU |
| Vector store | Qdrant |

## Methodology

Benchmarked with `locustfile.py`, 5-minute steady-state window (`--reset-stats`) per run.
User levels tested: **10 / 20 / 30 / 50 / 100** users.

Three tag-filtered runs per user level, in order:

| Run | Tag | Tasks |
|---|---|---|
| 1 | `nothink` | NT PLAIN (weight 1) + NT RAG (weight 3) |
| 2 | `think` | PLAIN (weight 1) + RAG (weight 3) |
| 3 | `mixed` | All four tasks (NT+think, 1:3:1:3 ratio) |

Thinking is disabled per-request via `chat_template_kwargs: {"enable_thinking": false}`.
RAG queries attach the Nürburgring knowledge base (8 domain-specific questions, hybrid search top-k=5 → reranked to 3).

---

# Model: Nemotron-3-Nano-30B-A3B-NVFP4

**Date:** 2026-04-08
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
**Raw CSVs:** `results/dgx-spark-gb10/nemotron-nano/`

## Throughput (req/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 1.035 | 0.423 | 0.712 |
| 20 | 1.767 | 0.647 | 1.274 |
| 30 | 2.109 | 0.915 | 1.397 |
| 50 | 2.720 | 1.175 | 1.696 |
| 100 | 2.926 | 1.163 | 1.854 |

## TTFT — time to first token (avg, seconds)

| Users | NT PLAIN | NT RAG | PLAIN (think) | RAG (think) |
|---|---|---|---|---|
| 10 | 1.41 | 31.20 | 49.74 | 86.14 |
| 20 | 0.47 | 37.69 | 35.77 | 99.89 |
| 30 | 0.63 | 40.89 | 53.66 | 110.08 |
| 50 | 0.82 | 54.81 | 43.76 | 138.02 |
| 100 | 16.13 | 121.99 | 100.68 | 224.75 |

> **Note:** NT RAG and RAG (think) TTFT includes full RAG retrieval (embed → hybrid search → rerank) before the first token. PLAIN (think) TTFT captures the first `<think>` token; answer tokens follow after the reasoning chain.

## E2E latency (avg, seconds)

| Users | NT PLAIN | NT RAG | PLAIN (think) | RAG (think) |
|---|---|---|---|---|
| 10 | 64.0 | 40.7 | 174.0 | 96.0 |
| 20 | 101.1 | 51.9 | 231.8 | 113.6 |
| 30 | 119.5 | 57.6 | 161.4 | 124.7 |
| 50 | 132.0 | 76.9 | 193.8 | 158.2 |
| 100 | 168.3 | 146.9 | 180.9 | 245.1 |

## ITL — inter-token latency (avg, ms)

| Users | nothink | think |
|---|---|---|
| 10 | 90.9 | 95.8 |
| 20 | 122.2 | 110.5 |
| 30 | 149.2 | 139.7 |
| 50 | 197.1 | 181.8 |
| 100 | 234.1 | 209.9 |

## RAG overhead (avg TTFT above PLAIN baseline, seconds)

| Users | nothink | think |
|---|---|---|
| 10 | 31.91 | 44.29 |
| 20 | 37.21 | 82.11 |
| 30 | 41.27 | 73.58 |
| 50 | 54.06 | 110.56 |
| 100 | 101.14 | 115.35 |

---

# Model: Qwen3.5-35B-A3B-FP8

**Date:** 2026-04-08
**Model:** `Qwen/Qwen3.5-35B-A3B-FP8`
**Raw CSVs:** `results/dgx-spark-gb10/qwen3.5-35b-fp8/`
**Reasoning parser:** `deepseek_r1` (built-in — no plugin required)

> **Think mode limitation:** With `VLLM_MAX_MODEL_LEN=8192`, Qwen3.5 in think mode generates reasoning chains long enough that most requests exceed the 5-minute benchmark window. At ≤30 users, zero think-mode requests complete; at 50–100 users only a handful complete. Think and mixed columns are marked `—` where throughput is effectively zero (< 0.06 req/s aggregated).

## Throughput (req/s)

| Users | nothink | think | mixed |
|---|---|---|---|
| 10 | 0.226 | — | — |
| 20 | 0.343 | — | — |
| 30 | 0.273 | — | 0.124 |
| 50 | 0.341 | 0.053 | 0.072 |
| 100 | 0.206 | — | 0.063 |

## TTFT — time to first token (avg, seconds)

Nothink only — think mode has too few completions for stable averages.

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 13.78 ¹ | 202.06 ¹ |
| 20 | 0.66 | 181.58 |
| 30 | 1.02 | 141.46 |
| 50 | 28.11 | 218.27 |
| 100 | 24.55 | — |

¹ *u10 results collected during model warm-up (first run after load); may overstate latency.*

## E2E latency (avg, seconds)

Nothink only.

| Users | NT PLAIN | NT RAG |
|---|---|---|
| 10 | 73.6 | 227.5 |
| 20 | 94.3 | 218.1 |
| 30 | 98.6 | 171.9 |
| 50 | 154.0 | 251.2 |
| 100 | 164.1 | — |

## ITL — inter-token latency (avg, ms)

Nothink only (think mode: insufficient completions).

| Users | nothink |
|---|---|
| 10 | 107.8 |
| 20 | 167.6 |
| 30 | 203.1 |
| 50 | 218.8 |
| 100 | 225.8 |

## RAG overhead (avg TTFT above PLAIN baseline, seconds)

Nothink only.

| Users | nothink |
|---|---|
| 10 | 201.63 ¹ |
| 20 | 213.69 |
| 30 | 140.37 |
| 50 | 163.60 |
| 100 | — |

---

# Cross-model comparison (DGX GB10)

## Throughput — nothink (req/s)

| Users | Nemotron-Nano | Qwen3.5-35B |
|---|---|---|
| 10 | 1.035 | 0.226 |
| 20 | 1.767 | 0.343 |
| 30 | 2.109 | 0.273 |
| 50 | 2.720 | 0.341 |
| 100 | 2.926 | 0.206 |

## NT PLAIN TTFT (avg, seconds)

| Users | Nemotron-Nano | Qwen3.5-35B |
|---|---|---|
| 10 | 1.41 | 13.78 ¹ |
| 20 | 0.47 | 0.66 |
| 30 | 0.63 | 1.02 |
| 50 | 0.82 | 28.11 |
| 100 | 16.13 | 24.55 |

## ITL nothink (avg, ms)

| Users | Nemotron-Nano | Qwen3.5-35B |
|---|---|---|
| 10 | 90.9 | 107.8 |
| 20 | 122.2 | 167.6 |
| 30 | 149.2 | 203.1 |
| 50 | 197.1 | 218.8 |
| 100 | 234.1 | 225.8 |

---

## Summary

**Nemotron-3-Nano-30B-A3B-NVFP4** saturates at ~2.9 req/s (nothink) around 50–100 users. NT PLAIN TTFT stays sub-second up to 50 users, then spikes at 100 (queue saturation). Think mode runs cleanly but at roughly half the throughput of nothink (longer generation). RAG overhead dominates TTFT in all modes (31–120 s depending on load), driven by hybrid search + reranking.

**Qwen3.5-35B-A3B-FP8** delivers 4–9× lower nothink throughput than Nemotron on this stack. NT PLAIN TTFT is competitive at light loads (u20–30) but degrades at higher concurrency. Think mode is not viable at `MAX_MODEL_LEN=8192` — reasoning chains overflow the benchmark window. RAG overhead for nothink is very high (140–214 s) due to the larger base model latency compounding retrieval cost. Increasing `VLLM_MAX_MODEL_LEN` or disabling thinking would be required for a fair think-mode comparison.

---

*Raw CSV and HTML reports: `results/dgx-spark-gb10/{nemotron-nano,qwen3.5-35b-fp8}/`*
*Naming convention: `{mode}_u{N}_stats.csv`*
