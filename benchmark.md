# Benchmark Results

Cross-hardware comparison for the Open WebUI + vLLM stack.
Per-hardware detail lives in each hardware's summary file.

| Hardware | Summary |
|---|---|
| DGX Spark GB10 | [`results/dgx-spark-gb10/benchmark_summary.md`](results/dgx-spark-gb10/benchmark_summary.md) |
| RTX Pro 6000 Blackwell | [`results/rtx-pro-6000/benchmark_summary.md`](results/rtx-pro-6000/benchmark_summary.md) |
| L40S | [`results/l40s/benchmark_summary.md`](results/l40s/benchmark_summary.md) |

---

## Hardware overview

| Hardware | VRAM | Memory type | GPU memory util | Max model len |
|---|---|---|---|---|
| DGX Spark GB10 | 128 GB (unified) | LPDDR5X | 0.70 | 8192 (Nemotron) / 32768 (Qwen FP8) |
| RTX Pro 6000 Blackwell | 97 GB | GDDR7 | 0.55 | 16384 |
| L40S | 48 GB | GDDR6 | 0.90 | 8192 (pilot) |

---

## Model: Nemotron-3-Nano-30B-A3B-NVFP4

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` · `VLLM_USE_FLASHINFER_MOE_FP4=1` · `VLLM_REASONING_PARSER=nano_v3`

### Throughput — nothink (completions/s)

| Users | DGX GB10 | RTX Pro 6000 |
|---|---|---|
| 10 | 0.072 ¹ | 1.233 |
| 20 | 0.149 | 1.963 |
| 30 | 0.206 | 2.575 |
| 50 | 0.220 | 3.403 |
| 100 | 0.258 | 4.249 |

¹ *Warm-up run on GB10.*

### Throughput — think (completions/s)

| Users | DGX GB10 | RTX Pro 6000 |
|---|---|---|
| 10 | 0.042 | 0.651 |
| 20 | 0.062 | 1.047 |
| 30 | 0.079 | 1.275 |
| 50 | 0.077 | 1.723 |
| 100 | 0.069 | 2.157 |

### NT PLAIN TTFT (avg, seconds)

| Users | DGX GB10 | RTX Pro 6000 |
|---|---|---|
| 10 | 17.4 ¹ | 0.06 |
| 20 | 0.4 | 0.08 |
| 30 | 1.1 | 0.11 |
| 50 | 0.6 | 0.15 |
| 100 | 22.0 | 2.34 |

### ITL — NT PLAIN (avg, ms)

| Users | DGX GB10 | RTX Pro 6000 |
|---|---|---|
| 10 | 81.8 | 9.4 |
| 20 | 105.5 | 13.1 |
| 30 | 119.3 | 16.0 |
| 50 | 146.1 | 20.3 |
| 100 | 155.5 | 26.5 |

### RAG overhead — nothink (avg, seconds)

| Users | DGX GB10 | RTX Pro 6000 |
|---|---|---|
| 10 | 19.6 ¹ | 3.79 |
| 20 | 28.0 | 4.80 |
| 30 | 29.7 | 5.62 |
| 50 | 37.8 | 7.27 |
| 100 | 86.3 | 12.59 |

---

## Model: Qwen3.5-35B-A3B-FP8

`Qwen/Qwen3.5-35B-A3B-FP8` · `VLLM_REASONING_PARSER=qwen3`

> Think mode: completes on all hardware only at `MAX_MODEL_LEN ≥ 32768`. On GB10 (FP8, 32768), generation is too slow to complete within the 5-min window — zero think completions. L40S data is pilot only (5 completions, nothink NT RAG).

### Throughput — nothink (completions/s)

| Users | DGX GB10 | L40S (pilot) |
|---|---|---|
| 10 | 0.014 ¹ | ~0.086 |
| 20 | 0.028 | — |
| 30 | 0.043 | — |
| 50 | 0.058 | — |
| 100 | — | — |

¹ *Warm-up run.*

### ITL — NT PLAIN (avg, ms)

| Users | DGX GB10 | L40S (pilot) |
|---|---|---|
| 10 | 107.5 | ~22 |
| 20 | 152.2 | — |
| 30 | 178.9 | — |
| 50 | 210.9 | — |

---

## Model: Qwen3.5-35B-A3B-NVFP4

`AxionML/Qwen3.5-35B-A3B-NVFP4` · `VLLM_USE_FLASHINFER_MOE_FP4=1` · `VLLM_REASONING_PARSER=qwen3`

> RTX Pro 6000 results only — DGX GB10 sweep pending.

### Throughput — nothink (completions/s)

| Users | RTX Pro 6000 |
|---|---|
| 10 | 0.282 |
| 20 | 0.398 |
| 30 | 0.575 |
| 50 | 0.679 |
| 100 | 0.906 |

### Throughput — think (completions/s)

| Users | RTX Pro 6000 |
|---|---|
| 10 | 0.082 |
| 20 | 0.156 |
| 30 | 0.179 |
| 50 | 0.255 |
| 100 | 0.260 |

---

## Summary

**RTX Pro 6000 Blackwell vs DGX Spark GB10 — Nemotron NVFP4:**
The RTX Pro 6000 delivers **~16× higher nothink throughput** (4.25 vs 0.26 completions/s at 100 users) and **~6× lower ITL** (26 ms vs 155 ms). TTFT stays sub-second up to 50 users on the RTX Pro 6000 vs spiking to 22 s at 100 users on GB10. The gap is driven by dedicated GDDR7 bandwidth (RTX Pro 6000) vs shared unified LPDDR5X memory (GB10).

**Qwen3.5-35B FP8 vs NVFP4 — RTX Pro 6000:**
NVFP4 (0.28–0.91 nothink completions/s) outperforms the FP8 variant on GB10 (0.014–0.058) by 4–8× at equivalent user levels. Critically, NVFP4 makes think mode viable at all concurrency levels (0.08–0.26 completions/s) whereas FP8 on GB10 produces zero think completions within the 5-minute window.

**L40S — Qwen3.5 FP8:**
Pilot ITL (~22 ms) is ~5× better than GB10 (107 ms) at light load, driven by dedicated GDDR6 bandwidth. Full sweep pending.

---

*Naming convention for raw CSVs: `{mode}_u{N}_stats.csv` under each `results/<hardware>/<model>/` directory.*
