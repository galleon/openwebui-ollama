# Open WebUI + vLLM + Docling on DGX Spark

Local AI stack optimised for the **NVIDIA DGX Spark (GB10 / Blackwell sm_121)**.
vLLM is the sole inference backend; [Infinity](https://github.com/michaelfeil/infinity) handles embeddings.
Ollama is not used.

| Service | Image | Port | Profile |
|---|---|---|---|
| Open WebUI | `ghcr.io/open-webui/open-webui:main` | 3000 | *(always on)* |
| vLLM | custom (NGC PyTorch 26.01 base) | 8000 | *(always on)* |
| Embedder | custom (NGC PyTorch 26.01 base) | 7997 | *(always on)* |
| Docling | custom (NGC PyTorch 26.01 base) | 5001 | *(always on)* |
| Reranker | custom (NGC PyTorch 26.01 base) | 7998 | `reranker` |
| Qdrant | `qdrant/qdrant:latest` | 6333 / 6334 | `qdrant` |

---

## Prerequisites

- NVIDIA Container Toolkit installed and configured:
  ```bash
  nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- Docker Engine >= 24 with Compose v2
- ~40 GB free disk (NGC PyTorch base ~10 GB, models downloaded on first run)

---

## Quick start

```bash
# 1. Configure environment
cp .env.example .env
#    Edit .env — set WEBUI_SECRET_KEY and VLLM_MODEL at minimum

# 2. Build the three GB10-compatible images
#    (pulls ~10 GB NGC PyTorch base on first run, shared across all three)
docker compose build

# 3. Start everything
#    vLLM downloads VLLM_MODEL from HuggingFace on first run (~14 GB for a 7B)
docker compose up -d

# 4. Open the UI — vLLM models appear automatically once healthy (~2 min)
open http://localhost:3000
```

Docling UI (for testing document extraction): http://localhost:5001/ui

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Open WebUI :3000                   │
│          (chat · RAG · document upload)              │
└──────────┬──────────────┬──────────────┬─────────────┘
           │              │              │
       vLLM API     Embedder API    Docling API
       :8000/v1      :7997/v1         :5001
           │              │              │
   ┌───────┴──────┐ ┌─────┴──────┐ ┌───┴────────────┐
   │     vLLM     │ │  Infinity  │ │    Docling     │
   │  (inference) │ │(embeddings)│ │ (OCR + extract)│
   └──────────────┘ └────────────┘ └────────────────┘
         GPU              GPU            GPU
```

---

## GB10 unified memory budget

The GB10 has **128 GB unified memory** shared between CPU and GPU.
With the default `VLLM_GPU_MEMORY_UTILIZATION=0.55` and a 7B model:

| Component | Memory |
|---|---|
| vLLM model weights (7B FP16) | ~14 GB |
| vLLM KV cache (0.55 utilization) | ~63 GB |
| Embedder (bge-m3) | ~3 GB |
| Docling (EasyOCR) | ~3 GB |
| OS + Open WebUI | ~2 GB |
| **Total** | **~85 GB** |
| **Headroom** | **~43 GB** |

Raise `VLLM_GPU_MEMORY_UTILIZATION` toward `0.80` for larger context windows;
add the reranker (~2 GB) with `--profile reranker`.

---

## Why custom images?

The upstream images (`vllm/vllm-openai`, `michaelfeil/infinity`, `docling-serve-cu128`) target **CUDA 12.x** and lack native `sm_121` kernels for the GB10 Blackwell architecture, causing runtime JIT compilation failures.

All three `Dockerfile.*` files build on `nvcr.io/nvidia/pytorch:26.01-py3` which ships **CUDA 13.1** with full `sm_121` support.

---

## Configuration

All tunables live in `.env`. Key ones:

| Variable | Default | Notes |
|---|---|---|
| `WEBUI_SECRET_KEY` | *(must set)* | Change before first run |
| `VLLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Any HuggingFace model ID |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.55` | See memory budget above |
| `VLLM_MAX_MODEL_LEN` | `8192` | Context window in tokens |
| `EMBEDDER_MODEL` | `BAAI/bge-m3` | Any sentence-transformers model |
| `OMP_NUM_THREADS` | `8` | Grace CPU has 72 Arm cores |
| `DOCLING_WORKERS` | `2` | Parallel doc extraction workers |

### Use a reranker (hybrid search)

```bash
docker compose build reranker
```

Set in `.env`:
```env
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ENABLE_RAG_HYBRID_SEARCH=true
RAG_RERANKING_ENGINE=external
RAG_TOP_K=5
RAG_TOP_K_RERANKER=3
```

Start:
```bash
docker compose --profile reranker up -d
docker compose restart open-webui
```

### Use Qdrant as the vector store

By default Open WebUI uses its built-in **Chroma** database. To switch to **Qdrant**:

```bash
docker compose --profile qdrant up -d
```

Set in `.env`:
```env
VECTOR_DB=qdrant
QDRANT_URI=http://qdrant:6333
```

Then `docker compose restart open-webui`.

To use an **existing external Qdrant instance** (skip the profile):
```env
VECTOR_DB=qdrant
QDRANT_URI=http://<your-qdrant-host>:6333
QDRANT_API_KEY=<optional-key>
```

### Disable OCR for digital PDFs

```env
DOCLING_SERVE_PIPELINE_OPTIONS__DO_OCR=false
```

### Switch to upstream Docling image (no GB10 GPU support)

Comment out the `build:` block in `docker-compose.yml` and replace with:

```yaml
docling:
  image: quay.io/docling-project/docling-serve-cu128:latest
```

---

## Useful commands

```bash
# Stream all logs
docker compose logs -f

# Check GPU usage
nvidia-smi

# Rebuild all images after Dockerfile changes
docker compose build --no-cache

# Stop and remove containers (volumes preserved)
docker compose down

# Stop and wipe all data
docker compose down -v
```

---

## Benchmarking

`locustfile.py` benchmarks the OpenWebUI stack under concurrent load and captures streaming-specific metrics that plain HTTP benchmarkers miss.

### Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time To First Token — from request send to first streamed token (ms) |
| **ITL avg** | Average inter-token latency — smoothness of streaming (ms) |
| **ITL p95** | 95th-percentile inter-token latency — tail jitter; high p95 vs avg indicates stalls (ms) |
| **TPS** | Output throughput reported as ms-per-token — answer-length-neutral, lower is faster |
| **E2E** | Total end-to-end latency including RAG retrieval + full generation (ms) |
| **RAG overhead** | `TTFT(RAG) − TTFT(PLAIN)` — isolates the pure cost of vector retrieval (ms) |

Each metric appears as a separate row in the Locust stats table and CSV, for both `RAG` and `PLAIN` (no-KB) task prefixes.

### Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies (`locust`, `python-dotenv`) are declared inline in the script and installed automatically on first run.

Fill in the benchmarking section of `.env` (copied from `.env.example`):

```env
OPENWEBUI_API_KEY=<your-api-key>
OPENWEBUI_KB_ID=<knowledge-base-uuid>
OPENWEBUI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Edit `bench_questions.json` to match your knowledge base content:

```json
["Where can I buy a ticket?", "What time does the event start?"]
```

### Get your Knowledge Base UUID

```bash
curl -s http://localhost:3000/api/v1/knowledge \
  -H "Authorization: Bearer $OPENWEBUI_API_KEY" | jq '.[].id'
```

### Run

```bash
# Interactive web UI at http://localhost:8089
./locustfile.py --host http://localhost:3000

# Headless — 10 concurrent users, ramp 2/s, 60 s, save CSV
./locustfile.py --host http://localhost:3000 \
  --headless -u 10 -r 2 --run-time 60s \
  --csv=results/bench
```

### Interpreting results

Key comparisons:
- **RAG overhead avg** — pure retrieval cost; should stay below ~500 ms for a good UX
- **ITL p95 / ITL avg ratio** — values above ~3× indicate bursty generation (VRAM pressure, GC)
- **TPS PLAIN vs RAG** — should be similar; a large gap suggests the RAG context is exceeding the model's optimal context window

### Task weights

The locustfile runs RAG queries at 3× the rate of plain queries. Adjust the `@task` weights at the bottom of the file to change the mix.

---

## Ports summary

| Service | URL |
|---|---|
| Open WebUI | http://localhost:3000 |
| vLLM API | http://localhost:8000/v1 |
| Embedder API | http://localhost:7997/v1 |
| Docling API | http://localhost:5001 |
| Docling UI | http://localhost:5001/ui |
