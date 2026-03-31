# Ollama + Open WebUI + Docling on DGX Spark

Local AI stack optimised for the **NVIDIA DGX Spark (GB10 / Blackwell sm_121)**.

| Service | Image | Port |
|---|---|---|
| Ollama | `ollama/ollama:latest` | 11434 |
| Open WebUI | `ghcr.io/open-webui/open-webui:main` | 3000 |
| Docling | custom (NGC PyTorch 26.01 base) | 5001 |

---

## Prerequisites

- NVIDIA Container Toolkit installed and configured:
  ```bash
  nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- Docker Engine >= 24 with Compose v2
- ~30 GB free disk (NGC base image ~10 GB, models ~7 GB each)

---

## Quick start

```bash
# 1. Configure environment
cp .env.example .env
#    Edit .env — at minimum set WEBUI_SECRET_KEY

# 2. Build the GB10-compatible Docling image
#    (pulls ~10 GB NGC PyTorch base on first run)
docker compose build docling

# 3. Start Ollama first and pull an embedding model
docker compose up -d ollama
docker exec ollama ollama pull nomic-embed-text   # required for RAG
docker exec ollama ollama pull llama3.2           # or any model you want

# 4. Start everything
docker compose up -d

# 5. Open the UI
open http://localhost:3000
```

Docling UI (for testing document extraction): http://localhost:5001/ui

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Open WebUI :3000           │
│  (chat · RAG · document upload)         │
└────────────┬───────────────┬────────────┘
             │               │
      Ollama API       Docling API
      :11434               :5001
             │               │
    ┌────────┴──────┐  ┌─────┴──────────────┐
    │    Ollama     │  │      Docling       │
    │  (inference + │  │  (doc extraction + │
    │   embeddings) │  │   OCR + tables)    │
    └───────────────┘  └────────────────────┘
          GPU                  GPU
```

Documents uploaded to Open WebUI are sent to Docling for extraction, then chunked and embedded via Ollama into the RAG vector store.

---

## Why a custom Docling image?

The upstream `docling-serve-cu128` targets **CUDA 12.8**, which does not include kernels for the GB10 Blackwell architecture (`sm_121`). This causes runtime JIT compilation failures.

`Dockerfile.docling` builds on `nvcr.io/nvidia/pytorch:26.01-py3` which ships **CUDA 13.1** with native `sm_121` support. See the [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/gb10-and-docling/360665) for background.

---

## PDF handling

Docling uses **two backends** selectable via `DOCLING_SERVE_PDF_BACKEND`:

| Backend | Behaviour |
|---|---|
| `pypdfium2` (default) | Extracts native text directly from digital PDFs — fast, no OCR needed |
| `docling_parse` | Docling's own parser — useful for malformed PDFs |

**OCR** (EasyOCR, GPU-accelerated) is controlled globally by `DOCLING_SERVE_PIPELINE_OPTIONS__DO_OCR`:
- Default: **on** — runs on every page even if the PDF has a text layer
- Set to `false` for digital-PDF-only workloads for a significant speed gain
- Tesseract is available as a CPU fallback: `DOCLING_OCR_ENGINE=tesseract`

---

## Configuration

All tunables live in `.env`. Key ones:

| Variable | Default | Notes |
|---|---|---|
| `WEBUI_SECRET_KEY` | *(must set)* | Change before first run |
| `WEBUI_AUTH` | `true` | Set `false` only on a trusted local network |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Any Ollama-hosted embedding model |
| `OLLAMA_NUM_PARALLEL` | `4` | Concurrent inference requests |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Models kept in GPU VRAM |
| `OMP_NUM_THREADS` | `8` | Grace CPU has 72 Arm cores — tune to workload |
| `MKL_NUM_THREADS` | `8` | Same |
| `DOCLING_WORKERS` | `2` | Parallel doc extraction workers |

### Disable OCR for digital PDFs

```env
# .env
DOCLING_SERVE_PIPELINE_OPTIONS__DO_OCR=false
```

### Switch to upstream Docling image (no GB10 GPU support)

Comment out the `build:` block in `docker-compose.yml` and replace with:

```yaml
docling:
  image: quay.io/docling-project/docling-serve-cu128:latest
```

This runs without GPU acceleration on GB10 but avoids the custom build.

---

## Useful commands

```bash
# Stream all logs
docker compose logs -f

# Pull a new model
docker exec ollama ollama pull mistral

# Check GPU usage
nvidia-smi

# Rebuild Docling after Dockerfile changes
docker compose build --no-cache docling

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

```bash
pip install locust python-dotenv
```

Fill in the benchmarking section of `.env` (copied from `.env.example`):

```env
OPENWEBUI_API_KEY=<your-api-key>
OPENWEBUI_KB_ID=<knowledge-base-uuid>
OPENWEBUI_MODEL=llama3.2
```

### Get your Knowledge Base UUID

In Open WebUI → Workspace → Knowledge, open the knowledge base and copy the UUID from the URL, or via API:

```bash
curl -s http://localhost:3000/api/v1/knowledge \
  -H "Authorization: Bearer $OPENWEBUI_API_KEY" | jq '.[].id'
```

### Run

```bash
# Interactive web UI at http://localhost:8089
locust -f locustfile.py --host http://localhost:3000

# Headless — 10 concurrent users, ramp 2/s, 60 s, save CSV
locust -f locustfile.py --host http://localhost:3000 \
  --headless -u 10 -r 2 --run-time 60s \
  --csv=results/bench
```

### Interpreting results

The `results/bench_stats.csv` will contain rows for each metric type:

```
Type             Name                     Req  Fail  Avg(ms)  50%   95%    99%
RAG TTFT         time_to_first_token      120  0     1843     1710  2950   3800
RAG ITL avg      inter_token_latency_avg  120  0     42       38    71     95
RAG ITL p95      inter_token_latency_p95  120  0     71       65    120    160
RAG TPS          ms_per_output_token      120  0     55       50    90     115
RAG E2E          end_to_end_latency       120  0     8920     8400  13200  15600
RAG overhead     retrieval_overhead       120  0     1520     1400  2600   3200
PLAIN TTFT       time_to_first_token      40   0     320      290   540    710
PLAIN ITL avg    inter_token_latency_avg  40   0     38       35    65     88
PLAIN ITL p95    inter_token_latency_p95  40   0     65       60    105    140
PLAIN TPS        ms_per_output_token      40   0     50       46    82     108
PLAIN E2E        end_to_end_latency       40   0     4100     3800  6200   7900
```

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
| Ollama API | http://localhost:11434 |
| Docling API | http://localhost:5001 |
| Docling UI | http://localhost:5001/ui |
