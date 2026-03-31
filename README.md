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
| **ITL** | Inter-Token Latency — average gap between consecutive tokens (ms) |
| **E2E** | End-to-end latency including RAG retrieval + full generation (ms) |

Each metric is reported as a separate row in the Locust stats table and CSV, for both the RAG task and the plain (no-KB) baseline.

### Setup

```bash
pip install locust
```

### Get your Knowledge Base UUID

In Open WebUI → Workspace → Knowledge, open the knowledge base and copy the UUID from the URL, or via API:

```bash
curl -s http://localhost:3000/api/v1/knowledge \
  -H "Authorization: Bearer <api-key>" | jq '.[].id'
```

### Run

```bash
export OPENWEBUI_API_KEY=<your-api-key>
export OPENWEBUI_KB_ID=<knowledge-base-uuid>
export OPENWEBUI_MODEL=llama3.2

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
Type        Name                     Req   Fail   Avg(ms)   50%   95%   99%   Max
RAG TTFT    time_to_first_token      120   0      1843      1710  2950  3800  4200
RAG ITL     inter_token_latency_avg  120   0      42        38    71    95    130
RAG E2E     end_to_end_latency       120   0      8920      8400  13200 15600 18000
PLAIN TTFT  time_to_first_token      40    0      320       290   540   710   850
PLAIN ITL   inter_token_latency_avg  40    0      38        35    65    88    120
PLAIN E2E   end_to_end_latency       40    0      4100      3800  6200  7900  9100
```

**TTFT difference (RAG − PLAIN)** is the retrieval overhead added by the RAG pipeline.

### Task weights

The locustfile runs RAG queries at 3× the rate of plain queries. Adjust at the bottom of the file to change the mix.

---

## Ports summary

| Service | URL |
|---|---|
| Open WebUI | http://localhost:3000 |
| Ollama API | http://localhost:11434 |
| Docling API | http://localhost:5001 |
| Docling UI | http://localhost:5001/ui |
