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
    │    Ollama     │  │      Docling        │
    │  (inference + │  │  (doc extraction +  │
    │   embeddings) │  │   OCR + tables)     │
    └───────────────┘  └─────────────────────┘
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

## Ports summary

| Service | URL |
|---|---|
| Open WebUI | http://localhost:3000 |
| Ollama API | http://localhost:11434 |
| Docling API | http://localhost:5001 |
| Docling UI | http://localhost:5001/ui |
