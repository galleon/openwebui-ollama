# Open WebUI + vLLM + Docling on DGX Spark

Local AI stack optimised for the **NVIDIA DGX Spark (GB10 / Blackwell sm_121)**.
vLLM is the sole inference backend; [Infinity](https://github.com/michaelfeil/infinity) handles embeddings.
Ollama is not used.

| Service | Image | Port | Profile |
|---|---|---|---|
| Open WebUI | `ghcr.io/open-webui/open-webui:main` | 3000 | *(always on)* |
| vLLM | `nvcr.io/nvidia/vllm:26.02-py3` | 8000 | *(always on)* |
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
#    Edit .env — set WEBUI_SECRET_KEY, VLLM_MODEL, and HUGGING_FACE_HUB_TOKEN

# 2. Download the Nemotron reasoning parser (required for all Nemotron-Nano models)
mkdir -p ./vllm_plugins
wget -O ./vllm_plugins/nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py

# 3. Build the GB10-compatible images (Docling + Infinity)
#    vLLM uses the official NGC image — no build needed for it
docker compose build docling embedder

# 4. Start everything
#    vLLM pulls nvcr.io/nvidia/vllm:26.02-py3 then downloads VLLM_MODEL from HF
docker compose up -d

# 5. Open the UI — vLLM models appear automatically once healthy (~15 min for 30B)
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

## Why custom images for Docling and Infinity?

The upstream `michaelfeil/infinity` and `docling-serve-cu128` images target **CUDA 12.x** and lack native `sm_121` kernels for the GB10 Blackwell architecture, causing runtime JIT compilation failures.

`Dockerfile.docling` and `Dockerfile.infinity` build on `nvcr.io/nvidia/pytorch:26.01-py3` which ships **CUDA 13.1** with full `sm_121` support.

vLLM uses NVIDIA's official NGC image (`nvcr.io/nvidia/vllm:26.02-py3`) which already includes Blackwell support — no custom build needed.

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

## Embedder and reranker: GPU vs CPU

The embedder and reranker default to `--device cuda`. This section explains the trade-off if you switch to CPU.

### Throughput comparison

From Infinity startup logs, `BAAI/bge-m3` on the GB10 GPU achieves **87–2056 embeddings/sec** (batch_size=32). On CPU:

| Scenario | GPU | CPU (Grace, 72 cores) |
|---|---|---|
| Query embedding (RAG lookup) | ~15 ms | ~100–300 ms |
| Document ingestion (batch) | ~12 ms/batch | ~200–500 ms/batch |
| Reranking 5 candidates | ~20 ms | ~150–400 ms |
| RAG overhead per query | ~50 ms | ~300–700 ms |

### GPU memory freed by switching to CPU

| Service | GPU memory reclaimed |
|---|---|
| Embedder (bge-m3) | ~3 GB |
| Reranker (bge-reranker-v2-m3) | ~2 GB |
| **Total** | **~5 GB** |

That 5 GB could be redirected to a larger vLLM KV cache (~4K–8K extra context tokens at fp8).

### When CPU is acceptable

| Workload | CPU verdict |
|---|---|
| Single user, interactive chat | Acceptable — 300–700 ms RAG overhead is barely noticeable |
| Concurrent users (> 3) | Bottleneck — CPU saturates, RAG latency spikes to seconds |
| Bulk document ingestion | Painful — a 100-page document takes minutes instead of seconds |

### Recommendation

Keep both services on GPU. The GB10 has 128 GB unified memory — 5 GB is not worth a 10–30× throughput regression. Switch to CPU only if you are running a model large enough to be genuinely memory-constrained.

To switch, change `--device cuda` to `--device cpu` in the `embedder` and `reranker` commands in `docker-compose.yml`.

---

## RAG architecture and HA considerations

### Default configuration

Out of the box, the stack uses three components that are **node-local**:

| Component | Default | Location |
|---|---|---|
| Vector store | Chroma (built-in) | Inside the `open-webui` container, stored in the `open_webui_data` Docker volume |
| Embedder | Infinity (`BAAI/bge-m3`) | Local GPU service on port 7997 |
| Reranker | Infinity (`BAAI/bge-reranker-v2-m3`) | Local GPU service on port 7998 (optional profile) |

All knowledge bases, chat history, and user data live inside the `open_webui_data` named volume on a single machine.

### Pros of the default setup

- **Zero external dependencies** — fully self-contained, works immediately after `docker compose up`
- **Low latency** — embedder and vector store are on the same host, no network round-trips for RAG
- **Simple operations** — one machine to manage, backup, or wipe
- **GPU-accelerated embeddings** — Infinity uses the GB10 GPU, much faster than CPU-based alternatives

### Cons for a 2-site HA setup

| Problem | Impact |
|---|---|
| **Chroma is not distributed** | Each site has its own independent vector store; knowledge bases are not shared between sites |
| **No replication** | Documents uploaded on site A are invisible on site B |
| **Local embedder** | Each site embeds independently — if embedding models diverge (version, config), vectors become incompatible across sites |
| **Stateful Open WebUI volume** | User accounts, chat history, and settings are not synchronised; a user logging in on site B sees a different state than on site A |
| **Single point of failure** | If the DGX Spark on one site goes down, that site loses the entire stack — there is no failover |

### Improving the setup for HA / multi-site

The two changes with the highest impact are externalising the vector store and the Open WebUI database.

#### 1. Shared vector store — external Qdrant cluster

Replace per-site Chroma with a **shared Qdrant cluster** (or Qdrant Cloud). Both sites point at the same instance; documents uploaded anywhere are immediately queryable everywhere.

```env
VECTOR_DB=qdrant
QDRANT_URI=http://<shared-qdrant-host>:6333
QDRANT_API_KEY=<your-key>
```

For true HA, deploy Qdrant in distributed mode across nodes (Qdrant supports sharding and replication natively). A minimal 2-node setup with `replication_factor=2` survives a single-node failure.

#### 2. Shared Open WebUI database — external Postgres

By default Open WebUI uses SQLite inside the container. For multi-site you need a shared relational database so that users, knowledge base metadata, and chat history are consistent across sites.

Set the `DATABASE_URL` environment variable in `docker-compose.yml` under `open-webui`:

```yaml
- DATABASE_URL=postgresql://user:password@<shared-postgres-host>:5432/openwebui
```

Open WebUI supports PostgreSQL out of the box via SQLAlchemy.

#### 3. Consistent embeddings across sites

Both sites must use the **same embedding model at the same version** — otherwise vectors stored by site A are not comparable to queries from site B, breaking RAG retrieval.

Pin the model explicitly in `.env` and avoid `latest` tags:

```env
EMBEDDER_MODEL=BAAI/bge-m3
```

The HF model cache (`./hf_cache`) should be pre-populated and kept in sync, or pointed at a shared NFS/S3-backed cache, to avoid re-downloading on each site.

#### 4. Active/passive vs active/active

| Mode | Approach | Complexity |
|---|---|---|
| **Active/passive** | DNS failover or load balancer points users to the live site; passive site is warm but idle | Low — shared Qdrant + Postgres is sufficient |
| **Active/active** | Both sites serve traffic simultaneously; load balancer distributes requests | High — also requires session affinity or stateless Open WebUI sessions |

For most deployments, **active/passive with shared Qdrant + Postgres** is the right starting point: it eliminates data loss on failover while keeping operational complexity manageable.

### Summary

```
Default (single site)          HA (2 sites)
─────────────────────          ────────────
Chroma (local volume)    →     Qdrant cluster (shared, replicated)
SQLite (local volume)    →     PostgreSQL (shared)
Embedder (local GPU)     →     Same model version on each site, same HF cache
Open WebUI (stateful)    →     Stateless sessions + shared DB
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
