# Benchmarking guide — RTX Pro 6000 Blackwell

This guide covers deploying the stack for concurrent-user benchmarking on one or two
RTX Pro 6000 Blackwell GPUs, including a split topology where Open WebUI runs on a
separate machine (e.g. a Mac) while GPU-intensive services run on the rental node.

---

## Renting on brev.nvidia.com

[Brev](https://brev.nvidia.com) is NVIDIA's GPU cloud platform and the recommended
way to rent an RTX Pro 6000 Blackwell for this benchmark.

### NGC authentication — why nvcr.io credentials are required

The vLLM service uses NVIDIA's official NGC container image
(`nvcr.io/nvidia/vllm:26.02-py3`). NGC is a private registry — unauthenticated pulls
will fail with a `401 Unauthorized` error. You need a free NGC account and API key:

1. Create an account at [ngc.nvidia.com](https://ngc.nvidia.com)
2. Go to **Account → Setup → Generate API Key**
3. Copy the key — you will use it as the password below

### Docker Registry Credentials in Brev

Brev lets you store Docker registry credentials once in the UI — they are injected at
the daemon level on every instance, so no manual `docker login` is needed on the node.

Add your NGC key under **Account → Docker Registry Credentials**:

| Field | Value |
|---|---|
| Registry | `nvcr.io` |
| Username | `$oauthtoken` |
| Password | Your NGC API key from step 2 above |

`docker compose pull` and `docker compose up` will authenticate automatically from
that point on.

### HF model cache

Brev instance disk is ephemeral across restarts. To avoid re-downloading models on
each session, point the HF cache at a path on the persistent volume Brev provides
(typically `/home/ubuntu`):

```env
HF_CACHE_DIR=/home/ubuntu/hf_cache
```

### Port forwarding to your Mac (Topology B)

To run Open WebUI on your Mac while GPU services run on the Brev instance, forward
the service ports over SSH using the Brev CLI:

```bash
# Install Brev CLI on your Mac
brew install brevdev/homebrew-brev/brev
brev login

# Forward all GPU service ports (run in a dedicated terminal, keep it open)
brev port-forward <instance-name> \
  --port 8000 \   # vLLM
  --port 7997 \   # embedder
  --port 7998 \   # reranker
  --port 5001 \   # docling
  --port 6333     # qdrant
```

Then use `RTX_HOST=localhost` in the Open WebUI `docker run` command from
[Topology B](#topology-b----open-webui-on-a-separate-machine-eg-mac).

### Quick-start on the Brev instance

```bash
# 1. Clone the repo
git clone https://github.com/galleon/openwebui-vllm.git
cd openwebui-vllm
git checkout bench/rtx-pro-6000

# 2. Download the Nemotron reasoning parser
mkdir -p ./vllm_plugins
wget -O ./vllm_plugins/nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py

# 3. Configure
cp .env.rtx6000 .env
# Edit .env — set WEBUI_SECRET_KEY, HUGGING_FACE_HUB_TOKEN, HF_CACHE_DIR

# 4. Build GB10-specific images (Docling + Infinity)
docker compose build docling embedder reranker

# 5. Start
docker compose --profile reranker \
  -f docker-compose.yml -f docker-compose.rtx6000.yml up -d
```

---

## Component CPU / GPU mapping

| Component | Compute | Reason |
|---|---|---|
| **Open WebUI** | CPU only | Python web app — no GPU workload at all |
| **vLLM** | GPU (required) | LLM inference is entirely GPU-bound |
| **Embedder (Infinity)** | GPU (recommended) | ~3 GB; 10–30× slower on CPU — bottlenecks RAG under concurrent load |
| **Reranker (Infinity)** | GPU (recommended) | ~2 GB; latency-sensitive on the hot RAG path |
| **Docling** | CPU (during benchmarks) | ~9 GB GPU saved for vLLM KV cache; ingestion is not on the query hot path |
| **Qdrant** | CPU only | Vector search is CPU-bound by design |

### Memory budget on a single RTX Pro 6000 (96 GB GDDR7) with Docling on CPU

| Component | VRAM |
|---|---|
| vLLM at 0.80 utilization (weights ~15 GB + KV cache ~62 GB) | ~77 GB |
| Embedder (bge-m3) | ~3 GB |
| Reranker (bge-reranker-v2-m3) | ~2 GB |
| **Total** | **~82 GB / 96 GB** |
| **Headroom** | **~14 GB** |

---

## Deployment topologies

### Topology A — all services on one RTX node (simplest)

```
┌─────────────────────────────── RTX Pro 6000 node ───────────────────────────────┐
│                                                                                   │
│  Open WebUI (CPU)  ──►  vLLM :8000 (GPU)                                        │
│                    ──►  Embedder :7997 (GPU)                                     │
│                    ──►  Reranker :7998 (GPU)                                     │
│                    ──►  Docling :5001 (CPU)                                      │
│                    ──►  Qdrant :6333 (CPU)                                       │
└───────────────────────────────────────────────────────────────────────────────────┘
```

```bash
cp .env.rtx6000 .env
# edit .env — set WEBUI_SECRET_KEY, HUGGING_FACE_HUB_TOKEN
docker compose --profile reranker --profile qdrant \
  -f docker-compose.yml -f docker-compose.rtx6000.yml up -d
```

---

### Topology B — Open WebUI on a separate machine (e.g. Mac)

Open WebUI has no GPU dependency. It is a pure web application that talks to the AI
services over HTTP. You can run it on any machine — including a Mac — and point it at
the GPU services on the rental node.

```
┌──── Mac (or any laptop) ────┐        ┌──── RTX Pro 6000 node ────────────────┐
│                              │        │                                        │
│  Open WebUI :3000 (CPU)  ───┼──────► │  vLLM :8000 (GPU)                    │
│                              │        │  Embedder :7997 (GPU)                 │
│                              │        │  Reranker :7998 (GPU)                 │
│                              │        │  Docling :5001 (CPU)                  │
│                              │        │  Qdrant :6333 (CPU)                   │
└──────────────────────────────┘        └────────────────────────────────────────┘
```

**On the RTX node** — start everything except Open WebUI:

```bash
cp .env.rtx6000 .env
# Edit .env — set WEBUI_SECRET_KEY, HUGGING_FACE_HUB_TOKEN
docker compose --profile reranker --profile qdrant \
  -f docker-compose.yml -f docker-compose.rtx6000.yml \
  up -d vllm embedder reranker docling qdrant
```

**On the Mac** — run only Open WebUI, pointing at the remote node (`RTX_HOST` = IP or hostname of the rental node):

```bash
export RTX_HOST=<rtx-node-ip>

docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -v open_webui_data:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://${RTX_HOST}:8000/v1 \
  -e OPENAI_API_KEY=EMPTY \
  -e CONTENT_EXTRACTION_ENGINE=docling \
  -e DOCLING_SERVER_URL=http://${RTX_HOST}:5001 \
  -e RAG_EMBEDDING_ENGINE=openai \
  -e RAG_EMBEDDING_MODEL=BAAI/bge-m3 \
  -e RAG_OPENAI_API_BASE_URL=http://${RTX_HOST}:7997 \
  -e RAG_OPENAI_API_KEY=EMPTY \
  -e ENABLE_RAG_HYBRID_SEARCH=true \
  -e RAG_RERANKING_ENGINE=external \
  -e RAG_EXTERNAL_RERANKER_URL=http://${RTX_HOST}:7998/rerank \
  -e RAG_EXTERNAL_RERANKER_API_KEY=EMPTY \
  -e VECTOR_DB=qdrant \
  -e QDRANT_URI=http://${RTX_HOST}:6333 \
  -e QDRANT_TIMEOUT=5 \
  -e WEBUI_AUTH=true \
  -e WEBUI_SECRET_KEY=<your-secret> \
  ghcr.io/open-webui/open-webui:main
```

> Make sure ports 8000, 7997, 7998, 5001, and 6333 are reachable from the Mac
> (firewall rules / security groups on the rental node).

---

### Topology C — two RTX nodes, independent stacks (horizontal scaling benchmark)

Two fully independent stacks behind a load balancer. Best for measuring how throughput
scales with additional GPUs under real concurrent load.

```
             ┌─── load balancer / locust ───┐
             │                              │
     ┌───────▼──────────┐        ┌──────────▼───────────┐
     │  RTX node 0      │        │  RTX node 1           │
     │  Stack A :3000   │        │  Stack B :3001        │
     │  vLLM    :8000   │        │  vLLM    :8001        │
     │  Embedder:7997   │        │  Embedder:7999        │
     │  Reranker:7998   │        │  Reranker:7998        │
     └──────────────────┘        └───────────────────────┘
              GPU 0                        GPU 1
```

```bash
# Stack A — GPU 0
CUDA_VISIBLE_DEVICES=0 docker compose --profile reranker \
  -f docker-compose.yml -f docker-compose.rtx6000.yml \
  --project-name stack-a up -d

# Stack B — GPU 1 (different ports)
CUDA_VISIBLE_DEVICES=1 \
VLLM_PORT=8001 EMBEDDER_PORT=7999 RERANKER_PORT=7900 WEBUI_PORT=3001 \
docker compose --profile reranker \
  -f docker-compose.yml -f docker-compose.rtx6000.yml \
  --project-name stack-b up -d
```

---

## Running the benchmark

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed on the machine running locust
- Open WebUI running and accessible
- A knowledge base created and populated with documents
- API key generated in Open WebUI → Settings → Account → API Keys

### Configure

Fill in `.env` (or export directly):

```env
OPENWEBUI_API_KEY=<your-api-key>
OPENWEBUI_KB_ID=<knowledge-base-uuid>
OPENWEBUI_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
```

Get the knowledge base UUID:

```bash
curl -s http://localhost:3000/api/v1/knowledge \
  -H "Authorization: Bearer $OPENWEBUI_API_KEY" | jq '.[].id'
```

Edit `bench_questions.json` to match your knowledge base content:

```json
["Where can I buy a ticket?", "What time does the event start?"]
```

### Run

```bash
# Interactive UI at http://localhost:8089
./locustfile.py --host http://localhost:3000

# Headless ramp: 1 → 32 users, measure saturation point
./locustfile.py --host http://localhost:3000 \
  --headless -u 32 -r 2 --run-time 300s \
  --csv=results/rtx6000_bench
```

For Topology C (two stacks), run locust against both and compare:

```bash
./locustfile.py --host http://<node-ip>:3000 --csv=results/gpu0 &
./locustfile.py --host http://<node-ip>:3001 --csv=results/gpu1
```

### Metrics to watch

| Metric | What to look for |
|---|---|
| **TTFT vs concurrency** | Should stay flat until request queue fills — inflection point = real concurrent capacity |
| **ITL p95 / ITL avg** | Ratio > 3× indicates KV cache eviction or scheduling pressure |
| **TPS PLAIN vs RAG** | Large gap means RAG context is pushing against the model's optimal context window |
| **RAG overhead** | Should stay below ~500 ms; spikes indicate embedder or reranker saturation |

### GPU monitoring during the run

```bash
# Live GPU utilization and memory
watch -n1 nvidia-smi

# Per-process memory breakdown
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv -l 5
```

### Comparing GB10 vs RTX Pro 6000

Run the identical locust profile on both machines and compare CSVs:

```bash
# On GB10
./locustfile.py --host http://dgx:3000 --headless -u 16 -r 1 \
  --run-time 120s --csv=results/gb10

# On RTX Pro 6000
./locustfile.py --host http://rtx:3000 --headless -u 16 -r 1 \
  --run-time 120s --csv=results/rtx6000
```

Key expected differences:
- **TPS**: RTX Pro 6000 GDDR7 (~1.8 TB/s) vs GB10 unified memory (~900 GB/s) — expect RTX to be ~1.5–2× faster on token generation
- **TTFT under load**: RTX dedicated VRAM avoids CPU/GPU memory contention present on GB10
- **Concurrent capacity**: Higher `VLLM_MAX_NUM_SEQS=32` + more KV cache allows more parallel requests before TTFT degrades
