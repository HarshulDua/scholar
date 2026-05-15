# Scholar Setup Guide — MVP Through Full Pipeline

> Complete setup instructions for Windows. Two sections:
> **Part A** — MVP (search + recommend working, no training required, ~30 min)
> **Part B** — Full Phase 2 pipeline (training, LoRA, all evals, ~12h GPU time)

---

## Prerequisites

- Python 3.11+ on PATH (Scholar was built on Python 3.13.5 / Anaconda)
- Docker Desktop installed and running
- NVIDIA GPU recommended (RTX 2050 4GB VRAM or better) for LoRA training and /explain
- Kaggle account + API key (for ArXiv dataset download)
  - Go to https://www.kaggle.com/settings → API → Create New Token → download kaggle.json
  - Place at `C:\Users\<you>\.kaggle\kaggle.json`

**Verified hardware (Scholar's dev machine):**
```
GPU:    RTX 2050 · 4.00 GB VRAM · Compute 8.6 · CUDA 12.4
RAM:    15.7 GB
Storage: 931.5 GB D drive with ~419 GB free
```

---

## Part A — MVP Setup (~30 minutes)

### A1. Clone and configure

```powershell
# Clone the project
git clone https://github.com/YOUR_USERNAME/scholar.git
cd scholar

# Create environment file
Copy-Item .env.example .env
notepad .env
```

Edit `.env` — change at minimum:
```
POSTGRES_PASSWORD=scholar123       # pick any password
DEVICE=cuda                        # or "cpu" if no GPU
GENERATION_AVAILABLE=true          # false if skipping /explain
```

Leave all other values as defaults.

### A2. Start PostgreSQL

```powershell
docker compose up db -d
docker ps   # should show scholar-db-1 with status "Up"
```

### A3. Create Python environment

```powershell
pip install uv
uv venv
.venv\Scripts\activate             # you should see (.venv) in prompt

uv pip install -e ".[dev]"         # ~45 seconds with uv (vs ~8 min with pip)
```

### A4. Download ArXiv data

```powershell
# Fast sample (~50MB):
kaggle datasets download -d Cornell-University/arxiv -p ./data --unzip
# Full dataset is ~4GB — loader streams it, no need to wait

# Alternative: run loader without full dataset (loads whatever exists)
```

### A5. Load papers into Postgres

```powershell
# 5,000 papers = fast test run (~2 minutes)
python -m scholar.ingest.arxiv_loader --limit 5000

# For MVP with meaningful search results:
python -m scholar.ingest.arxiv_loader --limit 50000   # ~5 minutes
```

### A6. Build indexes

```powershell
# Build both BM25 and dense (sentence-BERT) indexes
python scripts/build_index.py

# Or individually:
python scripts/build_index.py --bm25-only
python scripts/build_index.py --faiss-only    # actually builds embeddings.npy
```

This takes ~3-10 minutes for 50K papers. Output:
- `data/bm25.pkl` — BM25 inverted index
- `data/embeddings.npy` — 384-dim SBERT embeddings

### A7. Start the API

```powershell
uvicorn scholar.api.main:app --reload --port 8000
# Should print: "Application startup complete."
# Leave this terminal open.
```

### A8. Test via Swagger UI

Open http://localhost:8000/docs in your browser.

**Test sequence:**

1. **GET /health** → `{"status": "ok"}`

2. **POST /user** → creates a user, copy the `user_id`

3. **POST /search**
   ```json
   {"query": "diffusion models image generation", "level": "grad", "top_k": 5}
   ```
   → Returns ranked papers. Copy a `paper_id` from the response.

4. **POST /user/{user_id}/history** → add the paper to your history (enables personalization)

5. **POST /recommend** with your `user_id` → personalized recommendations

6. **POST /explain** (requires CUDA + Phi-3 loaded)
   ```json
   {"paper_id": "2301.00001", "level": "grad", "why": "top result for your query"}
   ```
   First call: downloads Phi-3-mini (~2.4GB, 5-10 min). Subsequent: ~10s.

### A9. Start the Gradio UI (optional)

```powershell
# In a new terminal (with .venv active):
python scholar/ui/gradio_app.py
# Open http://localhost:7860
```

Or use Docker Compose for everything:
```powershell
docker compose up   # starts db + api + ui together
```

---

## Part B — Full Phase 2 Pipeline

**Time estimate:** ~12 hours total (dominated by LoRA training at 10.3h on RTX 2050).
Run overnight — the scripts are designed to be launched and left.

**Prerequisites:** Part A complete + 200K papers loaded.

### B1. Load 200K papers (if not done yet)

```powershell
python -m scholar.ingest.arxiv_loader --limit 200000
# ~6 seconds with psycopg2 batch upsert (vs hours with naive approach)
```

### B2. Build full indexes

```powershell
python scripts/build_index.py
# BM25: ~2 minutes for 200K papers → data/bm25.pkl (~100MB)
# Embeddings: ~15-20 minutes on GPU → data/embeddings.npy (~307MB)
```

### B3. Load CiteULike interaction data

```powershell
# Download CiteULike-A dataset first:
# From: https://github.com/js05212/citeulike-a (download citeulike-a.zip)
# Extract to: ./data/citeulike-a/

python -m scholar.ingest.citeulike_loader --data-dir ./data/citeulike-a
# Matches CiteULike papers to ArXiv corpus via title matching
# ~158 of 16,980 CiteULike papers match (different ID systems)
# Creates user_history entries for matched papers
```

### B4. Train Mult-VAE

```powershell
python scholar/recsys/train_vae.py
# Trains on CiteULike interaction matrix
# ~20-40 minutes on GPU
# Saves: data/mult_vae.pt + data/vae_vocab.json
```

**Expected output:**
```
Epoch 10 | loss=0.045 | NDCG@10=0.000 | beta=0.10
...
Epoch 100 | loss=0.031 | best_ndcg=0.000 | beta=0.20
```

NDCG≈0 is expected (corpus mismatch — CiteULike papers not in ArXiv corpus).

### B5. Train LightGBM LTR

```powershell
python scripts/train_ltr.py
# Generates training data by running search queries and collecting LTR features
# ~5-10 minutes
# Saves: data/ltr_model.txt
```

### B6. Run retrieval eval

```powershell
python -m scholar.retrieval.eval
# Evaluates on NFCorpus (medical information retrieval dataset)
# ~10-15 minutes (downloads dataset first time)
# Updates docs/benchmarks.md Layer 1 table
```

**Expected results:**
```
BM25     | NDCG@10=0.xx | MRR@10=0.34 | p50=26.6ms | p99=44.5ms
Dense    | Recall@200=1.0             | p50=121.6ms | p99=196ms
Hybrid   | NDCG@10=0.xx              | p50=106.5ms | p99=196.2ms
```

### B7. Run recsys eval

```powershell
python -m scholar.recsys.eval
# Evaluates Mult-VAE + LTR + MMR + ε-greedy on CiteULike test split
# ~5-10 minutes
# Updates docs/benchmarks.md Layer 2 table
```

**Expected results:**
```
MMR only:           NDCG@10=0.000 | ILD=0.496
LTR + MMR:          NDCG@10=0.000 | ILD=0.496
LTR + MMR + greedy: NDCG@10=0.000 | ILD=0.576 (+16.2%)
```

### B8. Generate synthetic training data

```powershell
python scholar/generation/synth_data.py --limit 300
# Generates 300 training examples using base Phi-3
# 100 examples × 3 levels (undergrad, grad, researcher)
# ~2-3 hours on GPU (60 Phi-3 inference calls per hour)
# Saves: data/synth_train.jsonl
```

**Note:** Requires DEVICE=cuda in .env. Each generation takes ~60-120 seconds.

### B9. Run LoRA fine-tuning

```powershell
python scholar/generation/train_lora.py
# Fine-tunes Phi-3-mini-4k-instruct with QLoRA (4-bit NF4)
# 3 epochs × 114 steps = ~10.3 hours on RTX 2050
# Saves: data/lora_adapter/ (adapter_model.safetensors ~6.3MB)
```

**Expected output:**
```
{'loss': 1.245, 'grad_norm': 0.43, 'learning_rate': 2e-4, 'epoch': 0.26}
...
{'train_loss': 1.037, 'train_runtime': 37100, 'train_samples_per_second': 0.024}
```

**Hardware note:** QLoRA uses ~3.5GB VRAM during training. Close all other GPU-using apps.
Do NOT run this simultaneously with sentence-BERT embedding or any other Phi-3 process.

### B10. Run generation eval

```powershell
python -m scholar.generation.eval
# Evaluates base Phi-3 vs Phi-3+LoRA on 10 ArXiv papers × 3 levels
# ~1-2 hours on GPU (60 inference calls)
# Updates docs/benchmarks.md Layer 3 table
```

**Expected results (Scholar's actual):**

| Model | Level | ROUGE-L | BERTScore F1 |
|-------|-------|---------|--------------|
| Phi-3 base | undergrad | 0.070 | 0.835 |
| Phi-3 base | grad | 0.120 | 0.845 |
| **Phi-3 + LoRA** | grad | 0.124 | **0.850** |
| **Phi-3 + LoRA** | researcher | 0.113 | **0.849** |

### B11. Run tests

```powershell
pytest && ruff check scholar/ && mypy scholar/
# Expected: 63 passed, 0 ruff errors, 0 mypy errors
```

---

## Hardware Budget and Constraints

### Storage (D drive)

| Data | Size |
|------|------|
| ArXiv JSON (200K) | ~4.0 GB |
| PostgreSQL data | ~2.0 GB |
| embeddings.npy (200K×384) | ~0.3 GB |
| BM25 pickle | ~0.1 GB |
| HF: all-MiniLM-L6-v2 | ~0.1 GB |
| HF: Phi-3-mini 4-bit | ~2.3 GB |
| synth_train.jsonl | ~0.05 GB |
| LoRA adapter | ~0.03 GB |
| **Total 200K pipeline** | **~9 GB** |

### RAM

| Component | Memory |
|-----------|--------|
| BM25 index (200K) | ~100 MB |
| embeddings.npy | ~307 MB |
| sentence-BERT model | ~200 MB |
| FastAPI + Gradio | ~500 MB |
| **Total (no Phi-3)** | **~1.1 GB** |
| Phi-3 in VRAM | ~1.9 GB VRAM |

**Warning:** Never run sentence-BERT embedding and Phi-3 generation simultaneously —
they compete for VRAM. Build indexes first, then start API with Phi-3.

### VRAM Budget (RTX 2050, 4GB)

| Operation | VRAM |
|-----------|------|
| Phi-3 4-bit inference | ~2.7 GB |
| QLoRA training | ~3.5 GB (tight — close all other GPU apps) |
| Available buffer | ~0.5-1.3 GB |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `connection refused` starting API | `docker ps` — ensure db container is Up |
| `ModuleNotFoundError` | Ensure `.venv\Scripts\activate` is active |
| BM25/FAISS index not found | Run `python scripts/build_index.py`, then restart uvicorn |
| `OSError: paging file too small` | Close ALL other GPU apps; VRAM full. Increase paging file to 8-16GB in Windows advanced settings |
| `ValidationError: Extra inputs not permitted` | Set `extra="ignore"` in pydantic-settings config (already fixed in current codebase) |
| `SFTTrainer unexpected keyword 'tokenizer'` | Use `processing_class=tok` (trl 1.4+ API) — already fixed |
| `Windows fatal exception: access violation` | `import pyarrow as _pa` must be FIRST import in entrypoints — already fixed |
| arxiv_loader `file not found` | Ensure `arxiv-metadata-oai-snapshot.json` is in `./data/`. Check `ARXIV_DATA_PATH` in `.env` |
| `/explain` takes 5+ minutes (first call) | Expected — Phi-3 downloading. Set `WARMUP_GENERATION=true` to pre-load at startup |

---

## Two-Environment Setup

Scholar supports two configurations via `.env` files:

**CPU environment (`.env.cpu`):** data ingestion, index building, VAE training, testing.
Phi-3 /explain disabled (30s/token = unusable).
```
cp .env.cpu .env
```

**CUDA environment (`.env.cuda`):** full demo with AI explanations, synth data, LoRA training.
```
cp .env.cuda .env
```

Switch between environments by copying the relevant file to `.env` and restarting the API.

---

## What Files Are Created

```
data/
├── arxiv-metadata-oai-snapshot.json   ← raw ArXiv dump (~4GB)
├── bm25.pkl                           ← BM25 inverted index (~100MB)
├── embeddings.npy                     ← SBERT embeddings 200K×384 (~307MB)
├── embeddings_ids.json                ← parallel paper ID list
├── mult_vae.pt                        ← trained Mult-VAE weights
├── vae_vocab.json                     ← paper ID→index mapping for VAE
├── ltr_model.txt                      ← LightGBM LambdaRank model
├── synth_train.jsonl                  ← 300 synthetic training examples
├── lora_adapter/                      ← fine-tuned LoRA adapter (6.3MB)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
└── hf_cache/                          ← HuggingFace model cache
    ├── all-MiniLM-L6-v2/             ← sentence-BERT (~80MB)
    └── Phi-3-mini-4k-instruct/       ← Phi-3 weights (~2.3GB at 4-bit)
```
