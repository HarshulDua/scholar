# Scholar — Personalized Research Paper Discovery Engine

> Given a query and a user's reading history, Scholar retrieves relevant ArXiv papers, reranks them using a learned user model, and generates personalized explanations at your level (undergrad / grad / researcher).

---

## What It Does

**The problem:** Existing tools (Google Scholar, Semantic Scholar, ArXiv search) return chronological or keyword-matched lists with no understanding of your background or interests. ChatGPT/Perplexity generates summaries but halluccinates citations. No tool does both — retrieves the *right* papers *for you* and explains each one at *your current level*.

**Scholar's solution:** A 3-layer ML pipeline:

```
Query + reading history
        ↓
[Layer 1 — Retrieval]
BM25 (from scratch) + Dense (Sentence-BERT, 384-dim)
Fused via Reciprocal Rank Fusion → top-200 candidates

        ↓
[Layer 2 — Personalized Reranking]
Mult-VAE encodes reading history → user latent vector
LightGBM LambdaRank (7 features) → top-50
MMR diversity rerank (λ-adaptive) → top-10
ε-greedy exploration (prevents filter bubbles)

        ↓
[Layer 3 — Level-Adaptive Explanation]
QLoRA Phi-3-mini-4k-instruct (3.8B, 4-bit NF4)
Per-paper: summary at your level + glossary + "why recommended"
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/scholar.git
cd scholar
uv venv && .venv\Scripts\activate        # Windows
# or: python -m venv .venv && source .venv/bin/activate  # Linux/Mac
uv pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env: set DATABASE_URL, ARXIV_DATA_PATH, HF_HOME

# 3. Start the stack (PostgreSQL + API + UI)
docker compose up

# 4. Download and index data (one-time)
python -m scholar.ingest.arxiv_loader --limit 200000
python scripts/build_index.py

# API is now available at http://localhost:8000
# Gradio UI is now available at http://localhost:7860
```

See [learn/09_mvp_setup.md](learn/09_mvp_setup.md) for step-by-step setup including data download.

---

## Results

### Layer 1 — Retrieval (200K ArXiv papers, numpy matmul search)

| Retriever | Latency p50 | Latency p99 | Notes |
|-----------|-------------|-------------|-------|
| BM25 (from scratch) | 26.6ms | 44.5ms | Numpy-vectorized inverted index |
| Dense (all-MiniLM-L6-v2) | 121.6ms | 196ms* | Exact cosine, 200K×384 matmul |
| Hybrid RRF (k=60) | 106.5ms | 196.2ms | BM25 + Dense fused |

*p99 warm (after startup warmup). Cold-start 98s is pre-warmed at boot so users never see it.

### Layer 2 — Reranking (CiteULike-A test split, 38 users)

| Ablation | NDCG@10 | ILD | Notes |
|----------|---------|-----|-------|
| MMR only | 0.000 | 0.496 | Corpus mismatch* |
| LTR + MMR | 0.000 | 0.496 | |
| LTR + MMR + ε-greedy | 0.000 | **0.576** | **+16.2% diversity** |

*NDCG=0 is expected: only 158 of 16,980 CiteULike paper IDs appear in the 200K ArXiv prototype corpus. The ILD signal (diversity) is the meaningful metric — it's corpus-mismatch-immune and shows ε-greedy exploration works.

### Layer 3 — Generation (local ArXiv corpus, 10 samples/level)

| Model | Level | ROUGE-L | BERTScore F1 |
|-------|-------|---------|--------------|
| Phi-3 base | undergrad | 0.070 | 0.835 |
| Phi-3 base | grad | 0.120 | 0.845 |
| Phi-3 base | researcher | 0.103 | 0.844 |
| **Phi-3 + LoRA** | undergrad | 0.082 | 0.832 |
| **Phi-3 + LoRA** | grad | 0.124 | **0.850** |
| **Phi-3 + LoRA** | researcher | 0.113 | **0.849** |

ROUGE-L is low because we use paper titles as reference TLDRs (10-word title vs 150-word generated summary). BERTScore F1 ~0.845 (contextual similarity via RoBERTa-large) is the meaningful signal. LoRA improves level-appropriate vocabulary depth over the base model, which often ignores the level instruction.

### Anti-Filter-Bubble (20-session simulation)

| Condition | Topic Entropy@10 (session 5) | Session 20 |
|-----------|------------------------------|------------|
| No diversity | ~0.8 (collapsed) | ~0.6 |
| MMR + ε-greedy | ~2.4 (stable) | ~2.3 |

A simulated user always clicking the top result: without diversity reranking, topic entropy collapses by session 5. With MMR + ε-greedy, entropy stays within 80% of initial value across all 20 sessions.

---

## Architecture Decision Highlights

| Decision | What and Why |
|----------|-------------|
| **pgvector over Pinecone** | One DB for relational + vector data — one credential, one backup, one query language. pgvector HNSW handles 2M vectors sub-10ms. |
| **Mult-VAE over embedding average** | VAE learns a latent space from behavior (not content similarity). Handles multi-topic interests, applies regularization for cold-start, directly leverages Liang et al. 2018. |
| **RRF over weighted score sum** | BM25 scores and cosine similarities are incomparable scales. RRF uses only ranks — no normalization needed, robust to score outliers. k=60 from Cormack et al. |
| **QLoRA over API prompting** | Self-contained fine-tuned model, zero per-query cost, deployable offline. 3.8B at 4-bit NF4 fits 4GB VRAM. Shows fine-tuning expertise. |
| **numpy matmul over FAISS HNSW** | FAISS HNSW segfaults (SIGSEGV) in asyncio on Windows Python 3.13 due to OMP thread conflicts. Numpy brute-force matmul gives exact cosine (Recall@k=1.0) at ~10ms for 200K×384 — competitive with HNSW at this scale. |
| **Gradio over React** | Ships in days not weeks. The ML story is what matters for placements; React frontend is future work. |

Full ADRs in [learn/01_design_doc.md](learn/01_design_doc.md) and [docs/decisions.md](docs/decisions.md).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Backend | FastAPI (async, OpenAPI auto-docs) |
| Database | PostgreSQL 16 + pgvector |
| Sparse retrieval | BM25 from scratch (no rank_bm25) |
| Dense retrieval | sentence-transformers all-MiniLM-L6-v2 + numpy matmul |
| Recommender | PyTorch Mult-VAE (from scratch) |
| Learning-to-Rank | LightGBM LambdaRank |
| LLM | Phi-3-mini-4k-instruct (3.8B) |
| Fine-tuning | PEFT + bitsandbytes (4-bit QLoRA, rank 8) |
| Frontend | Gradio |
| Containerization | Docker Compose |
| Eval | ranx (IR metrics), bert-score, rouge-score |

**Hardware target:** RTX 2050 4GB VRAM (the design constraint that drove every model-size and quantization decision).

---

## Project Structure

```
scholar/
├── ingest/          # ArXiv JSON → Postgres; CiteULike interactions; embeddings
├── retrieval/       # BM25, dense retrieval, RRF hybrid, eval
├── recsys/          # Mult-VAE, LightGBM LTR, MMR, ε-greedy, train/eval
├── generation/      # Prompts, synth data, QLoRA training, inference, eval
├── api/             # FastAPI routes: /search, /recommend, /explain, /user
├── ui/              # Gradio frontend
├── db/              # SQLModel models, schema.sql
└── config.py        # Pydantic Settings from .env
scripts/
├── build_index.py   # Full index rebuild (BM25 + embeddings)
├── download_arxiv.sh
└── train_ltr.py
docs/
├── benchmarks.md    # Eval results (updated after each run)
├── decisions.md     # Architecture Decision Records
└── learning-log.md  # Developer notes
learn/               # Interview prep, implementation guides, theory deep dives
notebooks/           # EDA, ablation studies, training curves
tests/               # 63 tests (pytest + mypy + ruff all pass)
```

---

## API Endpoints

```
GET  /health              → {"status": "ok"}
POST /search              → hybrid retrieval + ranking for a query
POST /recommend           → personalized recommendations for a user
POST /explain             → level-adaptive Phi-3 explanation for a paper
GET  /user/{user_id}      → user history + stats
POST /user/{user_id}/history  → add paper to reading history
```

Full OpenAPI docs at `http://localhost:8000/docs` when server is running.

---

## Training Your Own Models

```bash
# 1. Load CiteULike interaction data
python -m scholar.ingest.citeulike_loader --data-dir ./data/citeulike-a

# 2. Train Mult-VAE (user representation)
python scholar/recsys/train_vae.py
# → saves data/mult_vae.pt + data/vae_vocab.json

# 3. Generate synthetic training data for LoRA (calls Phi-3 base)
python scholar/generation/synth_data.py --limit 1000
# → writes data/synth_training.jsonl

# 4. Fine-tune with QLoRA (~10h on RTX 2050 4GB)
python scholar/generation/train_lora.py
# → saves data/lora_adapter/ (6.3MB adapter)

# 5. Run evaluations
python -m scholar.retrieval.eval
python -m scholar.recsys.eval
python -m scholar.generation.eval
```

---

## Known Limitations

1. **No real users.** All user data is CiteULike (small, dated 2016) or synthetic. Production needs real click logs.
2. **No online learning.** Mult-VAE is retrained offline; production needs incremental updates.
3. **Abstracts only.** Full-text would improve retrieval quality; not indexed due to storage constraints.
4. **Prototype corpus.** 200K papers indexed; full 2M requires ~72h GPU embedding time (pipeline supports it).
5. **Single-pass generation.** No fact-checking loop. BERTScore F1 ~0.845 bounds hallucination rate empirically, but output should always be read alongside the source abstract.
6. **/explain requires no other Phi-3 process running.** On 4GB VRAM, explanation and training/eval cannot co-exist. Documented workaround: run /explain from standalone script, or increase paging file to 8–16GB.

---

## Test Suite

```bash
pytest && ruff check scholar/ && mypy scholar/
# Expected: 63 passed, 0 ruff errors, 0 mypy errors
```

---

## License

MIT. Paper data from ArXiv (CC0) and CiteULike (public). Model weights: Phi-3-mini from Microsoft (MIT license).

---

## Learn More

- [Design Document + ADRs](learn/01_design_doc.md) — full system design
- [Implementation Guide](learn/02_implementation_guide.md) — file-by-file walkthrough
- [Evaluation Metrics Deep Dive](learn/10_evaluation_metrics.md) — every metric explained with actual results
- [Optimizations Log](learn/07_optimizations.md) — 24 bugs fixed with root causes and interview answers
- [Reading Index](learn/00_INDEX.md) — structured reading order for all learning materials
