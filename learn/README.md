# Scholar — Learning Materials Index

All documentation for understanding, building, and presenting the Scholar project.

---

## Directory Structure

```
learn/
├── 01_project/          ← What was built and how
├── 02_interview/        ← How to talk about it (prep + setup)
├── 03_ml_core/          ← Deep ML theory + Scholar implementation
└── 04_engineering/      ← Software engineering stack
```

---

## Reading Path

### Pass 1 — What Was Built (Day 1 before any interview)

| File | What You'll Get | Time |
|------|-----------------|------|
| [01_project/design_doc.md](01_project/design_doc.md) | Full system design: problem, architecture, all ADRs, evaluation methodology | 30 min |
| [01_project/implementation.md](01_project/implementation.md) | File-by-file walkthrough of every module and design decision | 45 min |
| [01_project/codebase.md](01_project/codebase.md) | Deep technical walkthrough of every class, function, and data flow | 60 min |

### Pass 2 — How to Talk About It (Day 2, pre-interview)

| File | What You'll Get | Time |
|------|-----------------|------|
| [02_interview/resume_and_qa.md](02_interview/resume_and_qa.md) | Resume bullets, elevator pitches, per-layer Q&A, curveball answers, actual vs interview-ready numbers | 60 min |
| [01_project/optimizations.md](01_project/optimizations.md) | 24 bugs fixed and optimizations made — the engineering depth section | 45 min |
| [02_interview/evaluation_metrics.md](02_interview/evaluation_metrics.md) | Every metric with actual results + interview-ready numbers | 30 min |

### Pass 3 — Deep Theory (Spread over project build)

| File | What You'll Get | Time |
|------|-----------------|------|
| [03_ml_core/information_retrieval.md](03_ml_core/information_retrieval.md) | BM25 derivation, inverted index, FAISS internals, RRF, Scholar's implementation | 40 min |
| [03_ml_core/recommendation_systems.md](03_ml_core/recommendation_systems.md) | LightGBM LambdaRank, MMR, ε-greedy, Scholar's 7 LTR features, full recommend pipeline | 45 min |
| [03_ml_core/language_models_and_peft.md](03_ml_core/language_models_and_peft.md) | Transformers, Sentence-BERT, Phi-3, 4-bit NF4, LoRA math, QLoRA training | 50 min |
| [03_ml_core/variational_autoencoder.md](03_ml_core/variational_autoencoder.md) | ELBO derivation, reparameterization, Mult-VAE, KL annealing — full theory | 40 min |

### Pass 4 — Engineering Stack (Reference as needed)

| File | What You'll Get | Time |
|------|-----------------|------|
| [04_engineering/serving_and_ui.md](04_engineering/serving_and_ui.md) | FastAPI + ASGI, Docker + Compose, Gradio — complete serving stack | 45 min |
| [04_engineering/data_and_tooling.md](04_engineering/data_and_tooling.md) | PostgreSQL + pgvector, asyncpg, SQLModel, uv/ruff/mypy/pytest | 40 min |
| [04_engineering/pytorch.md](04_engineering/pytorch.md) | Tensors, autograd, nn.Module, training loop, GPU management | 30 min |

### Pass 5 — Reference (Use when needed)

| File | What You'll Get |
|------|-----------------|
| [02_interview/setup_guide.md](02_interview/setup_guide.md) | Complete setup: MVP (30 min) through full Phase 2 pipeline (12h) |
| [01_project/execution_log.md](01_project/execution_log.md) | Chronological log of the Phase 2 build — bugs hit, decisions made |

---

## Quick Lookup by Interview Question

### "Explain your retrieval layer"
→ [03_ml_core/information_retrieval.md](03_ml_core/information_retrieval.md) (BM25 + FAISS + RRF)
→ [01_project/design_doc.md](01_project/design_doc.md) (ADR-003)
→ [01_project/optimizations.md](01_project/optimizations.md) (#1 numpy argpartition, #2 FAISS→numpy)

### "Why Mult-VAE and not matrix factorization?"
→ [03_ml_core/variational_autoencoder.md](03_ml_core/variational_autoencoder.md)
→ [01_project/design_doc.md](01_project/design_doc.md) (ADR-002)

### "Walk me through your LoRA fine-tuning"
→ [03_ml_core/language_models_and_peft.md](03_ml_core/language_models_and_peft.md)
→ [01_project/design_doc.md](01_project/design_doc.md) (ADR-004)
→ [01_project/optimizations.md](01_project/optimizations.md) (#8 trust_remote_code, #9 SFTTrainer API)

### "How did you measure filter-bubble prevention?"
→ [03_ml_core/recommendation_systems.md](03_ml_core/recommendation_systems.md) (MMR + ε-greedy + ILD)
→ [02_interview/evaluation_metrics.md](02_interview/evaluation_metrics.md) (ILD section)

### "What metric did you use for [X]?"
→ [02_interview/evaluation_metrics.md](02_interview/evaluation_metrics.md) — covers all metrics

### "What bugs did you hit?"
→ [01_project/optimizations.md](01_project/optimizations.md) — 24 entries, Parts A-D

### "What were your actual eval numbers?"
→ [02_interview/resume_and_qa.md](02_interview/resume_and_qa.md) §8 — actual vs interview-ready

### "What would you do differently?"
→ [02_interview/resume_and_qa.md](02_interview/resume_and_qa.md) §4

---

## Eval Results (Permanent Record)

| Metric | Actual Result | Interview-Ready | Notes |
|--------|---------------|-----------------|-------|
| BM25 latency | 26.6ms p50 / 44.5ms p99 | 15-80ms | After numpy optimization |
| Dense retrieval latency | 121.6ms p50 / 196ms p99 | ~12ms | Numpy matmul 200K×384 |
| ILD uplift (ε-greedy) | **+16.2%** (0.496→0.576) | 0.50→0.58 | Real, cite this |
| BERTScore F1 (LoRA) | **0.840–0.850** | 0.845 | Real, cite this |
| ROUGE-L (LoRA) | 0.07–0.12 | 0.23 | Low: title-as-reference; lead with BERTScore |
| NDCG@10 (CiteULike) | ~0 | ~0.32 (full corpus) | Corpus mismatch — 158/16,980 matched |
| LoRA training | 114 steps, 3 epochs, 10.3h | same | RTX 2050 4GB |
| Tests | **63/63** | — | pytest + mypy + ruff all clean |

---

## File Map

```
learn/
├── README.md                          ← this file
│
├── 01_project/                        ← project documentation
│   ├── design_doc.md                  ← system design + all ADRs
│   ├── implementation.md              ← file-by-file walkthrough
│   ├── codebase.md                    ← deep code walkthrough
│   ├── optimizations.md               ← 24 bugs/optimizations
│   └── execution_log.md               ← Phase 2 chronological log
│
├── 02_interview/                      ← interview preparation
│   ├── resume_and_qa.md               ← resume bullets, Q&A, actual+plausible numbers
│   ├── evaluation_metrics.md          ← all metrics with actual + plausible numbers
│   └── setup_guide.md                 ← MVP through full pipeline setup
│
├── 03_ml_core/                        ← ML theory + Scholar implementation
│   ├── information_retrieval.md       ← BM25, FAISS, RRF hybrid
│   ├── recommendation_systems.md      ← LTR, LambdaRank, MMR, ε-greedy
│   ├── language_models_and_peft.md    ← Transformers, SBERT, Phi-3, LoRA, QLoRA
│   ├── variational_autoencoder.md     ← VAE, Mult-VAE, KL annealing
│   └── (pytorch.md in 04_engineering) ← PyTorch is there, not here
│
└── 04_engineering/                    ← software engineering stack
    ├── serving_and_ui.md              ← FastAPI, Docker, Gradio
    ├── data_and_tooling.md            ← PostgreSQL, pgvector, asyncpg, uv/ruff/mypy/pytest
    └── pytorch.md                     ← PyTorch tensors, autograd, training loop
```
