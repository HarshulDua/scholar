# Scholar — Design Document

> A personalized research paper discovery engine with hybrid retrieval, learned user representations, and level-adaptive AI-generated explanations.

---

## 1. Problem Statement

**The user problem.** A student or early researcher wants to learn about a topic (e.g., "diffusion models for medical imaging"). Today their options are:

- Google Scholar — surfaces a chronological list of papers, no understanding of user level or current interests
- Semantic Scholar / Connected Papers — strong on citation graphs, weak on personalization and explanation
- ArXiv search — pure keyword match, results in jargon-heavy abstracts
- ChatGPT/Perplexity — generates summaries but with hallucinated citations and no real retrieval grounding

The gap: **no system both (a) retrieves the right papers for a user given their history and stated interests, and (b) explains each paper at the user's current level.**

**What Scholar does.** Given a user query and the user's reading history:

1. Retrieves a relevant candidate set from ~2M ArXiv papers using hybrid BM25 + dense retrieval
2. Reranks candidates using a learned model that incorporates the user's interests (encoded by a Mult-VAE trained on reading histories)
3. Applies diversity-aware reranking (MMR) and exploration injection to avoid filter bubbles
4. For each top result, generates a level-adaptive explanation using a LoRA-fine-tuned Phi-3-mini

The user gets: a ranked list of papers, each with a short personalized summary, a "why this was recommended" justification, and a glossary of key concepts.

---

## 2. Non-Goals

To prevent scope creep, the following are explicitly out of scope:

- Full-text search (we use abstracts + titles only; full PDFs are linked, not indexed)
- Citation graph traversal beyond first-hop (no PageRank, no community detection)
- User authentication / production multi-user backend (single-user Gradio demo)
- Mobile app, browser extension
- Real-time training updates (the model is retrained offline)
- Multilingual support (English-only; deferred)
- Question answering over paper contents (only summarization and recommendation)

---

## 3. Target Users

- **Primary:** Final-year undergrads / Masters students exploring research areas
- **Secondary:** Working engineers picking up a new technical domain
- **Personas:** "Curious explorer" (low history, wants exploration), "Focused researcher" (high history, wants depth), "Switching domains" (high history but in irrelevant area, needs fresh start)

The system must serve all three personas. This is what motivates the exploration / diversity / fresh-start design decisions in §7.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Gradio UI)                         │
│   Query: "diffusion models for healthcare"                       │
│   History: [paper_id_1, paper_id_2, ...]                         │
│   Level: undergrad / grad / researcher                           │
│   Controls: Fresh Start | Diversity Slider | Explore More       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  LAYER 1: HYBRID RETRIEVAL                  │
        │  • BM25 over abstracts (rank_bm25)          │
        │  • Dense retrieval (Sentence-BERT + FAISS)  │
        │  • Score fusion (Reciprocal Rank Fusion)    │
        │  Output: top-200 candidates                 │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │  LAYER 2: PERSONALIZED RERANKING            │
        │  • Mult-VAE encodes user history → z_user   │
        │  • LightGBM LambdaRank reranker             │
        │    features: query-doc sim, user-doc sim,   │
        │    citation count, recency, topic match     │
        │  • MMR diversity rerank (tunable λ)         │
        │  • Exploration injection (epsilon-greedy)   │
        │  Output: top-10 papers                      │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │  LAYER 3: GENERATIVE EXPLANATION            │
        │  • Phi-3-mini-4k-instruct (QLoRA-tuned)     │
        │  • Per paper: summary @ user_level          │
        │  • Glossary of key concepts                 │
        │  • "Why this was recommended" reasoning     │
        │  Output: rendered cards                     │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │  POSTGRES (pgvector)                        │
        │  papers, embeddings, user_history,          │
        │  click_logs, evaluation_qrels               │
        └────────────────────────────────────────────┘
```

---

## 5. Tech Stack

| Layer | Technology | Why this choice |
|-------|-----------|-----------------|
| Language | Python 3.11 | Single language across all 3 layers |
| Backend | FastAPI | Async, type-safe, OpenAPI auto-docs, lighter than Django for this |
| Database | PostgreSQL 16 + pgvector | One DB for both relational and vector data — avoids the Postgres+Pinecone duplication trap |
| Sparse retrieval | rank_bm25 (then custom) | Start with library, swap to your own implementation in Phase 1 |
| Dense retrieval | sentence-transformers (`all-MiniLM-L6-v2`) | 80MB model, fast on CPU, strong baseline |
| Vector index | FAISS (CPU index) | Free, fast, no ops burden |
| Recsys core | PyTorch (Mult-VAE from scratch) | The interview-defendable piece |
| LTR | LightGBM with LambdaRank | Production-standard, fast training, interpretable feature importances |
| LLM | Phi-3-mini-4k-instruct (3.8B) | Fits RTX 2080 with QLoRA, strong instruction following |
| Fine-tuning | PEFT + bitsandbytes (4-bit QLoRA) | The only way to fit 3.8B on 8GB |
| Frontend | Gradio | Fastest path to a demo; replace with React only if you have time |
| Containerization | Docker Compose | One `docker compose up` brings up Postgres + API + UI |
| Eval / metrics | scikit-learn, ranx | ranx is a small library specifically for IR metrics (NDCG, MRR, MAP) |

**Hardware reality check (RTX 2080, 8GB VRAM):**
- BM25 + LightGBM: CPU-only, fine.
- Sentence-BERT inference: comfortable, ~1ms/abstract.
- Mult-VAE training: tiny, ~5 min/epoch on full dataset.
- Phi-3-mini QLoRA: ~6.5GB VRAM with batch size 1, gradient checkpointing on, sequence length 1024. Tight but works. If OOM, fall back to Phi-1.5 (1.3B) or TinyLlama 1.1B.

---

## 6. Data Sources

| Dataset | Use | Size | License |
|---------|-----|------|---------|
| ArXiv metadata (Kaggle: `Cornell-University/arxiv`) | Paper corpus (title, abstract, authors, categories) | ~2M papers, ~4GB JSON | CC0 |
| Semantic Scholar Open Research Corpus (S2ORC) | Citation counts, references | Optional, ~100GB if full | ODC-BY |
| CiteULike data dump | User reading histories (real, anonymized) | ~8K users, ~150K interactions | Public |
| SciTLDR | Paper → 1-sentence summary pairs for LoRA training | ~5K pairs | Apache 2.0 |
| NFCorpus / TREC-COVID | Held-out IR eval with relevance labels | ~3K queries | Public |

**For the LoRA fine-tuning corpus**, the plan is:
- 5K SciTLDR pairs as base
- Synthesize ~5K more with a teacher model (call GPT-4 or Claude API once on ArXiv abstracts to generate undergrad/grad/researcher-level summaries) — costs ~$10–20
- Total ~10K training examples, plenty for QLoRA on a 3.8B model

---

## 7. Key Design Decisions (ADRs)

Each decision below is an Architecture Decision Record — the format you'll keep in `docs/decisions.md` and reuse for interview answers.

### ADR-001: PostgreSQL + pgvector over Postgres + dedicated vector DB

**Context.** We need to store paper metadata, user history, click logs, AND vector embeddings. Options: (a) Postgres + Pinecone/Qdrant, (b) Postgres + pgvector, (c) all-in-one vector DB.

**Decision.** Postgres + pgvector.

**Rationale.** One database = one set of credentials, one backup story, one query language. pgvector's HNSW index gives sub-10ms ANN search up to ~10M vectors, which is comfortably above our 2M scale. We trade some peak performance for operational simplicity. At >10M vectors or >1000 QPS we'd revisit.

### ADR-002: Mult-VAE for user representation over averaged embeddings

**Context.** We need to represent a user's interests for personalized reranking. Options: (a) average the embeddings of papers they've read, (b) train a Mult-VAE (Liang et al. 2018), (c) train a sequence model (BERT4Rec, SASRec).

**Decision.** Mult-VAE.

**Rationale.** (a) loses combinatorial interests and treats all clicks equally. (c) needs much more data than we have for our small synthetic-user corpus. (b) gives a regularized, continuous latent space, handles cold-start gracefully via the encoder, and the VAE machinery directly leverages prior work from CS F437. Mult-VAE is a published, deployed-in-production technique — defendable in interviews.

**Consequence.** We accept that Mult-VAE is order-insensitive (it sees clicks as a bag, not a sequence). For our use case, where interests evolve over weeks not minutes, this is acceptable. We mitigate with time-decay weighting.

### ADR-003: Hybrid BM25 + dense retrieval over either alone

**Context.** Pure BM25 misses semantic synonyms ("LLM" vs "large language model"). Pure dense retrieval misses exact-term matches (paper titles, author names, equation symbols). 

**Decision.** Hybrid retrieval with Reciprocal Rank Fusion (RRF), k=60.

**Rationale.** RRF is parameter-light, doesn't need score normalization, and consistently beats either retriever alone in published benchmarks. The cost is one extra dense pass — negligible at our scale.

### ADR-004: QLoRA fine-tuning over prompted LLM API

**Context.** The generation layer can be (a) prompt engineering on a large API model (GPT-4, Claude), (b) full fine-tuning of an open model, (c) LoRA/QLoRA on an open model.

**Decision.** QLoRA on Phi-3-mini-4k-instruct.

**Rationale.** (a) is fine but signals less ML depth; we want to show fine-tuning expertise. (b) doesn't fit on an RTX 2080. (c) gives us a defendable training story, fits hardware, and produces a self-contained model we can ship. Cost: ~10 hours of training time over the project, plus the small data-generation cost.

### ADR-005: MMR diversity reranking with adaptive λ

**Context.** Pure relevance ranking produces filter bubbles. Pure diversity loses personalization.

**Decision.** Apply MMR rerank after LTR, with λ adapted to the user's history clustering coefficient: tight cluster (specialized user) → λ skews toward diversity; spread-out history → λ skews toward relevance.

**Rationale.** A specialized user benefits more from exposure to adjacent areas. A generalist benefits more from relevance. The adaptive λ codifies this. Tunable in the UI for transparency.

### ADR-006: Gradio over React frontend (for now)

**Context.** A real product needs a polished web UI. Project scope doesn't allow building both backend ML and a React app in 6 weeks.

**Decision.** Gradio for the demo. Document the React frontend as future work.

**Rationale.** Gradio gets us shipping in days, not weeks. The ML story is what matters for placements; UI polish is a bonus we can add later if time permits.

---

## 8. Evaluation Methodology

This is the section that separates a strong project from a forgettable one. Three layers, three evaluation strategies:

### 8.1 Retrieval (Layer 1)

- **Dataset:** Held-out NFCorpus and a self-built ArXiv eval set (200 queries with hand-labeled relevance)
- **Metrics:** NDCG@10, MRR, Recall@100, MAP
- **Baselines compared:** BM25 alone, dense alone, hybrid with RRF, hybrid with weighted sum
- **Expected outcome:** Hybrid beats either alone by 5-15% on NDCG@10

### 8.2 Recommendation (Layer 2)

- **Dataset:** CiteULike (real users, real interactions) — 80/10/10 train/val/test split per user
- **Metrics:** NDCG@10, Hit Rate@10, MRR, Coverage (fraction of catalog ever recommended), Gini coefficient (popularity bias)
- **Baselines compared:** popular (no personalization), Mult-VAE alone, Mult-VAE + LTR
- **Anti-bubble metrics:** Topic Entropy@10 (Shannon entropy of topic distribution in recommendations), Intra-List Diversity (1 - avg cosine sim between recommended papers)
- **Expected outcome:** Mult-VAE + LTR beats popular by 30%+ on NDCG. Adding MMR trades ~5% NDCG for 40%+ improvement in ILD.

### 8.3 Generation (Layer 3)

- **Dataset:** SciTLDR test split, plus a small (~50) hand-evaluated set
- **Automatic metrics:** ROUGE-L, BERTScore, summarization-specific (avg length, type-token ratio)
- **Human eval:** rate 50 outputs on (a) factual accuracy, (b) level appropriateness, (c) glossary usefulness — 5-point Likert
- **Baselines compared:** base Phi-3-mini (no LoRA), LoRA-tuned Phi-3-mini, GPT-3.5 prompted (ceiling)
- **Expected outcome:** LoRA closes 40-60% of the gap between base Phi-3 and GPT-3.5

### 8.4 End-to-end qualitative eval

- **Filter-bubble simulation:** simulate a user who clicks only diffusion papers for 20 sessions. Measure topic entropy over time. Show that MMR + exploration prevents collapse.
- **Cold-start case studies:** show recommendations for users with 1, 5, 20, 100 clicks. Demonstrate graceful degradation.
- **Fresh-start case study:** show before/after for a user resetting from a diffusion focus to a systems focus.

---

## 9. Known Limitations (be honest about these)

1. **No real users.** All user data is either CiteULike (small, dated) or synthetic. A production system would need real click logs.
2. **No online learning.** The Mult-VAE is retrained offline. A real system needs incremental updates.
3. **English only.** Multilingual deferred.
4. **Abstracts only.** Full-text would give better retrieval but blows up storage and indexing.
5. **Single-pass generation.** No fact-checking loop. Generated summaries can occasionally drift from the source paper. Mitigated by (a) including the source abstract in the output, (b) human eval to bound the error rate.
6. **Cold-start papers, not just users.** Very new papers with no citations have weak signal. A real system would use author/venue priors.

Owning these limitations in your README and interviews is a strength signal, not a weakness.

---

## 10. Success Criteria (what "done" looks like in 6 weeks)

- ✅ Working end-to-end pipeline: query in, ranked + explained papers out
- ✅ Retrieval AUROC/NDCG@10 ≥ 0.30 on the eval set (this is a competitive number)
- ✅ Mult-VAE outperforms "popular" baseline by ≥20% NDCG@10 on CiteULike
- ✅ LoRA-tuned generator beats base model on ROUGE-L by ≥10%
- ✅ Filter-bubble simulation showing MMR maintains topic entropy
- ✅ Gradio demo deployable with `docker compose up`
- ✅ README with screenshots, GIFs, deployed demo link
- ✅ 2,000-word blog post explaining the design
- ✅ Architecture Decision Records committed to repo
- ✅ Interview prep doc with rehearsed answers
