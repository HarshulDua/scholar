# Architecture Decision Records

## ADR-001: BM25 from Scratch vs. rank_bm25

**Status:** Accepted

**Context:**
We need a BM25 implementation for sparse retrieval. The `rank_bm25` library is the standard Python option, but using it creates an external dependency that abstracts away the scoring implementation.

**Decision:**
Implement BM25 from scratch using an inverted index with natural-log IDF and standard TF normalization.

**Consequences:**
- We have full control over the scoring formula and can tune parameters (k1, b) without library constraints.
- Persistence is straightforward (pickle the index object).
- Slightly more code to maintain, but BM25 is well-understood and unlikely to change.
- No dependency on `rank_bm25` — removes a potential incompatibility with newer Python versions.

---

## ADR-002: FAISS (CPU/HNSW) vs. pgvector for ANN Search

**Status:** Accepted

**Context:**
We need approximate nearest neighbor search over ~50K paper embeddings at query time. Two candidates: pgvector (Postgres extension with HNSW) and FAISS (in-process, CPU).

**Decision:**
Use FAISS as the primary ANN index loaded into API process memory. Store embeddings in pgvector as a backup and for exact-match SQL queries. Both indices are built from the same sentence-BERT embeddings.

**Consequences:**
- FAISS HNSW queries are sub-millisecond in-process with no network round-trip.
- pgvector remains available for exact similarity queries and joins with metadata.
- We carry both indices on disk (~200MB combined for 50K papers at 384 dims), which is acceptable.
- FAISS must be loaded at startup and kept in `app.state` — a restart reloads from disk, which is fast.

---

## ADR-003: Phi-3-mini with 4-bit Quantization vs. GPT-4 API

**Status:** Accepted

**Context:**
We need a language model to generate paper summaries, glossaries, and relevance explanations. Options: call a hosted API (OpenAI, Anthropic) or run a local quantized model.

**Decision:**
Run `microsoft/Phi-3-mini-4k-instruct` locally with 4-bit bitsandbytes quantization (NF4, fp16 compute). No LoRA adapter in MVP — base model + prompting only.

**Consequences:**
- No per-request API costs. Fully offline capable.
- Requires a GPU with ~4GB VRAM for the quantized model. Falls back to CPU (slow).
- Generation latency is ~2–5s per paper on GPU vs. <1s for a hosted API.
- Quality is lower than GPT-4 but sufficient for structured extraction (summary/glossary/why).
- LoRA fine-tuning path is preserved for post-MVP.

---

## ADR-004: Mult-VAE for Collaborative Filtering vs. Matrix Factorization

**Status:** Accepted

**Context:**
We need a user embedding model for personalized recommendations. Options include classical MF (SVD, ALS), neural MF, or variational approaches.

**Decision:**
Implement Mult-VAE (Variational Autoencoder with multinomial likelihood) from scratch in PyTorch.

**Consequences:**
- Mult-VAE handles implicit feedback (click/view) naturally via a multinomial reconstruction loss.
- The latent user embedding can be extracted at inference time as the encoder mean (no sampling).
- Cold-start fallback: if the model is untrained (near-init weights), we fall back to averaging sentence-BERT embeddings of history papers for the user vector.
- Training infrastructure is scaffolded but not invoked in MVP (no labeled interaction data yet).

---

## ADR-005: Reciprocal Rank Fusion for Hybrid Search

**Status:** Accepted

**Context:**
Combining BM25 and dense retrieval scores requires score normalization. Options: linear combination (requires score calibration), learned reranker (requires training data), or RRF (parameter-free).

**Decision:**
Use Reciprocal Rank Fusion (RRF) with k=60 to merge BM25 and dense ranked lists.

**Consequences:**
- RRF is robust to score distribution differences between BM25 and cosine similarity.
- No training data required — works out of the box.
- The k=60 constant was empirically validated in the original Cormack et al. (2009) paper.
- LightGBM LTR reranker is wired but not trained in MVP — it will replace RRF post-MVP once click data accumulates.

---

## ADR-006: MMR for Diversity vs. Pure Relevance Ranking

**Status:** Accepted

**Context:**
Users browsing paper discovery interfaces often benefit from diverse results rather than near-duplicate highly similar papers. Standard relevance ranking can return multiple papers on the same narrow topic.

**Decision:**
Apply Maximum Marginal Relevance (MMR) as a post-retrieval reranking step on the top-K candidates (K=200 after RRF), with an adaptive lambda controlled by user history diversity.

**Consequences:**
- MMR is O(K * selected) which is fast for K=200, top_k=10.
- Adaptive lambda personalizes the diversity-relevance tradeoff: specialist users (tight history cluster) get lambda=0.3 (more diversity); generalist users get lambda=0.7 (more relevance).
- Anonymous users skip MMR and get straight RRF ranking.
- The lambda can be overridden per-request via the `diversity_lambda` API parameter.
