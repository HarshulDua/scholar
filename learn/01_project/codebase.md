# Scholar — Complete Codebase Deep Dive

**The full picture: every file explained, every decision reasoned, every algorithm connected.**

This document is the single source of truth for understanding Scholar end-to-end. It covers:
- The architecture and why each piece exists
- Every key file, line by line, with the *why* not just the *what*
- How the three layers (retrieval → reranking → generation) connect
- Data flow from a raw user query to a rendered explanation
- Design decisions, trade-offs, and constraints

---

## Table of Contents

1. [The Problem Scholar Solves](#1-the-problem-scholar-solves)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Model and Schema](#3-data-model-and-schema)
4. [Layer 1: Hybrid Retrieval](#4-layer-1-hybrid-retrieval)
5. [Layer 2: Personalized Reranking](#5-layer-2-personalized-reranking)
6. [Layer 3: Explanation Generation](#6-layer-3-explanation-generation)
7. [API Layer](#7-api-layer)
8. [Data Ingestion Pipeline](#8-data-ingestion-pipeline)
9. [Training Scripts](#9-training-scripts)
10. [UI Layer](#10-ui-layer)
11. [Configuration and Infrastructure](#11-configuration-and-infrastructure)
12. [Complete Request Trace](#12-complete-request-trace)
13. [System Design Deep Dive](#13-system-design-deep-dive)
14. [API Schemas Deep Dive](#14-api-schemas-deep-dive--scholarapischemaspy)
15. [Ingestion Pipeline Complete Deep Dive](#15-ingestion-pipeline--complete-deep-dive)
16. [Training Pipelines Complete Deep Dive](#16-training-pipelines--complete-deep-dive)
17. [Evaluation Framework](#17-evaluation-framework)
18. [Deep Theory Appendix](#18-deep-theory-appendix)
19. [Performance Characteristics](#19-performance-characteristics-and-bottlenecks)
20. [The Scholar Mental Model](#20-connecting-everything-the-scholar-mental-model)

---

## 1. The Problem Scholar Solves

### The Cold-Start Information Overload Problem

ArXiv publishes ~500 papers per day across 40+ categories. A machine learning researcher who follows cs.LG, cs.AI, and stat.ML faces ~100 new papers daily. They can read maybe 5. How do you find the right 5?

The naive approach — keyword search on title — misses papers that use different terminology for the same concept. Full-text search helps but returns hundreds of results. And neither approach personalizes to your specific research interests.

Scholar's answer:
1. **Recall broadly** with hybrid lexical+semantic search → 200 candidates
2. **Rank precisely** with your reading history → 50 candidates
3. **Diversify intelligently** to avoid showing 10 papers on the same sub-topic → 10 final papers
4. **Explain adaptively** at the right technical depth for your background

### Why This Is Hard

Each layer has its own challenge:

| Layer | Challenge |
|-------|-----------|
| Retrieval | BM25 misses synonyms; dense retrieval misses exact terms; need both |
| Reranking | User preferences are implicit (clicks); cold-start problem; filter bubbles |
| Generation | 4GB VRAM budget; different users need different explanation depths |

The constraints from `CLAUDE.md` are not arbitrary — each was chosen to address a specific challenge:
- **BM25 from scratch**: proves you understand the algorithm, not just the library
- **Mult-VAE**: implicit collaborative filtering handles the "no ratings" problem  
- **RRF k=60**: rank-based fusion is more robust than score-based fusion across different scoring scales
- **MMR on top-50 only**: O(K²) is fine for 50 items; catastrophic for 200
- **QLoRA 4-bit NF4**: only way to run a 3.8B model on an RTX 2050

---

## 2. Architecture Overview

```
                        ┌─────────────────────────────────────────────────────┐
                        │                    Scholar System                    │
                        │                                                      │
Query + user_id         │  ┌────────────────────────────────────────────────┐ │
──────────────────►     │  │              Layer 1: Retrieval                 │ │
                        │  │                                                  │ │
                        │  │  BM25 (inverted index, exact)                   │ │
                        │  │    +                                             │ │
                        │  │  FAISS (HNSW, semantic)                         │ │
                        │  │    ↓                                             │ │
                        │  │  RRF (k=60, rank fusion) → 200 candidates       │ │
                        │  └────────────────────┬───────────────────────────┘ │
                        │                       │ 200 papers                  │
                        │  ┌────────────────────▼───────────────────────────┐ │
                        │  │              Layer 2: Reranking                 │ │
                        │  │                                                  │ │
                        │  │  Mult-VAE → user embedding                      │ │
                        │  │    +                                             │ │
                        │  │  LightGBM LambdaRank (7 features) → 50 papers  │ │
                        │  │    +                                             │ │
                        │  │  MMR (adaptive λ, diversity) → 10 papers        │ │
                        │  │    +                                             │ │
                        │  │  ε-greedy (10% exploration) → 10 final papers   │ │
                        │  └────────────────────┬───────────────────────────┘ │
                        │                       │ 10 papers                   │
                        │  ┌────────────────────▼───────────────────────────┐ │
                        │  │              Layer 3: Generation                │ │
                        │  │                                                  │ │
                        │  │  QLoRA Phi-3-mini-4k-instruct                  │ │
                        │  │  (4-bit NF4, LoRA r=8/α=16)                   │ │
                        │  │  Level-adaptive prompt → SUMMARY + GLOSSARY    │ │
                        │  └────────────────────────────────────────────────┘ │
                        └─────────────────────────────────────────────────────┘
```

### Module map

```
scholar/
├── config.py               # pydantic-settings: all config from .env
├── db/
│   ├── models.py           # SQLModel: Paper, User, UserHistory, ClickLog
│   └── schema.sql          # CREATE TABLE statements (applied at startup)
├── ingest/
│   ├── arxiv_loader.py     # downloads + parses ArXiv bulk JSON → DB
│   ├── citeulike_loader.py # loads CiteULike interactions → user_history
│   └── embeddings.py       # batch-encodes papers → pgvector + embeddings.npy
├── retrieval/
│   ├── bm25.py             # BM25 from scratch: InvertedIndex + BM25 class
│   ├── dense.py            # DenseRetriever: FAISS HNSW, sentence-BERT encoding
│   ├── hybrid.py           # HybridRetriever: RRF fusion of BM25 + dense
│   └── eval.py             # NFCorpus evaluation: NDCG@10, MRR, MAP
├── recsys/
│   ├── mult_vae.py         # MultVAE, MultVAELoss, get_user_embedding
│   ├── train_vae.py        # full training loop with KL annealing
│   ├── ltr.py              # LTRModel, extract_features (7 features)
│   ├── mmr.py              # mmr_rerank, adaptive_lambda
│   ├── exploration.py      # epsilon_greedy_inject
│   └── eval.py             # CiteULike evaluation: NDCG@K, Precision@K, ILD
├── generation/
│   ├── prompts.py          # UNDERGRAD/GRAD/RESEARCHER prompt templates
│   ├── inference.py        # Phi3Generator singleton: 4-bit loading + generate()
│   ├── synth_data.py       # synthetic training data generation
│   ├── train_lora.py       # QLoRA fine-tuning with SFTTrainer
│   └── eval.py             # SciTLDR evaluation
├── api/
│   ├── main.py             # FastAPI app: lifespan, middleware, router registration
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── search.py       # POST /search: hybrid retrieval + MMR
│       ├── recommend.py    # GET /recommend/{user_id}: full personalization pipeline
│       ├── explain.py      # POST /explain/{paper_id}: Phi-3 generation
│       └── user.py         # POST /user, GET /user/{id}, POST /user/{id}/history
└── ui/
    └── gradio_app.py       # Gradio Blocks UI: search, recommend, render HTML cards
```

---

## 3. Data Model and Schema

```python
# scholar/db/models.py
```

### Paper

```python
class Paper(SQLModel, table=True):
    __tablename__ = "papers"
    
    id: str = Field(primary_key=True)
    # ArXiv ID: "2301.00001" — natural primary key, also the URL slug
    # Using a string PK avoids a useless integer surrogate key
    
    title: str
    abstract: str
    authors: str          # stored as a single string, not TEXT[]
    # Why string not array: SQLModel + asyncpg handle TEXT[] awkwardly.
    # Application code splits on commas/semicolons when needed.
    
    categories: str       # "cs.LG cs.AI stat.ML" — space-separated
    # Why string not array: same reason. LTR's _category_overlap() splits on spaces.
    
    update_date: Optional[date] = Field(default=None)
    citation_count: int = Field(default=0)
    # citation_count: used as LTR feature + quality filter in ε-greedy
    
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(384)),
    )
    # 384 dimensions = all-MiniLM-L6-v2 output size
    # sa_column=Column(Vector(384)): pgvector type can't be expressed as a native SQLModel type
    # NULL embedding: papers inserted before the embedding job runs
```

### User + UserHistory + ClickLog

```python
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: str = Field(primary_key=True)  # UUID string from uuid.uuid4()
    created_at: Optional[datetime] = Field(default=None)

class UserHistory(SQLModel, table=True):
    __tablename__ = "user_history"
    user_id: str = Field(foreign_key="users.id", primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    clicked_at: Optional[datetime] = Field(default=None)
    # Composite PK (user_id, paper_id): inserting the same click twice is a no-op
    # (INSERT ... ON CONFLICT DO NOTHING in production, or explicit check in add_to_history)

class ClickLog(SQLModel, table=True):
    __tablename__ = "click_logs"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None)   # nullable: anonymous users
    paper_id: Optional[str] = Field(default=None)
    query: Optional[str] = Field(default=None)
    rank_position: Optional[int] = Field(default=None)
    clicked_at: Optional[datetime] = Field(default=None)
    # No FK constraints: logging should never fail due to referential integrity
    # If a paper gets deleted, its click logs should still exist for analytics
```

---

## 4. Layer 1: Hybrid Retrieval

### 4.1 BM25 — `scholar/retrieval/bm25.py`

BM25 (Best Match 25) is a probabilistic retrieval model based on the Robertson-Spärck Jones relevance model. The "25" refers to it being the 25th refinement in a series of experiments.

**The formula:**

For a query Q with terms q₁...qₙ and a document D:

```
BM25(D, Q) = Σᵢ IDF(qᵢ) × TF(qᵢ, D, avgdl)

IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        where N = total documents, n(q) = docs containing q

TF(q, D, avgdl) = (f(q,D) × (k1 + 1)) / (f(q,D) + k1 × (1 - b + b × |D|/avgdl))
        where f(q,D) = term frequency, |D| = doc length, avgdl = average doc length
        k1 = 1.5 (TF saturation), b = 0.75 (length normalization)
```

**Why these values?**
- **k1=1.5**: At k1→∞, TF has no saturation (linearly increasing score). At k1=0, TF is binary (present/absent). k1=1.5 means doubling TF only adds ~33% more score — prevents documents from winning just by repeating a term.
- **b=0.75**: At b=0, length normalization is disabled. At b=1, fully normalized. b=0.75 means longer documents get slight credit for length (they're more likely to discuss a topic in depth), but not much.

**Line-by-line: `_tokenize`**

```python
def _tokenize(text: str) -> list[str]:
    return re.split(r"[^a-z0-9]+", text.lower())
```

`text.lower()` → case-fold ("Transformer" == "transformer"). `re.split(r"[^a-z0-9]+")` → split on any non-alphanumeric character — handles spaces, commas, parentheses, hyphens, etc. This is a **conservative tokenizer**: no stemming, no stopword removal. Scholar trades recall for precision — "neural" and "neuronal" are different tokens, which is fine for scientific text where terminology is precise.

**Line-by-line: `InvertedIndex.build`**

```python
def build(self, docs: list[str], doc_ids: list[str]) -> None:
    self.num_docs = len(docs)
    total_length = 0

    for doc_id, doc in zip(doc_ids, docs):
        tokens = _tokenize(doc)
        tokens = [t for t in tokens if t]   # remove empty strings from splitting
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len  # store per-doc length for BM25 norm
        total_length += doc_len

        term_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1         # count term frequency per doc

        for term, freq in term_counts.items():
            self.postings[term].append((doc_id, freq))
            # postings[term] = [(doc_id, freq), ...] for every doc containing term
            # This is the core of an inverted index

    self.avg_doc_length = total_length / self.num_docs if self.num_docs else 0.0
```

The inverted index maps `term → [(doc_id, freq), ...]`. For 200K papers with ~100 tokens each: ~20M tokens, reduced to ~500K unique terms. Lookup for any term is O(1) (dict lookup) + O(posting_list_length).

**Line-by-line: `BM25.query`**

```python
def query(self, q: str, top_k: int = 10) -> list[tuple[str, float]]:
    tokens = [t for t in _tokenize(q) if t]
    if not tokens:
        return []   # empty query → no results

    scores: Dict[str, float] = defaultdict(float)
    N = self.index.num_docs
    avg_dl = self.index.avg_doc_length

    for term in tokens:
        df = self.index.doc_freq(term)   # how many docs contain this term
        if df == 0:
            continue   # unknown term → skip (IDF would be undefined)

        # Robertson-Spärck Jones IDF (+1 to stay positive even if df is large)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in self.index.postings[term]:
            dl = self.index.doc_lengths.get(doc_id, 0)
            # Length normalization factor
            norm = self.k1 * (1 - self.b + self.b * dl / avg_dl) if avg_dl else 1.0
            # TF saturation formula
            tf_score = (tf * (self.k1 + 1)) / (tf + norm)
            scores[doc_id] += idf * tf_score  # accumulate over query terms

    # Sort by score descending, return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

The loop structure is efficient: for each query term, look up its posting list and update only the documents that contain it. Documents with no query terms never appear in `scores`. This is much faster than iterating all documents.

**Save/load: pickle**

```python
def save(self, path: str | Path) -> None:
    data = {
        "k1": self.k1,
        "b": self.b,
        "postings": dict(self.index.postings),   # defaultdict → dict for pickle
        "doc_lengths": self.index.doc_lengths,
        "num_docs": self.index.num_docs,
        "avg_doc_length": self.index.avg_doc_length,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
```

The serialized index is ~500MB for 200K papers (mostly the posting lists). At startup, `BM25.load()` deserializes into memory — the API keeps this in `app.state.bm25`.

---

### 4.2 Dense Retrieval — `scholar/retrieval/dense.py`

```python
EMBED_MODEL = "all-MiniLM-L6-v2"
```

**The choice:** `all-MiniLM-L6-v2` is a 6-layer MiniLM distilled from a larger teacher model. 384 dimensions, 80MB, ~14K sentences/sec on GPU. For Scholar's use case (scientific abstracts, 200K papers), it's the right balance of quality and speed.

**Line-by-line: `DenseRetriever.__init__`**

```python
def __init__(self) -> None:
    self._index = None     # FAISS index — None until load() is called
    self._paper_ids: list[str] = []  # FAISS position → paper ArXiv ID mapping
    self._model = None     # SentenceTransformer — lazy loaded
```

Lazy initialization: the sentence-BERT model doesn't load until `_ensure_model()` is called. This means importing `DenseRetriever` is free — the 80MB model only loads when actually needed.

**Line-by-line: `load`**

```python
def load(self) -> None:
    import faiss
    from scholar.config import settings

    index_path = settings.faiss_index_path        # e.g. ./data/faiss.index
    ids_path = str(index_path) + ".ids.json"      # parallel file: ./data/faiss.index.ids.json

    self._index = faiss.read_index(str(index_path))
    with open(ids_path, "r") as f:
        self._paper_ids = json.load(f)
```

FAISS index stores only vectors — it doesn't know paper IDs. The parallel `.ids.json` file maps `FAISS_position → paper_id`. This is why the `id_to_idx` cache in `main.py` is built as `{pid: i for i, pid in enumerate(dense._paper_ids)}` — the reverse mapping for O(1) lookups.

**Line-by-line: `_encode_and_normalize`**

```python
def _encode_and_normalize(self, text: str) -> np.ndarray:
    self._ensure_model()
    vec = self._model.encode([text], convert_to_numpy=True)[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm   # L2 normalize to unit sphere
    return vec.astype(np.float32)
```

**Why normalize?** The FAISS index uses `IndexHNSWFlat` with inner product (IP) metric on the normalized vectors. For unit vectors: `inner_product(a, b) = cos(a, b)`. So the "distance" returned is actually cosine similarity — higher is better (more similar). If you didn't normalize, larger-magnitude vectors would dominate purely due to magnitude.

**Line-by-line: `query`**

```python
def query(self, query_text: str, top_k: int = 200) -> list[tuple[str, float]]:
    if self._index is None:
        raise RuntimeError("DenseRetriever not loaded. Call load() first.")

    vec = self._encode_and_normalize(query_text)
    vec_2d = vec.reshape(1, -1)    # FAISS expects (n_queries, d) shape

    k = min(top_k, len(self._paper_ids))   # can't retrieve more than index size
    distances, indices = self._index.search(vec_2d, k)
    # distances: shape (1, k) — similarity scores
    # indices: shape (1, k) — FAISS positions

    results: list[tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(self._paper_ids):
            continue   # FAISS returns -1 for missing results in some index types
        paper_id = self._paper_ids[idx]
        results.append((paper_id, float(dist)))

    return results
```

The `idx < 0` guard: FAISS HNSW sometimes returns -1 as a sentinel when fewer than k results exist. Without this guard, `self._paper_ids[-1]` would return the last paper — a subtle silent bug.

**`query_by_vector`** — used in recommend.py for user-embedding-based retrieval:

```python
def query_by_vector(self, vec: np.ndarray, top_k: int = 200) -> list[tuple[str, float]]:
    # Same as query() but takes a pre-computed vector (user embedding)
    # instead of text to encode
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm   # normalize the user embedding too
    vec_2d = vec.astype(np.float32).reshape(1, -1)
    ...
```

---

### 4.3 Hybrid Retrieval — `scholar/retrieval/hybrid.py`

```python
def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            #                                                    ↑
            #                                     RRF score: 1/(k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Why RRF works:**

BM25 scores: [0.00031, 0.00028, 0.00025, ...] — raw float scores
FAISS scores: [0.98, 0.95, 0.92, ...] — cosine similarities (0-1)

These scales are incompatible. Naively combining them (weighted sum) would require tuning the weight — and the right weight changes depending on the query. RRF discards the scores entirely and uses only the rank:

```
Paper appears at BM25 rank 5:   RRF contribution = 1/(60+5)  = 0.01538
Paper appears at FAISS rank 12: RRF contribution = 1/(60+12) = 0.01389
Combined RRF score:             0.01538 + 0.01389 = 0.02927
```

The `k=60` constant controls how much rank matters at the top:
- k=0: difference between rank 1 and rank 2 is huge (1/1 - 1/2 = 0.5)
- k=60: difference between rank 1 and rank 2 is small (1/61 - 1/62 ≈ 0.0003)
- k=60 is the empirically validated default from the original Cormack et al. (2009) paper

**Why not k=0 (pure rank)?** At k=0, rank-1 documents get huge scores regardless of quality. With k=60, a document at rank 61 gets score 1/121 — still meaningful. The fusion is more robust to slight ranking variations.

**HybridRetriever.query:**

```python
def query(self, q: str, top_k: int = 200) -> list[tuple[str, float]]:
    bm25_results = self.bm25.query(q, top_k=top_k)      # up to 200 BM25 results
    dense_results = self.dense.query(q, top_k=top_k)    # up to 200 FAISS results
    fused = _reciprocal_rank_fusion([bm25_results, dense_results], k=60)
    return fused[:top_k]   # at most 200 unique papers
```

The union of BM25 and FAISS results: up to 400 unique papers in the fused list. After RRF, papers that appeared in both lists score highest. Return top 200.

---

## 5. Layer 2: Personalized Reranking

### 5.1 Mult-VAE — `scholar/recsys/mult_vae.py`

**The goal:** Given a user's click history (a sparse binary vector over 200K papers), produce a dense 200-dimensional embedding that captures their interests.

**Why VAE not just mean pooling?** Mean pooling of paper embeddings gives a point estimate. The VAE's encoder produces a distribution (μ, σ) in latent space. The reparameterization trick samples from this distribution during training, regularized by the KL term. This prevents the model from collapsing all users to the same embedding.

**Architecture:**

```
Input: x ∈ {0,1}^vocab_size (binary interaction vector)
    vocab_size = number of unique papers (~200K for full CiteULike)

Encoder:
    h = tanh(Linear(vocab_size → 600)(x))      # hidden rep
    [μ, logvar] = Linear(600 → 400)(h)         # split into mean and log-variance

Reparameterize:
    ε ~ N(0, I)
    z = μ + exp(0.5 × logvar) × ε              # z ∈ ℝ^200

Decoder:
    h' = tanh(Linear(200 → 600)(z))
    logits = Linear(600 → vocab_size)(h')
    output = log_softmax(logits)               # log-probabilities over papers
```

**Loss function:**

```python
class MultVAELoss(nn.Module):
    def forward(self, recon_x, x, mu, logvar) -> torch.Tensor:
        # Multinomial reconstruction loss
        # recon_x: log-softmax outputs (log p(paper | z))
        # x: binary interaction vector (which papers user clicked)
        # -sum(recon_x * x) = negative log-likelihood of clicked papers
        recon_loss = -torch.mean(torch.sum(recon_x * x, dim=-1))
        
        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0,I)
        # Closed form: -0.5 * sum(1 + logvar - μ² - e^logvar)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        
        return recon_loss + self.beta * kl_loss
```

**Why multinomial (not Bernoulli or Gaussian)?** 
- Bernoulli likelihood: treats each paper independently. But the number of papers a user reads is fixed (sparse) — Bernoulli doesn't capture "you'll read ~10 papers" naturally.
- Multinomial likelihood: models which papers the user reads given they'll read some papers. Naturally handles sparsity.
- Gaussian likelihood: appropriate for continuous-valued data, not binary clicks.

**Reparameterization trick:**

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if self.training:
        std = torch.exp(0.5 * logvar)    # σ = exp(logvar/2)
        eps = torch.randn_like(std)       # ε ~ N(0, I)
        return mu + eps * std             # z = μ + σ × ε
    return mu   # at inference: use the mean (deterministic)
```

**Why use μ at inference?** The reparameterization trick is only needed during training to make the gradient flow through the sampling operation (gradients don't pass through `randn_like`). At inference, we want the most likely latent code, which is μ.

**KL annealing:**

```python
# train_vae.py — inside the training loop
beta = min(0.2, epoch / 20 * 0.2)   # β: 0 → 0.2 over 20 epochs
```

- Epoch 1: β=0.01 (almost no KL penalty — encoder can do anything)
- Epoch 10: β=0.10 (moderate regularization)
- Epoch 20+: β=0.20 (full regularization)

**Why anneal?** If you apply full KL penalty from epoch 1, the model takes the easy way out: set σ→0, making the encoder deterministic. The decoder then learns to reconstruct perfectly without ever using the stochastic bottleneck. The KL loss is 0 (delta distribution = no info loss), but the model is just a deterministic autoencoder. Annealing lets the model first learn a useful reconstruction, then gradually forces it to be probabilistic.

**`get_user_embedding` — the inference function:**

```python
def get_user_embedding(
    history_paper_ids: list[str],
    all_paper_ids: list[str],
    model: Optional[MultVAE],
) -> np.ndarray:
    # Cold start
    if not history_paper_ids:
        latent_dim = model.latent_dim if model is not None else 200
        return np.zeros(latent_dim, dtype=np.float32)

    if model is None:
        model = _load_trained_model()

    # Fallback if model untrained
    if model is None or _is_near_init(model):
        return _fallback_sentence_bert(history_paper_ids)
    
    # Build one-hot interaction vector
    id_to_idx = {pid: i for i, pid in enumerate(all_paper_ids)}
    x = torch.zeros(1, vocab_size, dtype=torch.float32)
    for pid in history_paper_ids:
        if pid in id_to_idx:
            x[0, id_to_idx[pid]] = 1.0

    # Get latent mean (no gradient needed)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.squeeze(0).numpy()   # shape (200,)
```

The `_is_near_init` heuristic: if the decoder's weight mean absolute value is < 0.05, the model hasn't trained meaningfully. Falls back to averaging sentence-BERT embeddings of the history papers — a reasonable substitute.

---

### 5.2 LightGBM LambdaRank — `scholar/recsys/ltr.py`

**The problem LTR solves:** After RRF, we have 200 papers sorted by a rank-fusion score. But this score doesn't know:
- The user's reading history
- Paper citation count
- How recently the paper was published
- Whether the paper's categories match the user's interests

LambdaRank learns a function that takes 7 features per (user, paper) pair and outputs a relevance score.

**The 7 features:**

```python
FEATURE_NAMES = [
    "faiss_score",      # cosine similarity from FAISS (0.85-0.99 range)
    "bm25_score",       # BM25 score — note: always 0.0 in Scholar (known issue)
    "citation_count",   # paper's citation count
    "days_since_pub",   # days since update_date (recency)
    "category_overlap", # Jaccard overlap between paper categories and user history categories
    "user_paper_cosine",# cosine similarity between user embedding and paper embedding
    "is_in_history",    # 1.0 if user has already read this paper, 0.0 otherwise
]
```

**Known gap — bm25_score=0.0:**

```python
rows.append([
    faiss_score,
    0.0,               # bm25_score — filled by caller... but the caller never fills it
    citation_count,
    ...
])
```

The comment "filled by caller" is aspirational — no caller actually fills it. BM25 scores aren't passed through to `extract_features`. This is documented in the codebase as a known limitation. The model still works because the other 6 features capture enough signal, and LightGBM's feature importance will just assign near-zero weight to `bm25_score`.

**LambdaRank gradient:**

LambdaRank is a listwise LTR method. Instead of pointwise MSE or pairwise cross-entropy, it directly optimizes NDCG@K by computing per-document lambda gradients:

```
For every pair (i, j) where i is relevant and j is not:
λᵢⱼ = |ΔNDCG| / (1 + exp(sᵢ - sⱼ))

Per-document gradient:
λᵢ = Σⱼ λᵢⱼ   (sum over all pairs involving document i)
```

`|ΔNDCG|` is the gain in NDCG@K from swapping the ranks of i and j. If swapping would cause a large NDCG improvement, the gradient is large — pushing i higher and j lower.

**Training data construction:**

```python
for uid, pos_papers in user_papers.items():
    # Positive samples: papers the user actually clicked (relevance label = 3)
    pos_papers_list = [p for p in pos_papers if p in paper_meta]
    
    # Negative samples: random papers from the corpus
    neg_pool = [p for p in all_paper_ids if p not in pos_papers]
    n_neg = min(len(pos_papers_list) * 4, len(neg_pool), 20)
    # 1:4 ratio — standard in learning-to-rank literature
    # Rationale: relevance is sparse; more negatives = better decision boundary
    neg_papers = list(rng.choice(neg_pool, n_neg, replace=False))
    
    labels = [3] * len(pos_papers_list) + [0] * len(neg_papers)
    # Label 3 = "definitely relevant" (clicked)
    # Label 0 = "not relevant" (never clicked)
    # Label scale 0-3 follows TREC relevance convention
```

**LightGBM parameters:**

```python
params = {
    "objective": "lambdarank",  # triggers LambdaRank gradient computation
    "metric": "ndcg",
    "ndcg_eval_at": [10],        # eval NDCG@10
    "learning_rate": 0.05,       # step size (lower = more robust, slower)
    "num_leaves": 31,            # tree complexity (31 = standard for tabular)
    "min_data_in_leaf": 1,       # minimum samples per leaf
}
model = lgb.train(params, train_data, num_boost_round=100)
```

The `groups` parameter for LambdaRank:

```python
train_data = lgb.Dataset(X_train, label=y_train, group=groups, feature_name=FEATURE_NAMES)
# group = [len(user1_candidates), len(user2_candidates), ...]
# tells LightGBM which rows belong to the same query
# pairs are only formed WITHIN groups — you don't compare user1's papers with user2's
```

**Inference:**

```python
def score(self, user_emb, candidates, paper_meta, paper_embeddings, history_cats):
    X = extract_features(...)    # (N, 7) feature matrix
    preds: np.ndarray = self._model.predict(X)  # (N,) relevance scores
    return sorted(
        zip(candidate_ids, preds.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    # Output: candidates sorted by LTR relevance score, descending
```

---

### 5.3 MMR — `scholar/recsys/mmr.py`

**The problem:** LambdaRank's top-50 might contain 10 papers all about "diffusion models for image synthesis". A user interested in diffusion models broadly would want to see applications to text, audio, 3D — not 10 variants of the same paper.

**Maximum Marginal Relevance** balances relevance and diversity:

```
MMR(dᵢ) = λ × Relevance(dᵢ) - (1-λ) × max_{dⱼ ∈ Selected} Similarity(dᵢ, dⱼ)
```

Greedy selection: at each step, add the document with the highest MMR score.

```python
def mmr_rerank(candidates, embeddings, lam=0.5, top_k=10):
    selected = []
    remaining = list(candidates_with_emb)

    while remaining and len(selected) < top_k:
        best_pid = None
        best_score = float("-inf")

        for pid, rel in remaining:
            emb = embeddings[pid]
            if not selected:
                diversity_penalty = 0.0   # first pick: no diversity constraint
            else:
                sims = [_cosine_sim(emb, embeddings[s_pid]) for s_pid, _ in selected]
                diversity_penalty = max(sims)   # penalize by similarity to MOST similar selected doc

            mmr_score = lam * rel - (1 - lam) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_pid = pid

        selected.append((best_pid, best_score))
        remaining = [(pid, rel) for pid, rel in remaining if pid != best_pid]

    return selected
```

**Why `max(sims)` not `mean(sims)`?** If you've already selected 3 papers about diffusion models, a 4th diffusion paper is very similar to all 3. Using `max` means even partial overlap with the most similar selected doc is heavily penalized. `mean` would undercount this overlap (the 4th paper might be similar to 3 selected but dissimilar to 2 others, lowering the mean).

**Adaptive lambda:**

```python
def adaptive_lambda(user_history_embeddings: list[np.ndarray]) -> float:
    if len(user_history_embeddings) < 3:
        return 0.3   # cold start: inject diversity (user hasn't shown preferences yet)

    # Compute average pairwise cosine similarity of history embeddings
    avg_sim = ...   # mean of all pairs

    if avg_sim > 0.7:   # specialist: reads only one topic
        return 0.3      # λ=0.3: prefer diversity (they're stuck in a bubble)
    if avg_sim < 0.4:   # generalist: reads many topics
        return 0.7      # λ=0.7: prefer relevance (they want quality hits)

    # Linear interpolation between 0.7 and 0.3 as avg_sim goes 0.4→0.7
    t = (avg_sim - 0.4) / (0.7 - 0.4)
    return 0.7 - t * (0.7 - 0.3)
```

**Why O(K²) is acceptable on top-50:**

MMR is O(K² × D) where K = candidates and D = embedding dimension. With K=50 and D=384:
- 50² × 384 = 960,000 operations per request
- At ~10ns/op: < 10ms
- Completely acceptable at Scholar's scale

If you ran MMR on all 200 candidates: 200² = 40,000 × more expensive. Still manageable (< 100ms), but Scholar uses top-50 to leave headroom and focus on high-quality candidates.

---

### 5.4 ε-Greedy Exploration — `scholar/recsys/exploration.py`

**The filter bubble problem:** A collaborative filtering system that always shows you papers similar to what you've read will reinforce your existing interests. You'll never discover adjacent fields that could be valuable.

ε-greedy is borrowed from multi-armed bandits. With probability ε=0.1, replace a result with an exploratory choice.

```python
def epsilon_greedy_inject(ranked, all_paper_ids, history_ids, epsilon=0.1,
                          quality_percentile=0.8, paper_meta=None):
    n_inject = max(1, math.ceil(epsilon * len(ranked)))
    n_inject = min(n_inject, 2)   # cap at 2 — don't destroy the experience
    
    # Only inject papers NOT already seen + NOT already ranked
    candidates = [
        pid for pid in all_paper_ids
        if pid not in (history_ids | ranked_ids) and pid in paper_meta
    ]
    
    # Quality filter: only high-citation papers (top 80% percentile)
    candidates.sort(key=lambda pid: paper_meta[pid].citation_count, reverse=True)
    cutoff = max(1, int(len(candidates) * quality_percentile))
    high_quality = candidates[:cutoff]   # top 80% by citation count
    
    # Random sample from high-quality pool
    injected = random.sample(high_quality, min(n_inject, len(high_quality)))
    
    # Replace the tail of ranked results (last 1-2 slots)
    min_score = min(score for _, score in ranked) if ranked else 0.0
    result = list(ranked[:-n_inject])   # keep all but last n_inject
    for pid in injected:
        result.append((pid, min_score * 0.9))   # slightly lower score than tail
    
    return result
```

**Key design choices:**
- **Replace tail, not head**: injecting at position 10 (last) is less disruptive than replacing position 1
- **Quality filter**: random injection from all papers could surface a 2-citation preprint. Restricting to top 80% citation count ensures exploratory papers are still worth reading
- **max 2 injections**: at 10 total results, 2 exploratory = 20% (double the nominal ε). Cap prevents the explore rate from overwhelming the exploit results.

---

## 6. Layer 3: Explanation Generation

### 6.1 Phi-3 Inference — `scholar/generation/inference.py`

**Why Phi-3-mini-4k-instruct:**
- 3.8B parameters — smallest model that produces coherent academic summaries
- MIT license — can use commercially + modify
- 4k context window — enough for title (200 chars) + abstract (1500 chars) + prompt template
- Microsoft's SLM (Small Language Model) — optimized for instruction following

**Memory budget on RTX 2050 (4GB):**

| Component | VRAM |
|-----------|------|
| Phi-3 weights (4-bit NF4) | ~1.9 GB |
| Activation memory (seq_len 400) | ~0.8 GB |
| CUDA runtime + PyTorch | ~0.3 GB |
| Margin | ~1.0 GB |
| **Total** | ~4.0 GB |

**Loading the model:**

```python
class Phi3Generator:
    _instance: Optional["Phi3Generator"] = None   # singleton

    def __init__(self) -> None:
        # 4-bit NF4 quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # use 4-bit precision
            bnb_4bit_quant_type="nf4",      # NF4: quantile-based, better than int4
            bnb_4bit_compute_dtype=torch.bfloat16,  # matmul in bfloat16
            bnb_4bit_use_double_quant=True, # quantize the quantization constants too
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",              # bitsandbytes decides which GPU/CPU layers
            quantization_config=bnb_config,
            trust_remote_code=True,         # Phi-3 has custom model code in HF Hub
            attn_implementation="eager",    # don't use Flash Attention (not installed)
        )
```

**NF4 vs int4:** Regular int4 quantization maps weights linearly to 4-bit integers. NF4 (Normal Float 4) uses quantile-based levels: more quantization levels near zero (where most weights cluster for neural networks), fewer at extremes. This gives better perplexity at the same bit width. The NF4 levels are: {-1, -0.6961, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}.

**LoRA adapter loading (if fine-tuned):**

```python
if settings.lora_adapter_path is not None and settings.lora_adapter_path.exists():
    from peft import PeftModel
    self.model = PeftModel.from_pretrained(
        self.model,
        str(settings.lora_adapter_path),
    )
```

`PeftModel.from_pretrained` wraps the base model with LoRA adapters. The adapters add ΔW = B×A to the QKV projection layers. The base model weights remain 4-bit frozen; only the LoRA A and B matrices (bfloat16, ~17MB) are loaded.

**Generation:**

```python
def generate(self, title: str, abstract: str, why: str, level: str = "grad") -> dict:
    prompt = get_prompt(level=level, title=title, abstract=abstract, why=why)
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

    with torch.no_grad():
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=400,     # cap at 400 tokens (~300 words)
            temperature=0.7,        # slightly random (0=deterministic, 1=very random)
            do_sample=True,         # sample instead of greedy
            pad_token_id=self.tokenizer.eos_token_id,  # Phi-3 has no pad token
        )

    # output_ids includes the input prompt tokens — slice them off
    generated = self.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],  # [prompt_length:]
        skip_special_tokens=True,
    )

    return _parse_output(generated)
```

**Why slice at `inputs["input_ids"].shape[1]:`?** `model.generate()` returns the full sequence: input tokens + generated tokens. If you decode the full sequence, you get the prompt text followed by the model's response. Slicing from `prompt_length:` gives only the generated text.

**Output parsing:**

```python
def _parse_output(text: str) -> dict:
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=GLOSSARY:|WHY:|$)", text, re.DOTALL)
    glossary_match = re.search(r"GLOSSARY:\s*(.+?)(?=WHY:|$)", text, re.DOTALL)
    why_match = re.search(r"WHY:\s*(.+?)$", text, re.DOTALL)
    ...
```

The structured output format (SUMMARY: / GLOSSARY: / WHY:) is enforced by the prompt template. The regex parsing is tolerant — if the model doesn't follow the format exactly, it falls back to using the raw text as the summary.

---

### 6.2 Prompts — `scholar/generation/prompts.py`

Three templates: UNDERGRAD, GRAD, RESEARCHER. They use Phi-3's chat format:

```
<|user|>
{instruction and paper context}
<|end|>
<|assistant|>
```

The `<|user|>`, `<|end|>`, `<|assistant|>` tokens are Phi-3's special tokens for instruction-following. Without these, the model generates in base completion mode rather than instruction-following mode — outputs would be poor.

**Level differentiation:**

| Level | Language | Glossary focus | Audience |
|-------|---------|----------------|---------|
| undergrad | "plain English", "everyday language", "first-year college student" | "simple definitions" | CS students |
| grad | "technical summary", "field terminology" | "key technical terms" | MS/PhD students |
| researcher | "expert summary", "novel contributions, methodology, limitations" | "highly technical terms, models, frameworks" | Post-docs, professors |

---

## 7. API Layer

### 7.1 Application Startup — `scholar/api/main.py`

The `lifespan` context manager runs once at startup and once at shutdown. It initializes all stateful resources and attaches them to `app.state`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ──────────────────────────────────────────────

    # 1. Apply DB schema (synchronous — before async event loop)
    _create_tables_sync()
    # Uses psycopg2 (sync) to run schema.sql: CREATE TABLE IF NOT EXISTS ...
    # `IF NOT EXISTS` = idempotent: safe to run on every restart

    # 2. Create async DB engine + session factory
    engine = create_async_engine(settings.database_url, echo=False)
    async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    app.state.async_session_factory = async_session_factory
    app.state.engine = engine
    # Store on app.state: FastAPI routes access this via request.app.state

    # 3. Load BM25 index (pickle file → memory)
    bm25 = _load_bm25()          # None if data/bm25.pkl doesn't exist
    app.state.bm25 = bm25

    # 4. Load FAISS index
    dense = _load_dense()        # None if data/faiss.index doesn't exist
    app.state.dense_retriever = dense

    # 5. Create HybridRetriever (wraps BM25 + FAISS)
    if bm25 and dense:
        app.state.hybrid_retriever = HybridRetriever(bm25=bm25, dense=dense)
    else:
        app.state.hybrid_retriever = None   # search returns 503 if None

    # 6. Build id_to_idx cache: paper_id → FAISS index position
    if dense is not None:
        app.state.id_to_idx = {pid: i for i, pid in enumerate(dense._paper_ids)}
    # This dict is O(N) memory (~200K entries × 40 bytes ≈ 8MB) but gives O(1) lookup
    # Alternative: faiss.reconstruct_n() is ~100x slower per lookup than dict

    # 7. Load LTR model
    ltr = _load_ltr()            # None if data/ltr_model.lgb doesn't exist
    app.state.ltr_model = ltr

    # 8. Optional: pre-warm Phi-3 generator
    if settings.warmup_generation and settings.generation_available:
        Phi3Generator.get_instance()   # loads 4-bit model (~2-4 minutes)

    yield   # ← app is running, serving requests

    # ── Shutdown ─────────────────────────────────────────────
    await engine.dispose()   # close all DB connections
```

**Why store on `app.state` and not as globals?**

Globals work in single-process servers. With multiple uvicorn workers (or when running tests with `TestClient`), each process/instance would need its own copy of the index, and there's no clean cleanup. `app.state` is tied to the application lifecycle and accessible from every request via `request.app.state`.

---

### 7.2 Search Route — `scholar/api/routes/search.py`

```python
@router.post("/search", response_model=SearchResponse)
async def search(
    request_body: SearchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> SearchResponse:

    # 1. Get resources from app.state
    hybrid_retriever = getattr(request.app.state, "hybrid_retriever", None)
    dense_retriever = getattr(request.app.state, "dense_retriever", None)
    async_session_factory = request.app.state.async_session_factory

    if hybrid_retriever is None:
        raise HTTPException(status_code=503, detail="Search index not loaded.")
        # 503 = Service Unavailable: the service exists but isn't ready

    # 2. Hybrid retrieval (BM25 + FAISS + RRF) → 200 candidates
    candidates = hybrid_retriever.query(request_body.query, top_k=200)
    candidate_ids = [pid for pid, _ in candidates]
    score_map = {pid: score for pid, score in candidates}

    # 3. Fetch paper metadata from DB (one query for all 200 papers)
    async with async_session_factory() as session:
        papers_map = await _get_papers_by_ids(session, candidate_ids)
        # Also fetch user history (for adaptive lambda)
        history_ids = await _get_user_history(session, request_body.user_id) if request_body.user_id else []

    # 4. Retrieve paper embeddings via FAISS reconstruct()
    id_to_idx = getattr(request.app.state, "id_to_idx", {})

    # 5. Compute adaptive lambda from user history diversity
    if request_body.user_id and history_ids and dense_retriever is not None:
        lam = request_body.diversity_lambda   # default 0.5
        if lam == 0.5:   # user wants auto-tuned lambda
            history_embeddings = [
                np.array(dense_retriever._index.reconstruct(id_to_idx[hid]), dtype=np.float32)
                for hid in history_ids if hid in id_to_idx
            ]
            if history_embeddings:
                lam = adaptive_lambda(history_embeddings)   # 0.3-0.7

        # 6. MMR reranking
        paper_embeddings = _get_embeddings_for_ids(candidate_ids, dense_retriever, id_to_idx)
        reranked = mmr_rerank(candidates, paper_embeddings, lam, top_k=request_body.top_k)
        final_candidates = reranked
    else:
        final_candidates = candidates[:request_body.top_k]   # no user → just take top-K

    # 7. Build response
    results = [
        PaperResult(id=paper.id, title=paper.title, ..., score=score)
        for paper_id, score in final_candidates
        if (paper := papers_map.get(paper_id)) is not None
    ]

    response = SearchResponse(query=request_body.query, results=results, total_candidates=total)

    # 8. Log clicks in background (non-blocking)
    background_tasks.add_task(
        _log_search_clicks,
        async_session_factory,
        request_body.user_id,
        request_body.query,
        results,
    )
    # BackgroundTask runs AFTER the response is sent — user gets response instantly

    return response
```

**Why `getattr(request.app.state, "hybrid_retriever", None)` not `request.app.state.hybrid_retriever`?**

If `hybrid_retriever` was never set (e.g., lifespan failed partway through), direct attribute access raises `AttributeError`. `getattr(..., None)` returns None gracefully, then the `if hybrid_retriever is None: raise HTTPException(503)` path handles it cleanly.

---

### 7.3 Recommend Route — `scholar/api/routes/recommend.py`

The recommend route is more complex than search — it must build a user embedding, use LTR, apply MMR, apply ε-greedy, and handle fallbacks at each step.

```python
@router.get("/recommend/{user_id}")
async def recommend(user_id: str, level: str = "grad", request: Request = None,
                    background_tasks: BackgroundTasks = None):

    # 1. Verify user exists + fetch history
    async with session_factory() as session:
        user = await session.execute(select(User).where(User.id == user_id))
        user = user.scalars().first()
        if user is None:
            raise HTTPException(404, f"User {user_id} not found.")

        history_ids = await session.execute(
            select(UserHistory.paper_id).where(UserHistory.user_id == user_id)
        )
        history_ids = list(history_ids.scalars().all())

    if not history_ids:
        raise HTTPException(422, "User has no history.")
        # 422 = Unprocessable Entity: request is valid but can't be completed
        # 404 would be wrong — the user exists, they just have no data

    # 2. Build user embedding from history
    id_to_idx = request.app.state.id_to_idx
    history_embeddings = [
        np.array(dense_retriever._index.reconstruct(id_to_idx[hid]), dtype=np.float32)
        for hid in history_ids if hid in id_to_idx
    ]

    if history_embeddings:
        user_emb = np.mean(history_embeddings, axis=0).astype(np.float32)
        # Simple average of paper embeddings = centroid of user's interest space
    else:
        user_emb = get_user_embedding(history_ids, dense_retriever._paper_ids, model=None)
        # Fallback to Mult-VAE (or sentence-BERT average of abstracts)

    # 3. FAISS retrieval using user embedding
    candidates = dense_retriever.query_by_vector(user_emb, top_k=200)
    # Returns papers close to the user's interest centroid
    candidates = [(pid, score) for pid, score in candidates if pid not in set(history_ids)]
    # Exclude already-seen papers

    # 4. Fetch metadata for all 200 candidates
    async with session_factory() as session:
        papers = {p.id: p for p in (await session.execute(select(Paper).where(
            Paper.id.in_([pid for pid, _ in candidates])
        ))).scalars().all()}

    # 5. Retrieve paper embeddings for LTR features
    paper_embeddings = {
        pid: np.array(dense_retriever._index.reconstruct(id_to_idx[pid]), dtype=np.float32)
        for pid in [p for p, _ in candidates] if pid in id_to_idx
    }

    # 6. LTR reranking → top 50
    if ltr_model is not None:
        history_cats = [papers[hid].categories for hid in history_ids if hid in papers]
        ltr_ranked = ltr_model.score(
            user_emb=user_emb,
            candidates=candidates,
            paper_meta=papers,
            paper_embeddings=paper_embeddings,
            history_cats=history_cats,
        )
        mmr_input = ltr_ranked[:50]   # top 50 for MMR
    else:
        mmr_input = candidates[:50]   # fallback: FAISS order

    # 7. MMR diversity reranking → 10 papers
    lam = adaptive_lambda(history_embeddings) if history_embeddings else 0.3
    reranked = mmr_rerank(candidates=mmr_input, embeddings=paper_embeddings, lam=lam, top_k=10)

    # 8. ε-greedy exploration (injects 1-2 random high-quality papers)
    reranked = epsilon_greedy_inject(
        ranked=reranked,
        all_paper_ids=dense_retriever._paper_ids,
        history_ids=set(history_ids),
        epsilon=0.1,
        paper_meta=papers,
    )

    # 9. Fetch metadata for any newly injected papers
    injected_ids = [pid for pid, _ in reranked if pid not in papers]
    if injected_ids:
        async with session_factory() as session:
            for p in (await session.execute(select(Paper).where(Paper.id.in_(injected_ids)))).scalars().all():
                papers[p.id] = p

    # 10. Build response + log in background
    results = [PaperResult(...) for paper_id, score in reranked if papers.get(paper_id)]
    background_tasks.add_task(_log_clicks, session_factory, user_id, reranked)
    return SearchResponse(query=f"Personalized recommendations for user {user_id}", results=results, ...)
```

---

### 7.4 User Routes — `scholar/api/routes/user.py`

```python
@router.post("/user")
async def create_user(request: Request) -> dict:
    user_id = str(uuid.uuid4())   # UUID4 = random 128-bit ID, collision probability negligible
    user = User(id=user_id, created_at=datetime.now())
    async with session_factory() as session:
        session.add(user)
        await session.commit()
    return {"user_id": user_id}

@router.post("/user/{user_id}/history")
async def add_to_history(user_id: str, body: UserHistoryItem, request: Request) -> dict:
    async with session_factory() as session:
        # Check if paper already in history (composite PK prevents duplicates)
        existing = await session.execute(
            select(UserHistory).where(
                UserHistory.user_id == user_id,
                UserHistory.paper_id == body.paper_id,
            )
        )
        if existing.scalars().first() is None:
            session.add(UserHistory(user_id=user_id, paper_id=body.paper_id, clicked_at=datetime.now()))
            await session.commit()
    return {"user_id": user_id, "paper_id": body.paper_id, "added": existing is None}
```

---

## 8. Data Ingestion Pipeline

### 8.1 ArXiv Loader

Scholar ingests from the [ArXiv bulk metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (~4GB JSON-lines file, 2M+ papers).

```python
# scholar/ingest/arxiv_loader.py (high-level flow)
async def load_arxiv(data_path: Path, limit: int = 50000) -> None:
    pool = await asyncpg.create_pool(settings.database_url)
    
    batch = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            paper = json.loads(line)
            # Extract: id, title, abstract, authors, categories, update_date
            record = (
                paper["id"],                                    # "2301.00001"
                paper["title"].strip().replace("\n", " "),      # clean whitespace
                paper.get("abstract", "").strip(),
                ", ".join(paper.get("authors_parsed", [["Unknown"]])[0]),
                " ".join(paper.get("categories", "cs.LG").split()),
                paper.get("update_date"),
            )
            batch.append(record)
            
            if len(batch) >= 1000:
                await _upsert_batch(pool, batch)
                batch = []
    
    if batch:
        await _upsert_batch(pool, batch)
    
    await pool.close()
```

### 8.2 Embedding Builder

```python
# scholar/ingest/embeddings.py (high-level flow)
async def build_embeddings(batch_size: int = 512) -> None:
    # 1. Load all papers from DB
    papers = await fetch_all_papers()   # returns list of (id, title, abstract)
    
    # 2. Encode in batches (batch_size=512 for GPU efficiency)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{title} {abstract}" for _, title, abstract in papers]
    embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)
    # ~13 minutes on RTX 2050 for 200K papers
    
    # 3. Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    
    # 4. Save to file (for FAISS index building)
    np.save("./data/embeddings.npy", embeddings)
    
    # 5. Update pgvector column in DB
    for (paper_id, _, _), emb in zip(papers, embeddings):
        await conn.execute("UPDATE papers SET embedding=$2 WHERE id=$1",
                          paper_id, emb.tolist())
```

### 8.3 Index Building

```python
# scripts/build_index.py
def build_faiss_index(embeddings_path: Path, faiss_path: Path) -> None:
    embeddings = np.load(str(embeddings_path))  # (N, 384) float32
    
    # IndexHNSWFlat: HNSW graph + exact distance computation within graph
    # M=32: each node connects to 32 nearest neighbors (higher = better recall, more memory)
    # Space: "ip" = inner product (equivalent to cosine for normalized vectors)
    index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200   # search width during build (200 = good quality)
    
    index.add(embeddings)   # adds all vectors; builds HNSW graph (~5 min for 200K)
    faiss.write_index(index, str(faiss_path))
    
    # Save paper ID → FAISS position mapping
    ids_path = str(faiss_path) + ".ids.json"
    with open(ids_path, "w") as f:
        json.dump(paper_ids, f)
```

---

## 9. Training Scripts

### 9.1 Mult-VAE Training — `scholar/recsys/train_vae.py`

```
Input: user_history table (user_id, paper_id) pairs from CiteULike
        ~204K users × 167K papers ≈ 4.7M interactions

Matrix construction:
    X[user_idx, paper_idx] = 1.0 if user clicked paper

Split: 80% train, 10% val, 10% test (by users, not by papers)

Model: MultVAE(vocab_size=167K, latent_dim=200, hidden_dim=600)

Training:
    Epochs: 100
    Batch size: 256 (each row = one user's binary interaction vector)
    Optimizer: Adam(lr=1e-3)
    KL annealing: β = min(0.2, epoch/20 × 0.2)
    Validation: NDCG@10 on reconstruction quality

Output: data/mult_vae.pt + data/vae_vocab.json
```

The 80/10/10 split is by users: all of one user's interactions go to the same split. This prevents data leakage — the model never sees validation users' interactions during training.

### 9.2 QLoRA Fine-Tuning — `scholar/generation/train_lora.py`

```
Training data: data/synth_data.json
    Format: {"title": ..., "abstract": ..., "why": ..., "level": ..., "output": ...}
    Generated by: synth_data.py using the base Phi-3 model with diverse prompts

Model setup:
    Base: Phi-3-mini-4k-instruct (4-bit NF4, frozen)
    LoRA: r=8, alpha=16, target_modules=["qkv_proj"]
    Effective rank ratio: alpha/r = 2.0

Training:
    SFTTrainer (HuggingFace trl library)
    batch=1, grad_accum=8 (effective batch=8)
    optimizer: paged_adamw_8bit (gradient checkpointing enabled)
    bf16=True, seq_len=768
    Epochs: 3

Output: data/lora_adapter/ (adapter_config.json + adapter_model.bin)
```

**LoRA math in Scholar:**
- Phi-3 qkv_proj: d_model=3072, projects to Q+K+V (3072×3 = 9216 dim)
- With r=8: A ∈ ℝ^{3072×8}, B ∈ ℝ^{8×9216}
- Parameters per layer: 3072×8 + 8×9216 = 24,576 + 73,728 = 98,304
- 32 transformer layers → 32 × 98,304 = 3.1M trainable parameters
- vs. 3.8B total parameters → 0.08% of model trained
- Memory for LoRA matrices: 3.1M × 2 bytes (bf16) ≈ 6MB

---

## 10. UI Layer — `scholar/ui/gradio_app.py`

The Gradio UI is a thin client over the FastAPI backend. It's a separate container in Docker, calling `http://api:8000` via Docker's internal DNS.

**Key pattern — render_paper_card:**

```python
def render_paper_card(paper: dict) -> str:
    return f"""
    <div class="paper-card">
        <a href="https://arxiv.org/abs/{paper['id']}" target="_blank">
            {paper['title']}
        </a>
        <div class="paper-meta">{paper['authors']} · {paper['categories']}</div>
        <div class="paper-summary">{paper['abstract'][:300]}...</div>
    </div>
    """
```

The `gr.HTML()` output component renders raw HTML. Custom CSS in `gr.Blocks(css=...)` styles the cards. This is more flexible than Gradio's built-in Dataframe or JSON components — it lets Scholar show styled cards with clickable ArXiv links.

**User session management:**

1. Page loads → `demo.load()` fires → `init_user()` creates a user in DB → stores UUID in `gr.State`
2. User searches → `search()` passes `user_id` → API logs clicks in DB
3. User requests recommendations → `recommend()` passes `user_id` → API fetches history from DB

---

## 11. Configuration and Infrastructure

### `scholar/config.py`

```python
class Settings(BaseSettings):
    database_url: str           # required — crash immediately if missing
    database_url_sync: str      # psycopg2 sync URL for startup schema creation
    faiss_index_path: Path = Path("./data/faiss.index")
    hf_home: str = "./data/hf_cache"
    lora_adapter_path: Optional[Path] = None
    device: str = "cuda"
    generation_available: bool = True
    warmup_generation: bool = False   # set True to pre-load Phi-3 at startup

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()   # instantiated at module import time
```

Every module imports `settings` directly: `from scholar.config import settings`. This creates a single config object for the whole application.

### `docker-compose.yml` — the three services

```
db (pgvector/pgvector:pg16)
    ← provides: Postgres 16 + pgvector
    ← healthcheck: pg_isready -U scholar
    ← volume: pgdata (persistent paper + user data)

api (built from Dockerfile)
    ← depends_on db: service_healthy
    ← volume: ./data (FAISS index, BM25 pickle, model weights)
    ← env: DATABASE_URL, HF_HOME, etc.

ui (built from Dockerfile)
    ← depends_on api
    ← calls: http://api:8000 (Docker internal DNS)
```

---

## 12. Complete Request Trace

### Scenario: "diffusion models" search, first-time user

```
1. Browser → POST localhost:7860/run/search
   Gradio receives: query="diffusion models", level="grad", user_id=None

2. Gradio init_user() ran on page load → user_id = "f47ac10b-..."
   (stored in gr.State, passed to search function)

3. Gradio → POST http://api:8000/search
   JSON: {"query": "diffusion models", "level": "grad", "user_id": "f47ac10b-..."}

4. FastAPI search route:
   a. hybrid_retriever.query("diffusion models", top_k=200)
      → bm25.query("diffusion models", 200):
          tokenize: ["diffusion", "models"]
          IDF("diffusion") = log((200K - 5K + 0.5)/(5K + 0.5) + 1) = 3.64
          IDF("models") = log((200K - 80K + 0.5)/(80K + 0.5) + 1) = 1.47
          for each doc in postings["diffusion"] + postings["models"]: compute BM25
          → [(paper_id, score), ...] top 200
      → dense.query("diffusion models", 200):
          encode("diffusion models") → 384-dim vector
          FAISS.search(vec, 200) → [(idx, cosine_sim), ...]
          → [(paper_id, similarity), ...] top 200
      → RRF([bm25_results, dense_results], k=60):
          for each list: score[doc] += 1/(60+rank)
          → up to 400 unique papers, sorted by RRF score

   b. await _get_papers_by_ids(session, candidate_ids)
      → SELECT * FROM papers WHERE id IN ('2301.00001', ...)
      → dict: {paper_id: Paper}

   c. No user history yet → skip adaptive lambda
      final_candidates = candidates[:10]   # top 10 by RRF

   d. Build SearchResponse with 10 PaperResult objects

   e. background_tasks.add_task(_log_search_clicks, ...)
      → async INSERT into click_logs (runs AFTER response sent)

5. FastAPI → 200 OK with JSON SearchResponse

6. Gradio renders HTML cards from paper dicts
   Browser displays: 10 paper cards with titles, abstracts, ArXiv links
```

### Scenario: personalized recommendations, user with 50-paper history

```
1. User clicks "Get Recommendations" after reading 50 papers

2. GET http://api:8000/recommend/f47ac10b-...?level=grad

3. FastAPI recommend route:
   a. Fetch user + history_ids (50 paper IDs)

   b. Build user embedding:
      - For each history paper: reconstruct FAISS vector via id_to_idx
      - user_emb = np.mean(50 × 384-dim vectors) → centroid of interest space
      (avg pairwise similarity of history ≈ 0.45 → adaptive_lambda ≈ 0.65)

   c. dense.query_by_vector(user_emb, 200)
      → FAISS finds 200 papers closest to user's interest centroid
      → remove 50 already-seen papers → 150 candidates

   d. Fetch paper metadata for 150 candidates

   e. LTR scoring (if model available):
      extract_features(user_emb, candidates, papers, embeddings, history_cats)
      → (150, 7) feature matrix
      ltr_model.predict(X) → 150 relevance scores
      ltr_ranked = sorted by score, take top 50

   f. MMR reranking of top 50:
      λ = 0.65 (slightly prefer relevance over diversity)
      greedy MMR selection → 10 papers with good diversity

   g. ε-greedy: inject 1 high-citation paper from top-80% of corpus

   h. Fetch metadata for any injected papers

   i. Return 10 personalized PaperResult objects

4. Gradio renders 10 recommendation cards
```

---

## 13. System Design Deep Dive

### Why Hybrid Retrieval (not just FAISS)?

| Scenario | BM25 wins | FAISS wins |
|----------|-----------|------------|
| Exact author name | "Vaswani 2017" finds the transformer paper | Might not — name not in embedding space |
| New terminology | "Mamba SSM 2024" → in abstracts | Might find by semantic context |
| Synonyms | "neural nets" and "deep learning" treated differently | Same embedding region |
| Domain jargon | Exact "EfficientViT" match | Might cluster with other ViT variants |

RRF gives you both. Papers that appear high in both lists (exact keyword match AND semantically relevant) rank highest. This is empirically the best-performing approach for diverse information needs.

### Why LambdaRank over BPR/NCF?

| Model | Problem |
|-------|---------|
| BPR (Bayesian Personalized Ranking) | Pairwise; can't incorporate side features |
| NCF (Neural Collaborative Filtering) | Needs rich interaction matrix; cold start issues |
| Matrix Factorization | Same as NCF; no side features |
| LambdaRank | Listwise; incorporates arbitrary features; trains on sparse data |

Scholar's training data is sparse: ~204K users, each with ~23 interactions on average. LambdaRank with tabular features handles this well — it doesn't need many examples per user.

### Why Mult-VAE over MF/ALS?

| Model | Problem |
|-------|---------|
| Matrix Factorization / ALS | Point estimate of user factor; no uncertainty |
| SVD++ | Requires explicit ratings; Scholar has only binary clicks |
| Mult-VAE | Probabilistic; handles implicit feedback; multinomial likelihood fits click data |

Mult-VAE is specifically designed for implicit feedback (clicks), which is all Scholar has. The stochastic latent code z provides a form of uncertainty — useful for exploration.

### Why QLoRA over full fine-tuning or RAG?

| Approach | Problem |
|----------|---------|
| Full fine-tuning Phi-3 | 7.6GB VRAM minimum for fp16; RTX 2050 has 4GB |
| RAG (retrieval-augmented generation) | Doesn't learn domain style; no level adaptation |
| Prompt engineering only | No adaptation to Scholar's structured output format |
| QLoRA | 3.4GB VRAM; fine-tunes output format + domain style; no information retrieval needed |

Scholar's use case is not knowledge-intensive (the paper abstract is in the prompt) — it's format-intensive (structured SUMMARY/GLOSSARY/WHY output). QLoRA fine-tunes the model to follow this format reliably.

### Known Limitations and Future Work

1. **bm25_score=0.0 in LTR**: The BM25 score isn't passed through to `extract_features`. Fix: pass `bm25_score` from the candidate scores dict.

2. **User embedding via centroid**: `np.mean(history_embeddings)` is a crude user embedding. The trained Mult-VAE provides a better one when the system has real interaction data.

3. **Cold start for new users**: A new user with no history gets recommendations based on a zero user embedding → FAISS returns arbitrary popular papers. Fix: add popularity-based cold-start recommendations (most-cited recent papers in selected categories).

4. **Phi-3 CPU inference (60-120s)**: Users without GPU wait over a minute for explanations. Fix: pre-generate explanations asynchronously, cache in DB, serve instantly.

5. **No real-time index updates**: Adding a new ArXiv paper requires rebuilding the FAISS index (30-minute job). Fix: FAISS's `IDMap` supports online adds, or use a two-index strategy (static index + small delta index merged periodically).

---

## 14. API Schemas Deep Dive — `scholar/api/schemas.py`

Pydantic models are the contract between the API and its clients. Every field has a reason.

### 14.1 PaperResult

```python
class PaperResult(BaseModel):
    id: str               # ArXiv ID "2301.00001" — used to build arxiv.org/abs/{id} link
    title: str
    abstract: str
    authors: str          # comma-separated, from DB (truncated to 10 at ingest time)
    categories: str       # space-separated "cs.LG cs.AI stat.ML"
    citation_count: int = 0
    score: float          # either RRF score (search) or LTR/MMR score (recommend)

    # Generation fields — only populated if /explain was called
    summary: Optional[str] = None
    glossary: List[str] = Field(default_factory=list)
    why_recommended: Optional[str] = None
```

The generation fields (`summary`, `glossary`, `why_recommended`) are `Optional` with `None` defaults because the search and recommend endpoints don't run Phi-3 — they return raw paper data. Only the `/explain/{paper_id}` endpoint populates these.

`Field(default_factory=list)` not `Field(default=[])` — mutable defaults in Python are shared across all instances. Using `default_factory` means each `PaperResult` gets its own list.

### 14.2 SearchRequest

```python
class SearchRequest(BaseModel):
    query: str                                      # free-text query
    user_id: Optional[str] = None                   # UUID — if None, no personalization
    level: Literal["undergrad", "grad", "researcher"] = "grad"
    diversity_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=50)
```

`Literal["undergrad", "grad", "researcher"]` — Pydantic validates that the string is one of these three values at parse time, before the route handler runs. Any other value returns a 422.

`ge=0.0, le=1.0` on `diversity_lambda` — Pydantic Field constraints. `ge` = greater-or-equal, `le` = less-or-equal. MMR requires λ ∈ [0, 1].

`ge=1, le=50` on `top_k` — prevents requesting 0 results (meaningless) or 10,000 results (would force full DB scan).

### 14.3 ExplainRequest

```python
class ExplainRequest(BaseModel):
    user_id: Optional[str] = None
    level: Literal["undergrad", "grad", "researcher"] = "grad"
    why: str = "This paper matched your search query."
```

The `why` field is the explanation of why the paper was recommended — passed through to the Phi-3 prompt template. The default makes the explain endpoint work without context. When called from the recommend route, `why` would be something like "This paper is highly relevant to your past reading on transformer architectures."

### 14.4 UserHistoryItem

```python
class UserHistoryItem(BaseModel):
    user_id: str
    paper_id: str
```

Minimal schema — the route is `POST /user/{user_id}/history` so `user_id` is in the URL path, but it's also in the body. The route handler uses the path parameter. The body field is redundant — this is a minor design inconsistency in the current codebase (the `user.py` route uses `body.paper_id` but ignores `body.user_id`).

---

## 15. Ingestion Pipeline — Complete Deep Dive

### 15.1 ArXiv Loader — `scholar/ingest/arxiv_loader.py`

The ArXiv Kaggle dataset is a JSON-lines file (~4GB uncompressed). Each line is one paper's metadata.

**Category filter:**

```python
TARGET_CATEGORIES = {"cs.", "stat.ML", "stat.AP", "math.ST"}

def _matches_category(categories: str) -> bool:
    for cat in TARGET_CATEGORIES:
        if cat in categories:
            return True
    return False
```

`"cs."` as a prefix matches all cs.* categories: cs.LG, cs.AI, cs.CV, cs.CL, cs.NE, etc. — 40+ categories in one check. `"stat.ML"` is a specific string (stats ML isn't a prefix family). This filter reduces 2M+ ArXiv papers to ~400K CS/ML/stats papers.

**Author name parsing:**

```python
authors_parsed = obj.get("authors_parsed", [])
if authors_parsed:
    author_names = []
    for a in authors_parsed:
        parts = [p.strip() for p in a if p.strip()]
        author_names.append(" ".join(reversed(parts[:2])))
    authors = ", ".join(author_names[:10])
```

ArXiv's `authors_parsed` field is a list of `[last, first, suffix]` arrays. `reversed(parts[:2])` converts `["Vaswani", "Ashish"]` → `"Ashish Vaswani"` (first last format). Takes only the first 10 authors to prevent extremely long strings.

**Upsert — `ON CONFLICT (id) DO NOTHING`:**

```python
cur.executemany(
    """
    INSERT INTO papers (id, title, abstract, authors, categories, update_date, citation_count)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
    """,
    rows,
)
```

`ON CONFLICT (id) DO NOTHING` — if a paper with this ArXiv ID already exists, skip it silently. This makes the loader idempotent: running it twice doesn't create duplicates or fail. No `DO UPDATE SET` because we don't want to overwrite citation counts that may have been enriched from other sources.

**Why psycopg2 for ingestion (not asyncpg)?**

`executemany` with a cursor is the fastest way to do bulk inserts in Python — it uses prepared statements and sends all rows in one roundtrip. asyncpg's equivalent (`copy_records_to_table`) is slightly faster for massive loads, but psycopg2's `executemany` is simpler and fast enough at batch_size=500.

**Batch size = 500:**

Each batch = one `BEGIN` + one `COMMIT`. With batch_size=500 and 400K papers: 800 transactions. Each transaction has overhead (~1ms for commit), so: 800 × 1ms = 0.8s overhead. The actual insert time dominates. Larger batches = fewer commits but more memory per batch. 500 is a good trade-off.

---

### 15.2 CiteULike Loader — `scholar/ingest/citeulike_loader.py`

CiteULike-A is an academic social bookmarking dataset: 5,551 users × 16,980 papers = ~204K interactions.

**Dataset format:**
```
users.dat:   (each line = one user's bookmarks)
    14 287 543 1029 ...    # space-separated 0-based paper indices
    
papers.txt:  (each line = one paper)
    "The Title of the Paper"
    "Another Paper Title"
```

**The title matching problem:**

CiteULike paper indices (0..16979) don't directly map to ArXiv IDs. The loader must match CiteULike paper titles to ArXiv papers in the DB by title string comparison.

```python
def _match_papers_to_db(titles: list[str], session) -> dict[int, str]:
    exact_lookup: dict[str, str] = {}
    for pid, title in db_papers:
        exact_lookup[title.lower().strip()] = pid

    mapping: dict[int, str] = {}
    for idx, title in enumerate(titles):
        norm = title.lower().strip()
        if norm in exact_lookup:
            mapping[idx] = exact_lookup[norm]   # citeulike_idx → arxiv_id
```

This is a simple exact lowercase match. Match rate: ~30-60% (many CiteULike papers are not in Scholar's ArXiv subset). The non-matched papers are silently skipped.

**User ID stability:**

```python
# Load user mapping cache so reruns are reproducible
mapping_file = data_dir / "user_id_mapping.json"
user_id_mapping: dict[str, str] = {}
if mapping_file.exists():
    with open(mapping_file) as f:
        user_id_mapping = json.load(f)
```

CiteULike users are indexed 0..5550. Each run, the loader generates UUID4s for new users and saves the mapping to `user_id_mapping.json`. On reruns, the same CiteULike user gets the same UUID — preventing duplicate users and allowing incremental loads.

**Idempotent inserts:**

```python
existing_hist = session.exec(
    select(UserHistory).where(
        UserHistory.user_id == db_user_id,
        UserHistory.paper_id == db_paper_id,
    )
).first()
if existing_hist is None:
    session.add(UserHistory(...))
```

The explicit existence check is slower than `INSERT ... ON CONFLICT DO NOTHING` (one query per potential duplicate vs. one INSERT per row). This was chosen for simplicity and correctness with SQLModel's ORM layer.

**Commit cadence:**

```python
if citeulike_uid % 500 == 0:
    session.commit()
```

Commit every 500 users (not every user). Each commit flushes the transaction to disk. With 5,551 users: ~11 commits. The final `session.commit()` after the loop handles the remainder.

---

### 15.3 Embeddings Builder — `scholar/ingest/embeddings.py`

The embedding pipeline runs after arxiv_loader. It reads all papers from DB, encodes them with sentence-BERT, writes a numpy array, and updates the pgvector column.

**Text format for encoding:**

```python
texts = [f"{title} {abstract}" for _, title, abstract in papers]
```

Concatenating title and abstract (not just abstract). The title often contains the key concepts in condensed form — including it biases the embedding toward the paper's main topic. Title-only embeddings are too short; abstract-only misses the crisp topic statement.

**Batch encoding:**

```python
embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)
```

batch_size=512 fills the GPU. RTX 2050 has 4GB VRAM. MiniLM is ~80MB; 512 abstracts × 512 tokens × 4 bytes = ~512MB activations. At 4GB, this is comfortable. Larger batches don't help (already GPU-bound on matmul).

**Timing:** ~200K papers × 100 tokens avg / (14,000 sentences/sec) ≈ 1,430 seconds ≈ 24 minutes on GPU.

**Dual write — numpy file + pgvector:**

```python
np.save("./data/embeddings.npy", embeddings)        # for FAISS index building
# ...
await conn.execute("UPDATE papers SET embedding=$2 WHERE id=$1", paper_id, emb.tolist())
```

Two destinations: `embeddings.npy` is the source for FAISS index building (script reads this file once). The pgvector column in Postgres enables SQL-level vector similarity queries (`<=>` operator) — not used in the hot path but available for experiments and analytics.

---

### 15.4 FAISS Index Building — `scripts/build_index.py`

HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm.

**The NSW (Navigable Small World) idea:**

Consider a graph where each node (paper vector) connects to its K nearest neighbors. To find the nearest neighbor of a query:
1. Start at a random entry node
2. Check all of its neighbors — is any of them closer to the query than the current node?
3. If yes, move to that neighbor. Repeat.
4. Stop when no neighbor is closer than the current node (local minimum = approximate NN)

Problem: NSW can get stuck in local minima, especially in high-dimensional spaces.

**HNSW fixes this with hierarchy:**

Layer 0: all vectors + connections to ~M nearest neighbors (M=32 in Scholar)
Layer 1: random subset (~1/e of layer 0) + longer-range connections
Layer 2: smaller subset + even longer-range connections
...

Search starts at the top layer (sparse, long-range) and descends to layer 0 (dense, short-range). This mimics a skip list: use highway connections to get close quickly, then local connections to find the exact neighbor.

```python
index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
#                            ↑    ↑
#                          dim   M (connections per node)
index.hnsw.efConstruction = 200  # beam width during build
```

`M=32`: each node stores 32 edges at layer 0. Higher M = better recall but more memory and slower build. M=32 is standard for 384-dim vectors.

`efConstruction=200`: the candidate queue size during graph construction. Higher = more accurate graph = better recall at query time, but slower build. 200 is a good default.

**Memory:** 200K vectors × 384 floats × 4 bytes + HNSW graph ~= 307MB + ~200MB = ~500MB on disk.

**Query time:** efSearch (set at query time, default=16) controls recall/speed trade-off. efSearch=16 gives ~99% recall at ~1ms per query. Scholar uses defaults.

---

## 16. Training Pipelines — Complete Deep Dive

### 16.1 Mult-VAE Training — `scholar/recsys/train_vae.py`

**Interaction matrix construction:**

```python
def _build_interaction_matrix(
    user_papers: dict[str, list[str]],
    all_paper_ids: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    paper_id_to_idx = {pid: i for i, pid in enumerate(all_paper_ids)}
    user_ids = sorted(user_papers.keys())

    # Dense boolean matrix: rows=users, cols=papers
    matrix = np.zeros((len(user_ids), len(all_paper_ids)), dtype=np.float32)
    for u_idx, uid in enumerate(user_ids):
        for pid in user_papers[uid]:
            p_idx = paper_id_to_idx.get(pid)
            if p_idx is not None:
                matrix[u_idx, p_idx] = 1.0

    return matrix, user_ids, all_paper_ids
```

With 204K users × 16,980 papers: matrix size = 204K × 16,980 × 4 bytes = ~13.8GB — too large for RAM. In practice, Scholar uses only the matched papers (~5,000-8,000 matched from CiteULike), reducing to 204K × 8K × 4 bytes ≈ 6.5GB. Still large — in production, this would use scipy sparse matrices.

**Training loop with KL annealing:**

```python
for epoch in range(num_epochs):
    beta = min(0.2, epoch / 20 * 0.2)     # 0→0.2 over 20 epochs
    model.train()

    for batch_start in range(0, n_train, batch_size):
        batch = torch.FloatTensor(train_matrix[batch_start:batch_start+batch_size])
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = criterion(recon, batch, mu, logvar)  # criterion uses self.beta
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation: NDCG@10 on held-out users
    model.eval()
    with torch.no_grad():
        val_recon, val_mu, val_logvar = model(val_tensor)
        # Top-K recommendation from reconstruction probabilities
        val_ndcg = _ndcg_at_k(val_recon, val_tensor, k=10)

    if val_ndcg > best_ndcg:
        best_ndcg = val_ndcg
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
```

Gradient clipping (`max_norm=1.0`) prevents exploding gradients — common with the KL term early in training when logvar can be very negative.

**NDCG@K for reconstruction evaluation:**

```python
def _ndcg_at_k(recon: torch.Tensor, true: torch.Tensor, k: int = 10) -> float:
    # recon: (batch, vocab) log-softmax probabilities
    # true: (batch, vocab) binary interactions
    # For each user, rank papers by recon probability, compute NDCG
    top_k = torch.argsort(recon, dim=1, descending=True)[:, :k]
    ...
```

The key difference from retrieval NDCG: "relevance" here is the binary true interaction vector. A perfect reconstruction would rank all interacted papers in the top K.

**Model checkpointing:**

```python
torch.save(best_state, "data/mult_vae.pt")
# Also save vocab mapping
with open("data/vae_vocab.json", "w") as f:
    json.dump({"paper_ids": all_paper_ids}, f)
```

The vocab mapping (`vae_vocab.json`) is critical: it maps paper indices (0..vocab_size) to ArXiv IDs. Without it, the model weights are useless (can't interpret the output probabilities).

---

### 16.2 LTR Training Data — `scholar/recsys/ltr.py`

**Feature extraction — the most important function:**

```python
def extract_features(
    user_emb: np.ndarray,
    candidates: list[tuple[str, float]],
    paper_meta: dict[str, Paper],
    paper_embeddings: dict[str, np.ndarray],
    history_cats: list[str],
) -> tuple[np.ndarray, list[str]]:

    rows = []
    candidate_ids = []

    for paper_id, faiss_score in candidates:
        paper = paper_meta.get(paper_id)
        if paper is None:
            continue

        # Feature 1: faiss_score — cosine similarity from FAISS query
        # Range: typically 0.85-0.99 for top-200 results
        f_faiss = float(faiss_score)

        # Feature 2: bm25_score — KNOWN GAP: always 0.0
        f_bm25 = 0.0

        # Feature 3: citation_count — proxy for paper quality/impact
        f_citations = float(paper.citation_count)

        # Feature 4: days since publication (recency signal)
        if paper.update_date:
            days = (datetime.now().date() - paper.update_date).days
            f_days = float(max(0, days))
        else:
            f_days = 365.0  # assume 1 year old if unknown

        # Feature 5: category overlap — Jaccard similarity
        # "how many of user's history categories does this paper share?"
        paper_cats = set(paper.categories.split()) if paper.categories else set()
        user_cats: set[str] = set()
        for cat_str in history_cats:
            user_cats.update(cat_str.split())
        if user_cats and paper_cats:
            f_cat = len(paper_cats & user_cats) / len(paper_cats | user_cats)  # Jaccard
        else:
            f_cat = 0.0

        # Feature 6: cosine similarity between user embedding and paper embedding
        paper_emb = paper_embeddings.get(paper_id)
        if paper_emb is not None and user_emb is not None:
            denom = (np.linalg.norm(user_emb) * np.linalg.norm(paper_emb))
            f_user_paper = float(np.dot(user_emb, paper_emb) / denom) if denom > 0 else 0.0
        else:
            f_user_paper = 0.0

        # Feature 7: is_in_history (binary: 1.0 if already read, 0.0 otherwise)
        # This should always be 0 for candidates (we filter history papers out)
        # but it's a safety net and useful for training data with mixed labels
        f_in_history = 0.0  # set by caller if needed

        rows.append([f_faiss, f_bm25, f_citations, f_days, f_cat, f_user_paper, f_in_history])
        candidate_ids.append(paper_id)

    return np.array(rows, dtype=np.float32), candidate_ids
```

**Feature importance at inference (expected LightGBM):**

Since `bm25_score` is always 0.0, LightGBM will give it near-zero importance. The dominant features are expected to be:
1. `user_paper_cosine` — direct relevance to user's interests
2. `faiss_score` — semantic similarity to query
3. `category_overlap` — topic alignment
4. `citation_count` — paper quality signal
5. `days_since_pub` — recency (depends on user preference)

**Training data construction:**

```python
for uid, pos_papers in user_papers.items():
    # Positive examples = papers the user clicked (relevance=3)
    pos_list = [p for p in pos_papers if p in paper_meta]

    # Negative examples = random unclicked papers (relevance=0)
    neg_pool = [p for p in all_paper_ids if p not in pos_papers]
    n_neg = min(len(pos_list) * 4, len(neg_pool), 20)  # 1:4 ratio, max 20 negatives
    neg_list = list(rng.choice(neg_pool, n_neg, replace=False))

    all_papers = pos_list + neg_list
    labels = [3] * len(pos_list) + [0] * len(neg_list)
    group_size = len(all_papers)   # all papers for this user form one "query group"
```

The 1:4 positive:negative ratio is standard in LTR. Too few negatives: model doesn't learn the boundary. Too many negatives: the gradient is dominated by negatives, degrading positive signal.

---

### 16.3 QLoRA Fine-Tuning — `scholar/generation/train_lora.py`

**The full training setup:**

```python
LORA_RANK = 8
LORA_ALPHA = 16
SEQ_LEN = 768
BATCH_SIZE = 1
GRAD_ACCUM = 8   # effective batch = 1 × 8 = 8
EPOCHS = 3
LR = 2e-4

# 1. Load base model in 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, ...)

# 2. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
# ^ enables gradient checkpointing, casts LayerNorm to float32 for stability

# 3. Attach LoRA adapters
peft_config = LoraConfig(
    r=LORA_RANK,                        # rank of decomposition
    lora_alpha=LORA_ALPHA,              # scaling factor
    target_modules=["qkv_proj"],        # only QKV projection in Phi-3
    lora_dropout=0.05,
    bias="none",                        # don't adapt biases
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
# ^ inserts LoRA matrices alongside qkv_proj layers

# 4. SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt",        # field with the full Phi-3 formatted text
    max_seq_length=SEQ_LEN,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_8bit",       # 8-bit Adam: quantizes optimizer moments
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        bf16=True,
        gradient_checkpointing=True,    # trade compute for memory
    ),
)
trainer.train()
```

**LoRA math — why it works:**

A weight matrix W ∈ ℝ^{d×k} in a transformer has d×k parameters. After training a large model, fine-tuning experiments show that the update ΔW has low intrinsic rank — it can be well-approximated by a low-rank matrix.

LoRA decomposes: ΔW = B × A where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d, k)

During forward pass: h = (W₀ + ΔW)x = W₀x + BAx

During training: W₀ is frozen (4-bit, no gradient), B and A are trained in bf16.

Scaling: ΔW is scaled by α/r before adding. With α=16, r=8: scaling factor = 2.0. This compensates for the initialization of A (random Gaussian) and B (zeros) — at init, ΔW = 0, so the model starts exactly at the pre-trained weights.

**Memory breakdown for fine-tuning on RTX 2050:**

| Component | Memory |
|-----------|--------|
| Phi-3 base weights (4-bit NF4) | ~1.9 GB |
| LoRA A, B matrices (bf16) | ~6 MB |
| Optimizer states (paged_adamw_8bit) | ~12 MB (pageable!) |
| Activations (batch=1, seq=768, grad_ckpt) | ~0.8 GB |
| CUDA runtime | ~0.3 GB |
| **Total** | ~3.0 GB |

`paged_adamw_8bit` uses NVIDIA's unified memory: optimizer states are 8-bit and can be paged to CPU RAM when GPU is full. This is the key to fitting fine-tuning in 4GB.

**`gradient_checkpointing=True`:** Instead of caching all intermediate activations for backprop (~2GB), recompute them during backward pass. ~33% slower but ~40% less VRAM.

---

### 16.4 Synthetic Data Generation — `scholar/generation/synth_data.py`

```python
def generate_synth_data(limit: int = 5000, why: str = "Recommended based on your interests.") -> None:
    papers = _load_papers(limit)    # (paper_id, title, abstract) from DB
    generator = Phi3Generator.get_instance()   # base model (no LoRA adapter)
    levels = ["undergrad", "grad", "researcher"]

    for pid, title, abstract in tqdm(papers):
        for level in levels:
            result = generator.generate(title=title, abstract=abstract, why=why, level=level)
            # result = {"summary": "...", "glossary": [...], "why": "..."}

            # Reconstruct the structured response text (as the model should output it)
            glossary_str = "\n".join(f"- {t}" for t in result.get("glossary", []))
            response_text = (
                f"SUMMARY: {result.get('summary', '')}\n"
                f"GLOSSARY:\n{glossary_str}\n"
                f"WHY: {result.get('why', '')}"
            )

            record = {
                "paper_id": pid,
                "level": level,
                "prompt": get_prompt(level=level, title=title, abstract=abstract, why=why),
                "response": response_text,
            }
            out.write(json.dumps(record) + "\n")
```

**Why self-distillation (base model → training data → fine-tuned model)?**

We don't have human-written gold summaries. The base Phi-3 already produces reasonable summaries — they just don't follow the SUMMARY:/GLOSSARY:/WHY: structure consistently.

By generating examples with the base model and fine-tuning on them:
1. The fine-tuned model reliably produces the structured format
2. The content quality is similar to the base model (no hallucination from bad gold data)
3. The level differentiation (undergrad/grad/researcher vocabulary) is learned

This is a form of **self-distillation** — the model teaches itself a specific format. The limitation: the fine-tuned model can't be better than the base at content, only at format. For Scholar's use case (format consistency matters more than content improvement), this is fine.

**Output format:**

```jsonl
{"paper_id": "2301.00001", "level": "undergrad", "prompt": "<|user|>\n...", "response": "SUMMARY: ...\nGLOSSARY:\n- term1\n- term2\nWHY: ..."}
{"paper_id": "2301.00001", "level": "grad", "prompt": "<|user|>\n...", "response": "..."}
{"paper_id": "2301.00001", "level": "researcher", "prompt": "<|user|>\n...", "response": "..."}
```

With `limit=5000` papers × 3 levels = 15,000 training examples. SFTTrainer reads this as a text dataset and tokenizes each `prompt` field.

---

## 17. Evaluation Framework

### 17.1 Retrieval Evaluation — `scholar/retrieval/eval.py`

**NFCorpus background:**

NFCorpus (Non-Factoid Medical Corpus) is a small IR benchmark with 323 queries and 3,633 documents. It's used for retrieval evaluation because:
1. It's freely available via `ir_datasets`
2. It has multi-level relevance judgments (0=not relevant, 1=possibly relevant, 2=relevant, 3=highly relevant)
3. Medical text is similar in vocabulary richness to scientific ArXiv abstracts

**NDCG@K — Normalized Discounted Cumulative Gain:**

```
DCG@K = Σᵢ₌₁ᴷ (2^relᵢ - 1) / log₂(i + 1)

IDCG@K = DCG of the ideal ranking (best possible)

NDCG@K = DCG@K / IDCG@K
```

Scholar's implementation uses linear gains (not exponential):

```python
def _dcg_at_k(relevances: list[float], k: int) -> float:
    gains = relevances[:k]
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    #          ↑                          ↑
    #   linear gain (not 2^rel)    position discount (i+2 because i is 0-indexed)
```

Linear gains: relevance=3 contributes 3x more than relevance=1. Exponential gains (2³-1=7 vs 2¹-1=1) would give even more weight to highly relevant docs. Scholar's linear choice is conservative.

The `+2` in `log₂(i+2)` is because enumerate is 0-indexed: position 0 → log₂(2)=1 (no discount), position 1 → log₂(3)≈1.58, etc. If you used `i+1`, position 0 would divide by log₂(1)=0 → infinite score.

**MRR@K — Mean Reciprocal Rank:**

```python
def _mrr_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if relevant.get(pid, 0) > 0:
            return 1.0 / rank   # reciprocal of the rank of FIRST relevant result
    return 0.0
```

MRR asks: "how high is the first relevant result?" MRR@10=0.5 means the first relevant result is typically at rank 2. Useful for navigational queries where the user just wants any one relevant document.

**MAP@K — Mean Average Precision:**

```python
def _map_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if relevant.get(pid, 0) > 0:
            hits += 1
            precision_sum += hits / rank   # precision AT this rank
    return precision_sum / min(num_relevant, k)
```

AP (Average Precision) is the area under the precision-recall curve. High MAP means both high precision (few irrelevant results at top) and high recall (many relevant results appear before rank K).

**Comparing the three metrics:**

| Metric | Penalizes | Rewards | Best for |
|--------|-----------|---------|---------|
| NDCG@K | Relevant docs at low ranks | Graded relevance | Ranked lists with multiple relevant docs |
| MRR@K | First relevant doc below rank 1 | High-precision navigation | "Give me one good result" |
| MAP@K | Any relevant doc below rank K | All-relevant recall | Comprehensive recall |

For Scholar's use case (showing 10 papers to a researcher), NDCG@10 is most informative.

**Latency tracking:**

```python
t0 = time.perf_counter()
results = retriever_fn(query_text, top_k=100)
lat = (time.perf_counter() - t0) * 1000   # ms

latencies.append(lat)
# ...
"lat_p50": float(np.percentile(latencies, 50)),   # median
"lat_p99": float(np.percentile(latencies, 99)),   # worst 1% case
```

p99 latency (not max) — max is misleading because it's vulnerable to one-off outliers (cold cache, OS scheduling). p99 captures the tail behavior that real users occasionally experience.

---

### 17.2 Recsys Evaluation — `scholar/recsys/eval.py`

**Evaluation protocol:**

For each test user (10% of all users):
1. Split their interactions: 80% train, 20% test (by paper order, not random)
2. Build user embedding from train papers only
3. Run recommendation pipeline → 10 papers
4. Measure: how many test papers appear in the 10 recommendations?

```python
# Per-user split
papers_list = sorted(all_papers)           # deterministic sort
n_user_train = max(1, int(0.8 * len(papers_list)))
train_papers = set(papers_list[:n_user_train])
test_papers = set(papers_list[n_user_train:])
```

**ILD — Intra-List Diversity:**

```python
def _ild(paper_ids: list[str], paper_embeddings: dict[str, np.ndarray]) -> float:
    embs = [paper_embeddings[pid] for pid in paper_ids if pid in paper_embeddings]
    dists = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dists.append(_cosine_dist(embs[i], embs[j]))   # cosine distance, not similarity
    return float(np.mean(dists))
```

ILD measures whether the recommended papers are about different topics. It's O(K²) on 10 papers = 45 pairwise distances. High ILD (close to 1.0) = recommendations are maximally diverse. Low ILD = all about the same topic.

ILD and NDCG@10 are in tension: a perfectly relevant list (all papers very similar to user's history) will have low ILD. MMR explicitly optimizes this trade-off.

**Ablation study design:**

```python
ablations = ["mmr", "ltr+mmr", "ltr+mmr+exploration"]
```

Three ablations answer three questions:
1. `mmr`: Does FAISS + MMR alone work? (baseline)
2. `ltr+mmr`: Does adding LambdaRank improve NDCG? (should significantly help)
3. `ltr+mmr+exploration`: Does ε-greedy hurt NDCG but increase discovery?

Expected results:
- `ltr+mmr` > `mmr` on NDCG (LTR better ranking)
- `ltr+mmr+exploration` ≈ `ltr+mmr` on NDCG (exploration adds noise)
- `ltr+mmr+exploration` > `ltr+mmr` on ILD (exploration adds diversity)

---

## 18. Deep Theory Appendix

### 18.1 Information Retrieval — The Probabilistic Model

BM25 is derived from the **Binary Independence Model (BIM)** of Robertson and Spärck Jones (1976). The core idea: model P(R=1 | d, q) — the probability that document d is relevant to query q.

**Derivation sketch:**

Using Bayes' rule and assuming term independence:

```
P(R|d,q) / P(NR|d,q) = Π_{t∈q} p(t∈d | R) × (1-p(t∈d | NR)) / 
                                  (1-p(t∈d | R)) × p(t∈d | NR)
```

Taking log and simplifying with Robertson-Spärck Jones weights:

```
RSJ_weight(t) = log [p(t|R)(1-p(t|NR))] / [(1-p(t|R))p(t|NR)]

Estimated as: log [(r + 0.5)/(R - r + 0.5)] / [(df - r + 0.5)/(N - df - R + r + 0.5)]
```

Where r = relevant docs containing t, R = total relevant docs, df = docs containing t, N = corpus size.

In the asymptotic case (R→0, r→0):

```
RSJ_weight(t) ≈ log (N - df + 0.5) / (df + 0.5)
```

This is exactly Scholar's IDF formula. BM25 adds the TF component empirically: Robertson (1994) showed that saturating TF (via k1) and length normalization (via b) empirically improve ranking without a clean probabilistic derivation.

**Why logarithm in IDF?**

Zipf's law: the frequency of a word is inversely proportional to its rank. Common words appear exponentially more often. The log compresses this scale so that "the" (df=200K) and "diffusion" (df=5K) don't have a 40:1 IDF ratio but rather log(200K/200K + 0.5) ≈ 0 vs log(200K/5K + 0.5) ≈ 3.7.

---

### 18.2 Variational Inference — ELBO Derivation

The Mult-VAE optimizes the Evidence Lower BOund (ELBO). Here's the derivation from first principles.

**Goal:** Given observed interactions x (which papers user clicked), maximize the log-likelihood log p(x).

```
log p(x) = log ∫ p(x|z) p(z) dz
```

This integral is intractable for neural networks (can't enumerate all z). **Variational inference** introduces an approximate posterior q(z|x) (the encoder):

```
log p(x) = log ∫ p(x|z) p(z) dz
         = log ∫ p(x|z) p(z) × q(z|x)/q(z|x) dz
         = log 𝔼_{q(z|x)}[p(x|z) p(z) / q(z|x)]
         ≥ 𝔼_{q(z|x)}[log p(x|z)] - D_KL(q(z|x) || p(z))   # Jensen's inequality
```

This lower bound is the ELBO:

```
ELBO(θ, φ; x) = 𝔼_{q_φ(z|x)}[log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))
                ↑ reconstruction term                ↑ regularization term
```

**For Mult-VAE specifically:**

- q_φ(z|x) = N(μ_φ(x), diag(σ²_φ(x))) — Gaussian encoder
- p(z) = N(0, I) — standard normal prior
- p_θ(x|z) = Multinomial(softmax(f_θ(z))) — multinomial decoder
- KL term has closed form: -0.5 × Σ (1 + log σ² - μ² - σ²)

The β-VAE adds a weight: ELBO_β = 𝔼[log p(x|z)] - β × D_KL(...). KL annealing starts β=0 (pure reconstruction) and increases to β=0.2.

**Why the ELBO is tight (gap analysis):**

```
log p(x) - ELBO = D_KL(q(z|x) || p(z|x))
```

The gap between the true log-likelihood and the ELBO equals the KL divergence between the approximate posterior q and the true posterior p(z|x). Minimizing the ELBO ≡ minimizing this gap ≡ making q as close as possible to the true posterior.

---

### 18.3 Transformer Attention — The Math Scholar Relies On

Scholar uses Phi-3 (transformer architecture) for generation and all-MiniLM-L6-v2 (transformer) for embeddings. Understanding attention is essential.

**Scaled dot-product attention:**

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Q ∈ ℝ^{n × d_k}  (query: n tokens, d_k dimensions)
K ∈ ℝ^{m × d_k}  (key:   m tokens, d_k dimensions)
V ∈ ℝ^{m × d_v}  (value: m tokens, d_v dimensions)
```

The `√d_k` scaling prevents the dot products from growing too large (which would push softmax into saturation where gradients vanish). For d_k=64 (typical in BERT-base), √64=8.

**Why QKᵀ works for relevance:**

Each query vector q_i (the representation of token i asking "what do I need?") is compared to all key vectors k_j ("what do I offer?") via dot product. High dot product = token j is relevant to token i's query. Softmax converts these to attention weights, and V (the content) is aggregated weighted by attention.

**Multi-head attention in Phi-3:**

Phi-3 uses MHA with d_model=3072, 32 heads, d_k=d_v=96. The `qkv_proj` layer (Scholar's LoRA target) is the W_Q, W_K, W_V matrices combined into one projection:

```
[Q, K, V] = X × W_qkv    where W_qkv ∈ ℝ^{3072 × (3 × 96 × 32)}
                                       = ℝ^{3072 × 9216}
```

That's why LoRA on `qkv_proj` has: A ∈ ℝ^{3072×8}, B ∈ ℝ^{8×9216}.

---

### 18.4 Sentence Transformers — How MiniLM-L6-v2 Was Trained

Scholar uses `all-MiniLM-L6-v2` for semantic embeddings. Understanding how it was trained helps understand its behavior.

**Bi-encoder architecture:**

The model takes a text input and produces a fixed-size embedding via mean pooling over all token representations:

```
embedding(text) = mean_pool(BERT(text))    # (n_tokens, 384) → (384,)
```

Mean pooling (vs [CLS] token): averages all token embeddings including non-special tokens. Empirically better for semantic similarity than [CLS] alone.

**Training via contrastive learning:**

MiniLM-L6-v2 was trained with Multiple Negatives Ranking Loss on 1 billion sentence pairs. For each anchor-positive pair (a, p):

```
Loss = -log( exp(sim(a,p)/τ) / Σ_{j=1}^{B} exp(sim(a,pⱼ)/τ) )
```

Where B = batch size, j indexes all other positives in the batch (in-batch negatives), τ = temperature.

In-batch negatives: every other positive pair in the batch serves as a negative. With batch=512: each anchor has 511 negatives. This is extremely efficient — one forward pass creates 512 training examples.

**Why L2 normalization before cosine:**

After training, embeddings are normalized to the unit sphere. Cosine similarity = dot product for unit vectors. FAISS's `METRIC_INNER_PRODUCT` on normalized vectors is exactly cosine similarity. Without normalization, longer texts with higher-magnitude embeddings would dominate.

---

### 18.5 LambdaRank — NDCG Optimization via Proxy Gradients

**Why direct NDCG optimization is hard:**

NDCG is not differentiable — it depends on ranks, which are integers. You can't compute ∂NDCG/∂score_i directly.

**LambdaRank's insight:** You don't need to compute ∂NDCG/∂score — you only need gradients that, when followed, would increase NDCG. LambdaRank defines gradients by looking at all pairs (i, j) where i is relevant and j is not:

```
λᵢⱼ = |ΔNDCG_ij| / (1 + exp(sᵢ - sⱼ))

- |ΔNDCG_ij|: how much NDCG would improve if we swapped i and j
- (1 + exp(sᵢ - sⱼ)): sigmoid weight — if sᵢ >> sⱼ, already correctly ranked, small gradient
```

The total gradient for document i:

```
λᵢ = Σⱼ∈D⁻ λᵢⱼ - Σⱼ∈D⁺ λᵢⱼ
```

(increase score of relevant docs, decrease score of irrelevant docs)

**Why LightGBM for LambdaRank?**

LightGBM uses LambdaRank as its gradient computation. Each tree is fit to the lambda gradients (not the labels directly). The key property: trees split on features that most decrease the sum of squared lambdas — automatically focusing on the most impactful feature splits.

**NDCG@K swap formula for Scholar (K=10, label scale 0-3):**

```
ΔNDCG_ij = |DCG@K_i_swapped - DCG@K_j_swapped| / IDCG@K

DCG@K contribution of doc at position p: label / log₂(p + 1)
```

If document i (label=3) is at position 8 and document j (label=0) is at position 2, swapping them:
- Current: 3/log₂(9) + 0/log₂(3) = 0.947
- After swap: 0/log₂(9) + 3/log₂(3) = 1.893
- |ΔDCG| = 0.946 (large — high-value swap)

---

### 18.6 NF4 Quantization — How Phi-3 Fits in 4GB

**Standard int4:** Divide the range [min_weight, max_weight] into 16 equal intervals. Map each weight to the nearest interval center. Loss: weights near zero (where most neural network weights cluster) are quantized coarsely.

**NF4 (Normal Float 4) insight:** Neural network weights after training follow a roughly normal distribution N(0, σ). Instead of equally spaced levels, NF4 uses quantile-based levels — more levels near zero (high density), fewer at extremes (low density):

```
NF4 levels: {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,
             0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
```

These are the 16 quantiles of N(0,1). Each weight is assigned to its nearest level. For a typical neural network weight distribution, NF4 has 0.25-0.35 bits lower perplexity than int4.

**Double quantization:** The quantization constants themselves (one per 64-weight block) are also quantized to int8. This saves an additional ~0.37 bits/parameter. For Phi-3-mini: 3.8B params × 0.37 bits / 8 ≈ 175MB saved.

**BitsAndBytesConfig in Scholar:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # quantile levels as above
    bnb_4bit_compute_dtype=torch.bfloat16, # dequantize to bf16 for matmul
    bnb_4bit_use_double_quant=True,       # quantize quantization constants
)
```

At inference: weights are stored as NF4 (4 bits), dequantized to bfloat16 just before each matrix multiply, result in bfloat16. Dequantization is fast (a lookup table on 4-bit indices) and happens on-chip.

---

### 18.7 Complete Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SCHOLAR DATA FLOWS                                  │
│                                                                          │
│  INGESTION (offline)                                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  arxiv-metadata.json                                                     │
│    │ arxiv_loader.py                                                     │
│    │  - filter: TARGET_CATEGORIES (cs.*, stat.ML, ...)                  │
│    │  - parse: authors, dates, clean text                                │
│    │  - upsert: INSERT ... ON CONFLICT DO NOTHING (batch=500)           │
│    ▼                                                                     │
│  papers table (Postgres)                                                 │
│    │ embeddings.py                                                       │
│    │  - encode: title + abstract → all-MiniLM-L6-v2 → 384-dim         │
│    │  - normalize: L2 to unit sphere                                    │
│    │  - write: embeddings.npy + UPDATE papers SET embedding=...        │
│    ▼                                                                     │
│  faiss.index + embeddings.npy                                            │
│    │ build_index.py                                                      │
│    │  - IndexHNSWFlat(384, M=32, METRIC_IP)                            │
│    │  - index.add(all_embeddings)                                       │
│    │  - faiss.write_index() + .ids.json                                │
│    ▼                                                                     │
│  data/faiss.index                                                        │
│                                                                          │
│  citeulike_loader.py                                                     │
│    │ - title-match CiteULike → DB papers                               │
│    │ - INSERT users + user_history                                      │
│    ▼                                                                     │
│  users + user_history tables                                             │
│                                                                          │
│  BM25 INDEX                                                              │
│  build_index.py --bm25-only                                              │
│    │ - fetch all papers from DB                                         │
│    │ - BM25.build(docs, doc_ids)                                        │
│    │ - pickle.dump → data/bm25.pkl                                      │
│    ▼                                                                     │
│  data/bm25.pkl                                                           │
│                                                                          │
│  TRAINING (offline)                                                      │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  train_vae.py                                                            │
│    │ - load user_history → interaction matrix (users × papers)         │
│    │ - MultVAE training (100 epochs, KL annealing, Adam)               │
│    │ - best checkpoint → data/mult_vae.pt                              │
│    ▼                                                                     │
│  data/mult_vae.pt                                                        │
│                                                                          │
│  ltr.py train                                                            │
│    │ - sample pos/neg papers per user (1:4 ratio)                      │
│    │ - extract_features (7 features)                                    │
│    │ - lgb.train(lambdarank, 100 rounds)                                │
│    │ → data/ltr_model.lgb                                              │
│                                                                          │
│  synth_data.py                                                           │
│    │ - fetch 5000 papers from DB                                       │
│    │ - generate 3 levels × 5000 = 15000 examples                      │
│    │ → data/synth_train.jsonl                                          │
│                                                                          │
│  train_lora.py                                                           │
│    │ - load Phi-3 (4-bit NF4)                                          │
│    │ - attach LoRA (r=8, alpha=16, qkv_proj)                          │
│    │ - SFTTrainer (3 epochs, paged_adamw_8bit)                         │
│    │ → data/lora_adapter/                                              │
│                                                                          │
│  SERVING (online)                                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  FastAPI startup (lifespan):                                             │
│    bm25.pkl → BM25 in memory (~500MB)                                   │
│    faiss.index → DenseRetriever in memory (~500MB)                      │
│    ltr_model.lgb → LTRModel in memory (~5MB)                           │
│    phi-3 (optional warmup) → Phi3Generator in VRAM (~1.9GB)            │
│                                                                          │
│  POST /search?query="diffusion models"                                  │
│    BM25.query(q, 200) → 200 (paper_id, bm25_score)                     │
│    DenseRetriever.query(q, 200) → 200 (paper_id, cosine_sim)           │
│    RRF(k=60) → 400→200 unique papers                                    │
│    DB: SELECT * FROM papers WHERE id IN (...)                           │
│    adaptive_lambda(user_history) → λ ∈ [0.3, 0.7]                      │
│    MMR(top-200, λ, k=10) → 10 final papers                             │
│    BackgroundTask: INSERT click_logs                                     │
│    → SearchResponse(10 PaperResults)                                    │
│                                                                          │
│  GET /recommend/{user_id}                                               │
│    DB: SELECT user_history WHERE user_id=...                            │
│    FAISS.reconstruct(id_to_idx[pid]) × N → user_emb = mean(embs)       │
│    DenseRetriever.query_by_vector(user_emb, 200) → 200 candidates       │
│    LTR.score(user_emb, candidates, papers, embs, cats) → 50 papers     │
│    adaptive_lambda → λ                                                  │
│    MMR(top-50, λ, k=10) → 10 papers                                    │
│    epsilon_greedy_inject(ε=0.1) → maybe swap 1-2 tail papers           │
│    → SearchResponse(10 PaperResults)                                    │
│                                                                          │
│  POST /explain/{paper_id}                                               │
│    DB: SELECT * FROM papers WHERE id=...                                │
│    Phi3Generator.generate(title, abstract, why, level)                  │
│      → tokenize → 4-bit forward pass → decode                          │
│    → {summary, glossary, why}                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 19. Performance Characteristics and Bottlenecks

### 19.1 Request Latency Budget

| Step | Typical latency | Notes |
|------|----------------|-------|
| BM25.query (200K docs) | 10-30ms | Dict lookups, sorting |
| FAISS.search (200K vecs) | 2-5ms | HNSW ANN graph traversal |
| MiniLM encode (1 query) | 5-10ms | GPU; 50-100ms on CPU |
| DB fetch (200 papers) | 10-20ms | SELECT with IN clause |
| LTR.score (50 papers) | 1-2ms | LightGBM inference is fast |
| MMR (top-50, K=10) | 1-3ms | 50×50 cosine matrix |
| epsilon_greedy_inject | <1ms | Python list ops |
| **Total /search** | **30-70ms** | GPU inference only |
| **Total /recommend** | **40-80ms** | GPU inference only |
| **Phi-3 generate (GPU)** | **2-5s** | 400 tokens at ~80 tok/s |
| **Phi-3 generate (CPU)** | **60-120s** | 400 tokens at ~3-4 tok/s |

### 19.2 Memory Footprint

| Component | RAM | VRAM |
|-----------|-----|------|
| BM25 index (200K docs) | ~500MB | 0 |
| FAISS index (200K × 384) | ~500MB | 0 |
| id_to_idx cache | ~8MB | 0 |
| LTR model | ~5MB | 0 |
| Postgres (shared_buffers) | ~256MB | 0 |
| **Phi-3 + LoRA (4-bit NF4)** | 0 | ~1.9GB |
| **Activations (inference)** | 0 | ~0.8GB |
| Total (no Phi-3) | ~1.3GB RAM | 0 |
| Total (with Phi-3) | ~1.3GB RAM | ~2.7GB VRAM |

RTX 2050 (4GB) has ~1.3GB headroom for Phi-3 after system VRAM usage.

### 19.3 Throughput

| Operation | Throughput |
|-----------|-----------|
| Paper embedding (GPU) | 14,000 sentences/sec |
| ArXiv ingestion | ~5,000 papers/sec (batch_size=500) |
| BM25 index build | ~200K docs/minute |
| FAISS index build | ~50K vectors/minute (with efConstruction=200) |
| VAE training (epoch) | ~20 minutes (200K users, batch=256) |
| QLoRA fine-tuning (epoch) | ~4 hours (15K examples, seq_len=768) |

---

## 20. Connecting Everything: The Scholar Mental Model

Think of Scholar as a **funnel with personality**:

```
200K papers in DB
    ↓ [hybrid retrieval: BM25 + FAISS → RRF]
200 topically relevant candidates
    ↓ [LambdaRank: personalize to this user's reading history]
50 papers ranked by personal relevance
    ↓ [MMR: ensure no topic is over-represented]
10 diverse, relevant papers
    ↓ [ε-greedy: inject 1-2 adjacent-topic explorers]
10 final recommendations
    ↓ [Phi-3: explain at right technical depth]
User reads, clicks, Mult-VAE learns, cycle improves
```

**The feedback loop:**
1. User reads paper → `POST /user/{id}/history`
2. User history feeds `user_emb` in `/recommend`
3. User emb feeds LTR feature `user_paper_cosine`
4. Clicks logged in `click_logs` → future Mult-VAE retraining
5. Retrain VAE → better user embeddings → better `user_paper_cosine` feature
6. LTR retraining with new embeddings → better ranking

**Why each algorithm was chosen vs. the obvious alternative:**

| Component | Scholar uses | Obvious alternative | Reason rejected |
|-----------|-------------|--------------------|-|
| Tokenizer | `re.split(r"[^a-z0-9]+")` | NLTK/spaCy | 80MB+ dependency for marginal gain in scientific text |
| BM25 | From scratch | `rank_bm25` library | CLAUDE.md constraint; also proves understanding |
| Dense encoder | all-MiniLM-L6-v2 | OpenAI ada-002 | Free, local, no API cost |
| Vector index | FAISS HNSW | pgvector HNSW | FAISS is 3-5x faster for batch queries; pgvector is SQL-native (kept for analytics) |
| Score fusion | RRF | Weighted linear | Scale-invariant; no tuning; empirically robust |
| User model | Mult-VAE | Matrix factorization | Implicit feedback native; probabilistic latent code |
| Reranker | LightGBM LambdaRank | Neural reranker (BERT) | 0.1ms vs 500ms inference; tabular features composable |
| Diversity | MMR | Greedy clustering | MMR is O(K²) and explicitly balances relevance/diversity |
| Exploration | ε-greedy | Thompson sampling | Simplest correct solution; Thompson needs uncertainty estimates |
| LLM | Phi-3-mini | GPT-4o | Free, local, 4GB VRAM fits; format fine-tuning sufficient |
| Fine-tuning | QLoRA | Full fine-tuning | 4GB VRAM; 0.08% trainable params sufficient for format learning |
| API | FastAPI | Flask/Django | Native async; auto OpenAPI docs; Pydantic integration |
| ORM | SQLModel | SQLAlchemy | SQLModel = SQLAlchemy + Pydantic, less boilerplate |
| DB driver | asyncpg | psycopg2 | 3-5x faster; binary protocol; native async |
| Vector storage | pgvector | Milvus/Weaviate | Eliminates a separate vector DB service; Postgres-native |
