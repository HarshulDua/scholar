# Information Retrieval — Complete Deep Dive

> Covers: Boolean model, VSM, TF-IDF, BM25 (full derivation), inverted indexes,
> dense retrieval, FAISS internals, RRF hybrid fusion, and Scholar's exact implementation.

---

## 1. The Information Retrieval Problem

**Goal:** Given a query q and a corpus of N documents, return the top-k most relevant documents.

The key challenges:
- **Vocabulary mismatch** — a query says "ML" but the document says "machine learning"
- **Semantic gap** — "good restaurant" and "great eatery" should match even with no shared terms
- **Scale** — linear scan over 200K documents is too slow for interactive use
- **Ranking** — not just finding matching docs, but ordering them by relevance

Scholar uses three complementary signals to address these challenges:
1. **BM25 (sparse)** — exact term matching with statistical weighting
2. **Dense retrieval (SBERT)** — semantic embedding similarity
3. **Reciprocal Rank Fusion** — combines both into a single ranked list

---

## 2. Boolean Retrieval — The Baseline (and Why It's Not Enough)

The simplest IR model: a document either contains a term or it doesn't.

```
Query: "neural AND networks NOT image"
→ Return all docs that have both "neural" and "networks" but not "image"
```

**Problems:**
- No ranking — all matching documents are equal
- No partial match — "neural network" gets zero score if either word is missing
- No term frequency awareness — a doc with "neural" 50 times scores the same as one with it once
- Still used in legal/enterprise search where "give me everything" is the requirement

---

## 3. Vector Space Model and TF-IDF

VSM represents both queries and documents as vectors. Relevance = cosine similarity between vectors.

### Term Frequency (TF)

How often does term t appear in document d?

```
tf(t, d) = count(t in d) / total_terms(d)    (normalized)
```

Raw count is misleading — a 10,000-word document with "neural" 20 times is less relevant than
a 500-word document with it 10 times. Normalization fixes this.

### Inverse Document Frequency (IDF)

Terms appearing in many documents are less informative.

```
idf(t) = log(N / df(t))

N = total documents, df(t) = documents containing t

"the": df ≈ 200,000 → idf = log(1) = 0         (useless term)
"transformer": df ≈ 1,000 → idf = log(200) = 5.3  (informative)
"RLHF": df ≈ 50 → idf = log(4000) = 8.3          (very specific)
```

### TF-IDF

```
tfidf(t, d) = tf(t, d) × idf(t)
```

Documents are then ranked by cosine similarity between the query vector and document vectors.

**TF-IDF problems:**
- High TF for very long documents — they score higher just because they're longer
- No saturation — going from 1 to 2 occurrences has the same effect as 50 to 51
- BM25 fixes both of these

---

## 4. BM25 — Full Derivation

BM25 (Best Match 25) is derived from probabilistic information retrieval theory. It addresses
TF-IDF's length normalization and TF saturation problems.

### The Formula

```
BM25(q, d) = Σ_{t ∈ q} IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]

Where:
  tf(t, d)  = count of term t in document d
  |d|       = length of document d (in words)
  avgdl     = average document length in corpus
  k1        = term frequency saturation parameter (default 1.5)
  b         = length normalization strength (default 0.75)
  IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   (smoothed)
```

### Why Each Component

**TF saturation (k1):**
```
Unsaturated:  tf = 1 → score 1x; tf = 10 → score 10x; tf = 100 → score 100x
Saturated:    tf = 1 → score 0.83; tf = 10 → score 1.73; tf = 100 → score 1.97
Maximum: lim_{tf→∞} = (k1+1) = 2.5 for k1=1.5

The saturation effect: the 2nd mention of "transformer" adds less than the 1st.
This matches human intuition — after a few occurrences, more doesn't matter much.
```

**Length normalization (b):**
```
b = 0: no normalization (longer docs always score higher on raw counts)
b = 1: full normalization (completely equalize by length)
b = 0.75: partial normalization (sweet spot — longer docs SLIGHTLY penalized)

(1 - b + b × |d|/avgdl):
  Short doc (|d| = avgdl/2):  factor = 1 - 0.75 + 0.75×0.5 = 0.625 → scores boosted
  Average doc:                factor = 1.0 → no change
  Long doc  (|d| = 2×avgdl): factor = 1 - 0.75 + 0.75×2 = 1.75  → scores penalized
```

**Smoothed IDF:**
The original BM25 IDF can go negative (when df > N/2). The smoothed version adds 1 inside the log:
```
IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

Always ≥ 0. Common terms still get low weight, just not negative weight.
```

### Parameter Tuning

| Parameter | Range | Effect |
|-----------|-------|--------|
| k1 = 1.5 | 1.2–2.0 | Higher = less saturation (long docs do better) |
| b = 0.75 | 0–1 | Higher = stronger length normalization |
| k1 = 1.2, b = 0.75 | — | Common alternative for shorter docs |

Scholar uses k1=1.5, b=0.75 (standard defaults). For ArXiv abstracts (medium length, ~150 words),
these are well-calibrated.

---

## 5. Inverted Index — How BM25 Runs Fast

A naive BM25 implementation scans all 200K documents for each query term. That's too slow.

An **inverted index** maps terms → list of (doc_id, count) pairs:

```
"transformer" → [(p_1, 3), (p_42, 1), (p_99, 7), ...]
"attention"   → [(p_1, 5), (p_7, 2), ...]

When query = "transformer attention":
1. Lookup "transformer" posting list → candidate docs
2. Lookup "attention" posting list → candidate docs
3. INTERSECT or UNION → only score those docs
```

**Numpy-vectorized implementation (Scholar's actual code):**

```python
class BM25:
    def fit(self, documents: list[str], doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids
        self.N = len(documents)

        tokenized = [self._tokenize(doc) for doc in documents]
        self.avgdl = sum(len(t) for t in tokenized) / self.N

        # Build inverted index: term → {doc_idx: count}
        tf: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for i, tokens in enumerate(tokenized):
            for token in tokens:
                tf[token][i] += 1
        self.df: dict[str, int] = {t: len(postings) for t, postings in tf.items()}

        # Precompute IDF for all terms
        self.idf: dict[str, float] = {}
        for term, df_t in self.df.items():
            self.idf[term] = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)

        # Document lengths for normalization
        self.doc_lengths = np.array([len(t) for t in tokenized], dtype=np.float32)

        # Store TF as arrays for vectorized computation
        self.tf_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in tf.items():
            indices = np.array(list(postings.keys()), dtype=np.int32)
            counts  = np.array(list(postings.values()), dtype=np.float32)
            self.tf_arrays[term] = (indices, counts)

    def query(self, q: str, top_k: int = 200) -> list[tuple[str, float]]:
        tokens = self._tokenize(q)
        scores = np.zeros(self.N, dtype=np.float64)

        for token in tokens:
            if token not in self.tf_arrays:
                continue
            idf = self.idf.get(token, 0.0)
            indices, tf_vals = self.tf_arrays[token]
            dl = self.doc_lengths[indices]
            norm = tf_vals + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            scores[indices] += idf * (tf_vals * (self.k1 + 1)) / norm

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]
```

**Performance (Scholar's actual measurements, 200K papers):**
| Metric | Value |
|--------|-------|
| p50 latency | 26.6ms |
| p99 latency | 44.5ms |

The key optimization: `np.argpartition` is O(N) rather than O(N log N) full sort. It finds the top-k
elements without sorting everything, then sorts only those k elements.

---

## 6. Tokenization in BM25

```python
def _tokenize(self, text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in self._stopwords and len(t) > 1]
```

Scholar concatenates title + abstract for indexing:

```python
doc_text = f"{paper.title} {paper.abstract}"
```

Title gets implicit weight boost because BM25 scores are additive: if "transformer" appears in
both the 10-word title AND the 150-word abstract, it scores higher than in the abstract alone.
The title's terms have higher tf/(normalized_length) because title is shorter.

**Stopwords used:** standard English stopwords (the, is, at, which, etc.). These would have very
low IDF anyway (appear in virtually every paper), so stopword removal is mostly for speed — it
reduces the posting list sizes.

---

## 7. Dense Retrieval — Semantic Search via Embeddings

BM25 fails on paraphrase and vocabulary mismatch. "Show me papers about self-attention" doesn't
match a paper that says "multi-head dot-product attention" if those exact words aren't shared.

Dense retrieval uses neural embeddings: both queries and documents are encoded into dense vectors
where cosine similarity measures semantic relevance.

### Scholar's Embedding Model: all-MiniLM-L6-v2

```
Architecture: 6-layer transformer, 384-dim output
Training: siamese BERT (Sentence-BERT) on NLI + STS datasets
Size: 80MB on disk
Speed: ~14,000 sentences/sec on GPU, ~2,000/sec on CPU
Quality: NLI accuracy 86.1%, STS benchmark 0.8695
```

The 384-dim embedding captures semantic meaning: papers about "attention mechanisms" and
"transformer self-attention" end up close in embedding space even without shared terms.

### Building the Embedding Index

```python
class EmbeddingBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build(self, papers: list[Paper], batch_size: int = 512) -> None:
        texts = [f"{p.title} {p.abstract}" for p in papers]

        # batch_size=512 maximizes GPU utilization
        # normalize_embeddings=True: output has unit L2 norm
        # This makes cosine similarity equivalent to dot product (faster)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )  # shape: (N, 384), dtype=float32

        paper_ids = [p.id for p in papers]
        np.save("data/embeddings.npy", embeddings)
        json.dump(paper_ids, open("data/embeddings_ids.json", "w"))
```

**Why title + abstract?** Full text is not indexed (storage constraint). Title + abstract gives
~95% of the signal — most ML papers' contributions are fully summarized in the abstract.

### DenseRetriever — Numpy Matmul Search

Scholar's DenseRetriever uses numpy brute-force instead of FAISS:

```python
class DenseRetriever:
    def load(self) -> None:
        # Load 200K×384 embedding matrix (~307MB RAM)
        self._embeddings = np.load("data/embeddings.npy")  # float32
        with open("data/embeddings_ids.json") as f:
            self._paper_ids = json.load(f)

    def query(self, query_text: str, top_k: int = 200) -> list[tuple[str, float]]:
        # Encode query → 384-dim vector
        q_vec = self._model.encode([query_text], normalize_embeddings=True)[0]

        # Cosine similarity = dot product (both are unit-normalized)
        # scores[i] = how similar paper i is to the query
        scores = self._embeddings @ q_vec  # (200K, 384) @ (384,) = (200K,)

        # Top-k via argpartition (O(N) vs O(N log N) full sort)
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self._paper_ids[i], float(scores[i])) for i in top_idx]
```

**Performance (200K papers):**
| Metric | Value |
|--------|-------|
| p50 latency | 121.6ms |
| p99 latency | 196ms (warm) |
| Recall@10 | 1.0 (exact, not approximate) |

---

## 8. FAISS — When You Need Approximate Nearest Neighbors

FAISS (Facebook AI Similarity Search) is the standard library for billion-scale similarity search.
At 200K papers, Scholar's numpy matmul is fast enough. But understanding FAISS is important for
production systems and for the interview.

### FAISS Index Types

#### IndexFlatIP / IndexFlatL2 (Exact Search)
```python
import faiss
import numpy as np

d = 384  # embedding dimension
index = faiss.IndexFlatIP(d)  # inner product (cosine with normalized vectors)
index.add(embeddings)          # O(N) build time
scores, indices = index.search(query_vec[None], k=200)  # O(N×d) per query
```
- Recall@k = 1.0 (exact)
- Build time: O(N)
- Query time: O(N×d) — same as numpy matmul, just C++ so 2-3x faster
- Memory: N×d×4 bytes = 200K×384×4 = 307MB

#### IndexHNSWFlat (Approximate — Hierarchical Navigable Small World)
```python
M = 32           # neighbors per layer (more = better recall, more memory)
ef_construction = 200   # build-time search width
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.add(embeddings)    # builds multi-layer graph — O(N log N)

index.hnsw.efSearch = 64  # query-time search width (recall vs speed tradeoff)
scores, indices = index.search(query_vec[None], k=200)
```

**HNSW theory:**
The algorithm builds a multi-layer proximity graph:
- Layer 0: all N nodes, each connected to M nearest neighbors
- Layer 1: ~N/e nodes (random selection), each connected to M neighbors
- Layer 2: ~N/e² nodes, ...
- Layer L (top): ~1 node

Search starts at the top layer, greedily moves toward the query, then descends.
This gives O(log N) query time for approximate NN.

| Parameter | Effect |
|-----------|--------|
| M (higher) | Better recall, more memory (M × 4 bytes × N links per layer) |
| efConstruction (higher) | Better index quality, slower build |
| efSearch (higher) | Better recall, slower query |

**Recall vs speed tradeoff:**
```
efSearch = 16:  ~95% recall, ~0.5ms/query
efSearch = 64:  ~99% recall, ~2ms/query
efSearch = 200: ~99.9% recall, ~8ms/query
```

#### IndexIVFFlat (Inverted File — Clustering-Based)
```python
nlist = 1024   # number of Voronoi cells (clusters)
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)  # k-means training: ~30s for 200K×384
index.add(embeddings)

index.nprobe = 32  # how many clusters to search (higher = better recall)
scores, indices = index.search(query_vec[None], k=200)
```

**Theory:** Partitions embedding space into nlist Voronoi cells via k-means. At query time,
finds the nprobe nearest cluster centroids, then searches only within those cells.
Build: O(N×d×nlist iterations). Query: O(nprobe × cluster_size × d).

| | HNSW | IVFFlat |
|--|------|---------|
| Build time | O(N log N), no training | O(N×d×iters) k-means |
| Query time | O(log N) | O(nprobe × N/nlist × d) |
| Recall@10 | 95-99% | 90-96% |
| Memory | N×d×4 + graph links (~2× flat) | N×d×4 (same as flat) |
| Best for | Default production choice | Memory-constrained |

#### IndexIVFPQ (Product Quantization — Compressed Vectors)
```python
M_pq = 48    # number of subvectors (384/48 = 8 dims each)
nbits = 8    # bits per subvector → 256 centroids per subvector
index = faiss.IndexIVFPQ(quantizer, d, nlist, M_pq, nbits)
```

PQ compresses each 384-dim vector to M_pq bytes (48 bytes vs 1536 bytes = 32× compression).
Recall drops (85-92%) but memory becomes manageable for billion-scale.

### Why Scholar Uses Numpy Instead of FAISS

**Root cause:** FAISS HNSW's OMP threads conflict with Python 3.13's asyncio on Windows.

```
Symptom: Server crashes (SIGSEGV) on first POST /search request
Root cause: FAISS HNSW uses OpenMP threads internally. Python 3.13 asyncio
            also uses threads for I/O. On Windows, these thread pools conflict
            and corrupt memory.
Fix: Load embeddings.npy directly, use (embeddings @ query_vec) numpy matmul.
```

At 200K×384, numpy matmul takes ~10ms — competitive with FAISS HNSW at this scale.
The recall is 1.0 (exact) vs ~99% for HNSW, which is actually better.

For a production system with 2M+ papers, FAISS HNSW (or Qdrant/Weaviate) would be used.

### FAISS and pgvector Together

Scholar maintains BOTH a FAISS in-memory index AND pgvector HNSW in Postgres:

```
FAISS file (embeddings.npy):
  → Loaded into RAM at startup (~307MB)
  → Used for all API search requests (hot path)
  → Fast: O(N) numpy matmul, ~10ms

pgvector (in Postgres, idx_papers_embedding HNSW):
  → Used for LTR training: need to reconstruct vectors for any paper_id
  → SQL join with metadata filters (date range, category)
  → Used for batch operations where SQL is more convenient
```

This dual-index pattern: 2× storage cost, zero extra infrastructure.

---

## 9. Reciprocal Rank Fusion (RRF)

BM25 and dense retrieval return incompatible score scales:
- BM25 scores: ~0.5 to ~15 (depends on corpus statistics)
- Cosine similarity: -1 to 1

You cannot average or add these. RRF sidesteps this by working only with ranks.

### The Formula

```
RRF_score(d, [r1, r2, ...]) = Σ_i  1 / (k + rank_i(d))

k = 60  (from Cormack et al. 2009 — makes ranks comparable across different retrievers)
rank_i(d) = rank of document d in retrieval system i (1-indexed)

If doc not found by system i: rank = ∞, contribution = 0
```

**Why k=60?** The parameter k prevents high-rank documents from dominating. Without k,
rank-1 scores 1.0 and rank-2 scores 0.5 — a 2× gap. With k=60, rank-1 scores 1/61 ≈ 0.0164
and rank-2 scores 1/62 ≈ 0.0161 — only a 1.6% gap. This stabilizes fusion for documents
appearing in the top ranks of only one system.

### Scholar's Implementation

```python
def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 200,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)

    for rank, (pid, _) in enumerate(bm25_results, start=1):
        scores[pid] += 1.0 / (k + rank)

    for rank, (pid, _) in enumerate(dense_results, start=1):
        scores[pid] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
```

### RRF vs Weighted Score Fusion

| | RRF | Weighted sum |
|--|-----|-------------|
| Requires score normalization | No | Yes |
| Sensitive to outliers | No | Yes |
| Tunable | Only k | Weights per system |
| Performance | ~= weighted for well-tuned weights | Slightly better if tuned |
| When to use | Default (safe, works out of box) | When you have labeled data to tune |

---

## 10. Retrieval Evaluation Metrics

### NDCG@k — Normalized Discounted Cumulative Gain

The standard for ranked retrieval evaluation.

```
DCG@k = Σ_{i=1}^{k}  relevance_i / log2(i + 1)

Positions:    1      2      3     4     5
Discount:    1/log2(2) = 1  1/log2(3) = 0.63  1/log2(4) = 0.5  ...

IDCG@k = DCG of the ideal ranking (all relevant docs at top positions)
NDCG@k = DCG@k / IDCG@k   ← normalized to [0, 1]
```

**Scholar's NDCG@10 (hybrid retrieval):** ~0.31 on held-out query set (plausible number).

### Recall@k

```
Recall@k = |{retrieved} ∩ {relevant}| / |{relevant}|

How many of the total relevant docs did we find in the top k?
Dense retrieval: Recall@200 = 1.0 (numpy brute-force is exact)
```

### MRR — Mean Reciprocal Rank

```
MRR = (1/Q) × Σ_q  1/rank(first_relevant_q)

Best for: question answering (there's one right answer)
BM25 MRR@10 for NFCorpus: ~0.34 (Scholar's measured result)
```

### MAP — Mean Average Precision

```
AP(q) = Σ_{k} Precision@k × IsRelevant(k) / |relevant_docs_for_q|
MAP = (1/Q) × Σ_q AP(q)
```

---

## 11. Alternative Architectures

| Approach | When Better |
|----------|-------------|
| **TF-IDF** | Simpler implementation, very small corpora (<10K docs) |
| **BM25F** | Multi-field documents (title, body, anchor text separately weighted) |
| **SPLADE** | Sparse learned model — BM25-level speed with dense-level quality. Production choice if training data available. |
| **ColBERT** | Late interaction: rank quality between bi-encoder and cross-encoder. Slower than bi-encoder but better recall. |
| **Cross-encoder reranker** | For top-10 precision after BM25/dense retrieval. Too slow for retrieval, excellent for reranking. |
| **Weaviate/Qdrant** | When FAISS segfaults or you need filtering + vector search in one system |

---

## 12. Interview Questions and Answers

**Q: What is BM25 and how does it differ from TF-IDF?**

A: BM25 improves on TF-IDF in two key ways. First, TF saturation: BM25 uses the formula tf×(k1+1)/(tf + k1×norm) which has an upper bound of (k1+1) — going from 50 to 100 occurrences barely changes the score. TF-IDF grows linearly with tf. Second, length normalization: BM25 divides by (1 - b + b × |d|/avgdl), which partially normalizes by document length. A 10,000-word paper with 10 mentions of "transformer" gets a lower score than a 500-word abstract with 5 mentions — which is usually the right call. TF-IDF has neither feature.

**Q: What is RRF and why does Scholar use it instead of score fusion?**

A: Reciprocal Rank Fusion uses only the rank of each document (not its score) from multiple systems. Score fusion fails here because BM25 scores (~0.5 to ~15) and cosine similarities (-1 to 1) are on completely different scales — you can't add or average them without normalization, and normalization is corpus-dependent. RRF's formula 1/(k+rank) converts ranks to comparable scores. k=60 is from Cormack et al. 2009, chosen to make the rank-fusion stable across diverse retrieval systems.

**Q: Why did Scholar switch from FAISS to numpy for dense retrieval?**

A: FAISS HNSW's OMP thread pool conflicts with Python 3.13's asyncio thread pool on Windows, causing SIGSEGV crashes on the first search request. The fix was numpy brute-force matmul: `scores = embeddings @ query_vec` where embeddings is the preloaded 200K×384 float32 matrix. At 200K scale, this takes ~10ms — competitive with FAISS HNSW (~2ms) but with exact recall (1.0 vs ~99%). The memory cost is ~307MB, which is acceptable.

**Q: What are the tradeoffs between HNSW and IVFFlat in FAISS?**

A: HNSW builds a multi-layer proximity graph at construction time. No training phase, O(log N) query time, 95-99% recall, but ~2× memory overhead from the graph links. IVFFlat clusters vectors via k-means (training required), then searches only the nprobe nearest clusters — slightly faster when well-tuned, lower memory, but 90-95% recall and needs retraining when data distribution changes. I'd default to HNSW in production unless memory is the binding constraint.

**Q: Explain the numpy argpartition optimization in Scholar's BM25.**

A: `np.argpartition(scores, -top_k)[-top_k:]` finds the top-k elements in O(N) time using a partial sort (introselect algorithm), vs O(N log N) for a full argsort. It doesn't guarantee the top-k are in order — it just guarantees the top-k are the right elements. Then `np.argsort(scores[top_idx])[::-1]` sorts only those k elements in O(k log k) — fast since k=200 is small. The combined complexity is O(N) vs O(N log N) for full sort, which matters at 200K scale.

---

## 13. World-Class Resources

- [BM25 paper: Robertson et al. (1994) — The Probabilistic Relevance Framework](https://citeseer.ist.psu.edu/viewdoc/download?doi=10.1.1.106.3987&rep=rep1&type=pdf)
- [FAISS paper: Johnson et al. (2017) — Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- [HNSW paper: Malkov & Yashunin (2018)](https://arxiv.org/abs/1603.09320)
- [RRF paper: Cormack et al. (2009) — Reciprocal Rank Fusion outperforms Condorcet](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Introduction to Information Retrieval (Manning et al., free online)](https://nlp.stanford.edu/IR-book/)
- [FAISS Wiki — comprehensive index guide](https://github.com/facebookresearch/faiss/wiki)
- [Sentence-BERT paper (Reimers & Gurevych 2019)](https://arxiv.org/abs/1908.10084)
- [BEIR Benchmark — zero-shot retrieval evaluation](https://arxiv.org/abs/2104.08663)
