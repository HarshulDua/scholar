# Recommendation Systems — Complete Deep Dive

> Covers: RecSys fundamentals, collaborative filtering, matrix factorization, LTR theory,
> LightGBM LambdaRank, MMR diversity, ε-greedy exploration, and Scholar's exact Layer 2 pipeline.

---

## 1. The Recommendation Problem

**Goal:** Given a user's interaction history and a set of candidate items, rank the items so the
ones the user will find most relevant appear at the top.

**Scholar's formulation:**
- Input: a user_id, their reading history (paper IDs), and 200 candidate papers from retrieval
- Output: a ranked list of 10 papers personalized to the user's interests and level

The challenge has four dimensions:
1. **Relevance** — does the paper match the user's research interests?
2. **Novelty** — have they already read it / is it too similar to what they know?
3. **Diversity** — does the ranked list cover multiple topics, not just the most popular cluster?
4. **Exploration** — do we avoid trapping the user in a filter bubble of what they've read before?

Scholar's Layer 2 pipeline addresses all four with a sequence of models.

---

## 2. Collaborative Filtering — The Foundation

**Core idea:** users who agreed in the past will agree in the future.

```
Alice reads: [NLP, NLP, RL, NLP]
Bob reads:   [NLP, NLP, NLP]
Carol reads: [CV, CV, RL, CV]

Alice and Bob have similar history → recommend to Alice what Bob liked next
Alice and Carol are different → Carol's next read is less useful for Alice
```

### User-Based CF

For a target user u, find the k most similar users (based on interaction overlap), then recommend
items those users liked that u hasn't seen.

```python
# Jaccard similarity
sim(u, v) = |history_u ∩ history_v| / |history_u ∪ history_v|

# Prediction: weighted average of neighbor ratings
pred(u, i) = Σ_v sim(u, v) × rating(v, i) / Σ_v |sim(u, v)|
```

**Problems:** doesn't scale past ~100K users (O(U²) pairwise similarity), and sparse histories
make similarities unreliable.

### Item-Based CF

Compute item-item similarities instead. "Papers similar to paper A" is more stable than
"users similar to user u" because items don't change behavior.

```python
sim(item_i, item_j) = cosine_similarity(interaction_vector_i, interaction_vector_j)
```

Item-based CF underlies most real-world recommendation engines (Amazon, Netflix historically).

### Matrix Factorization

Factorize the user-item interaction matrix R into two low-rank matrices:

```
R ≈ U × V^T

R: (n_users × n_items)
U: (n_users × k)   — user latent factors
V: (n_items × k)   — item latent factors

pred(u, i) = dot(U[u], V[i])
```

SVD is the theoretical basis, but ALS (Alternating Least Squares) and SGD are used in practice
for sparse matrices. Scholar upgrades from MF to Mult-VAE to get probabilistic user representations.

---

## 3. Mult-VAE — Scholar's User Representation

Mult-VAE (see `variational_autoencoder.md` for full theory) maps a user's binary interaction
vector to a 200-dim latent representation.

```
user_history_ids → binary vector (200K dim) → encoder → (μ, σ) → z (200 dim)
```

The 200-dim vector `μ` is the user's personalized representation. It's used as:
1. A query vector for FAISS/numpy semantic search (find papers near the user's taste)
2. A feature in LightGBM LTR: `user_paper_cosine = cosine(μ, paper_emb)`
3. Input to `adaptive_lambda()` for MMR diversity scaling

**Fallback for cold-start:** when user has no history, average sentence-BERT embeddings of
recently popular papers provides a reasonable 384-dim fallback vector.

---

## 4. Learning to Rank (LTR)

LTR treats ranking as a supervised ML problem. Instead of hand-tuning a ranking formula, you
learn the ranking function from labeled data (or implicit feedback like clicks).

### Three Paradigms

**Pointwise:** predict a relevance score for each (query, document) pair independently.
Train a regression model: score(q, d) → predict click probability.
```
Loss = MSE(predicted_score, actual_relevance_label)
```
Problem: ignores relative ordering — a model predicting all docs at 0.5 would have zero loss
if all true labels are 0.5.

**Pairwise:** for each pair of documents, predict which should rank higher.
```
Loss = Σ_{i,j: rel_i > rel_j} max(0, 1 - (score_i - score_j))   (hinge loss)
```
RankSVM, RankNet. Better than pointwise but still doesn't directly optimize ranking metrics.

**Listwise:** treat the entire ranked list as the optimization target.
```
Loss = -NDCG(predicted_ranking, true_ranking)   (approximately)
```
LambdaRank, ListNet. Directly optimizes the metric you care about.

---

## 5. LightGBM — Gradient Boosted Trees

### Decision Trees and Gradient Boosting

A **decision tree** recursively splits the feature space:
```
IF bm25_score > 3.2 AND citation_count > 50 THEN score = 0.8
ELSE IF category_overlap > 0.5 THEN score = 0.6
ELSE score = 0.3
```

A single tree overfits and doesn't generalize. **Gradient boosting** builds trees sequentially:
each new tree fits the residual errors of all previous trees.

```
F_0(x) = initial prediction (e.g., mean)
F_m(x) = F_{m-1}(x) + η × h_m(x)

where h_m = new tree trained on residuals:
  residuals_i = -∂L/∂F_{m-1}(x_i)   (negative gradient of loss)
  η = learning rate (shrinkage — prevents any single tree from dominating)
```

### LightGBM Innovations

**Histogram-based splits:** instead of sorting all feature values to find split points
(O(N) per feature per node), LightGBM bins features into k histograms (k=255) and finds
the best split among bin boundaries. This reduces split finding from O(N) to O(k).

**GOSS (Gradient-based One-Side Sampling):** keep all data points with large gradients
(high error = high learning signal), sample only a fraction of small-gradient points.
This maintains accuracy while using less data per iteration.

**EFB (Exclusive Feature Bundling):** sparse features that rarely are both nonzero can be
combined into a single feature, reducing the effective feature count.

Together: LightGBM is typically 5-10× faster than XGBoost on tabular data with comparable
accuracy.

---

## 6. LambdaRank — Listwise Ranking Loss

LambdaRank (Burges et al. 2006) trains a model to directly optimize NDCG.

**The insight:** you don't need a loss function — you need gradients. Define the gradient
directly as the change in NDCG when you swap a pair of documents:

```python
# For each pair (i, j) where doc i should rank above doc j:

# Score difference through sigmoid
s_ij = score_i - score_j
sigma_ij = sigmoid(s_ij) = 1 / (1 + exp(-s_ij))

# ΔNDCG: how much would NDCG change if we swapped i and j?
delta_ndcg = |NDCG with swap - NDCG without swap|

# Lambda gradient (used directly as pseudo-gradient in gradient boosting)
λ_ij = |ΔNDCG_ij| × (1 - sigma_ij)
     = |ΔNDCG_ij| × exp(-s_ij) / (1 + exp(-s_ij))

# If i correctly outranks j (s_ij > 0): σ ≈ 1, (1-σ) ≈ 0 → small gradient
# If j incorrectly outranks i (s_ij < 0): σ ≈ 0, (1-σ) ≈ 1 → large gradient
# If |ΔNDCG| large (swapping would hurt a lot): gradient amplified
```

The gradient is large when:
1. Documents are in the wrong order (i should rank above j but doesn't)
2. Swapping them would change NDCG significantly (e.g., swapping rank-1 and rank-2)

**LightGBM integration:**
```python
import lightgbm as lgb

# group parameter: how many docs per query (critical!)
# LambdaRank needs to know which docs belong to the same query
query_groups = [len(candidates_for_query) for query in train_queries]
# e.g., [10, 8, 12, 10, ...] if each query has different numbers of candidates

dtrain = lgb.Dataset(X_train, label=y_train, group=query_groups)
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],   # evaluate NDCG@10
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 1,
}
model = lgb.train(params, dtrain, num_boost_round=100)
```

**Why group matters:** without group, LightGBM treats all rows as one query and tries to rank
200K papers against each other — meaningless. With group, it ranks only within each query's
candidates, which is the actual task.

---

## 7. Scholar's 7 LTR Features

```python
def extract_features(
    user_emb: np.ndarray,      # 200-dim Mult-VAE embedding
    candidate_ids: list[str],
    candidate_scores: dict[str, float],    # RRF scores from retrieval
    paper_meta: dict[str, Paper],
    paper_embeddings: dict[str, np.ndarray],  # 384-dim SBERT embeddings
    history_cats: list[str],    # user's history paper categories
    history_ids: set[str],      # user's history paper IDs
) -> np.ndarray:  # shape: (len(candidate_ids), 7)
```

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `rrf_score` | Reciprocal rank fusion score from retrieval | Overall retrieval relevance |
| `bm25_rank_norm` | BM25 rank / total candidates (0-1) | BM25 signal separate from dense |
| `dense_rank_norm` | Dense retrieval rank / total candidates (0-1) | Dense signal separate from BM25 |
| `user_paper_cosine` | cosine(user_emb, paper_emb) | Personalization — user taste match |
| `citation_count_norm` | log(citation_count + 1) / log(max_citations + 1) | Paper quality signal |
| `recency_score` | days_since_paper / max_days (0-1, inverted) | Newer papers preferred |
| `category_overlap` | len(user_history_cats ∩ paper_cats) / len(paper_cats) | Domain relevance |

**Feature engineering decisions:**
- Ranks normalized to [0,1] so LightGBM doesn't need to learn the scale
- Citation count log-normalized: the difference between 100 and 1000 citations matters,
  but 10,000 vs 100,000 matters much less
- Category overlap uses set intersection, not string matching, because ArXiv categories
  are structured (cs.LG, cs.AI, stat.ML, etc.)

---

## 8. The Recommend Pipeline (End-to-End)

```python
# api/routes/recommend.py — full flow with code

@router.post("/recommend")
async def recommend(req: RecommendRequest, ...):
    # 1. Load user embedding from Mult-VAE
    user_emb = await _get_user_embedding(user_id, request.state)
    # Returns 200-dim or 384-dim fallback vector

    # 2. Get user history
    history_ids, history_cats = await _get_user_history(session, user_id)

    # 3. Use user_emb to get initial candidate set from dense retrieval
    #    (in addition to a generic query-based retrieval)
    dense_candidates = dense_retriever.query_by_vector(user_emb, top_k=200)
    # This is the personalization: searching embedding space near the user's taste

    # 4. Extract LTR features for all candidates
    candidate_ids = [pid for pid, _ in dense_candidates]
    paper_meta = await _get_papers_by_ids(session, candidate_ids)
    paper_embeddings = {pid: dense_retriever.get_embedding(pid) for pid in candidate_ids}
    candidate_scores = {pid: score for pid, score in dense_candidates}

    X = extract_features(
        user_emb, candidate_ids, candidate_scores,
        paper_meta, paper_embeddings, history_cats, history_ids
    )

    # 5. LightGBM LambdaRank scoring
    ltr_model = request.state.ltr_model
    scores = ltr_model.predict(X)  # shape: (200,)

    # 6. Sort by LTR score, take top-50
    sorted_idx = np.argsort(scores)[::-1][:50]
    top_50 = [candidate_ids[i] for i in sorted_idx]
    top_50_embs = np.array([paper_embeddings[pid] for pid in top_50])

    # 7. MMR diversity reranking: top-50 → top-10
    query_vec = user_emb if len(user_emb) == 384 else None  # MMR uses 384-dim
    top_10_ids = mmr_rerank(top_50, top_50_embs, query_vec, k=10, lam=adaptive_lambda(...))

    # 8. ε-greedy exploration: inject ~1-2 random papers
    final_ids = epsilon_greedy_inject(top_10_ids, all_paper_ids, epsilon=0.1)

    # 9. Fetch full paper objects and return
    result_papers = [paper_meta[pid] for pid in final_ids if pid in paper_meta]
    return {"papers": result_papers, "user_id": user_id}
```

---

## 9. MMR — Maximum Marginal Relevance

After LTR gives us a relevance-ranked list of 50 papers, MMR reranks to maximize both
relevance AND diversity.

### Formula

```
MMR(d, S, Q) = argmax_{d ∈ R\S} [λ × sim(d, Q) - (1-λ) × max_{s ∈ S} sim(d, s)]

R = candidate set (top-50 papers)
S = already-selected set (starts empty, grows)
Q = query vector (user embedding or query text)
λ ∈ [0,1]: 1 = pure relevance, 0 = pure diversity
sim = cosine similarity (normalized embeddings → dot product)
```

**Algorithm (greedy):**
```python
def mmr_rerank(
    candidates: list[str],
    embeddings: np.ndarray,    # (50, 384)
    query_vec: np.ndarray,     # (384,)
    k: int = 10,
    lam: float = 0.5,
) -> list[str]:
    # Precompute all cosine similarities
    # (unit-normalized, so dot product = cosine)
    query_sims = embeddings @ query_vec          # (50,) — relevance to query
    paper_sims = embeddings @ embeddings.T       # (50, 50) — pairwise similarity

    selected_idx = []
    candidate_idx = list(range(len(candidates)))

    for _ in range(k):
        best_score = -np.inf
        best_i = None

        for i in candidate_idx:
            relevance = query_sims[i]

            # Diversity: maximum similarity to already-selected papers
            if selected_idx:
                redundancy = max(paper_sims[i, j] for j in selected_idx)
            else:
                redundancy = 0.0

            mmr_score = lam * relevance - (1 - lam) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i

        selected_idx.append(best_i)
        candidate_idx.remove(best_i)

    return [candidates[i] for i in selected_idx]
```

**Time complexity:** O(K² × D) where K=50 (candidates) and D=384 (embedding dim).
This is why MMR is done on top-50 only, not all 200 candidates — K² grows fast.

### Adaptive Lambda

Rather than a fixed λ, Scholar computes it from the user's reading history:

```python
def adaptive_lambda(history_embeddings: np.ndarray) -> float:
    """
    If user has a diverse reading history → they explore → lower λ (more diversity)
    If user has a focused reading history → they specialize → higher λ (more relevance)
    """
    if len(history_embeddings) < 2:
        return 0.5   # not enough history to determine preference

    # Compute pairwise cosine similarities across history
    norms = np.linalg.norm(history_embeddings, axis=1, keepdims=True)
    normalized = history_embeddings / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T
    n = len(history_embeddings)
    # Average off-diagonal similarity
    avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))

    # High avg_sim → focused user → prefer relevance (high λ)
    # Low avg_sim → diverse user → prefer diversity (low λ)
    lam = 0.3 + 0.4 * avg_sim   # maps [0,1] avg_sim → [0.3, 0.7]
    return float(np.clip(lam, 0.3, 0.7))
```

**Example values:**
- User reads only NLP papers: avg_sim ≈ 0.9 → λ ≈ 0.66 (more relevance)
- User reads NLP + CV + RL: avg_sim ≈ 0.3 → λ ≈ 0.42 (more diversity)

---

## 10. ε-Greedy Exploration — Filter Bubble Prevention

MMR improves diversity within a query, but long-term, a user who always clicks NLP papers
gets an increasingly NLP-only feed. ε-greedy breaks this pattern.

### The Filter Bubble Problem

```
Session 1: User clicks NLP papers → history becomes NLP-heavy
Session 2: Embeddings recommend more NLP → user sees NLP → clicks NLP
Session 3: Feedback loop deepens → user never sees CV, RL, etc.
Session 20: Topic entropy collapsed to ~0.6 (Scholar's simulation result)
```

### ε-Greedy Solution

With probability ε, replace one recommended paper with a random paper from outside the
top-k relevance bubble:

```python
def epsilon_greedy_inject(
    ranked_ids: list[str],
    all_paper_ids: list[str],
    epsilon: float = 0.1,
    seed: int | None = None,
) -> list[str]:
    """
    With probability epsilon, swap one paper in ranked_ids for a random exploration paper.
    The exploration paper is NOT in the current top-k (it comes from outside the bubble).
    """
    rng = np.random.default_rng(seed)
    result = list(ranked_ids)

    if rng.random() < epsilon:
        # Pick a random paper NOT already in the ranked list
        candidates = [pid for pid in all_paper_ids if pid not in set(ranked_ids)]
        if candidates:
            explore_paper = rng.choice(candidates)
            # Replace the LAST paper (lowest relevance position)
            # Don't disrupt the top results
            result[-1] = explore_paper

    return result
```

**ε = 0.1 design choice:**
- Too high (ε=0.5): 50% random results → destroys relevance, users leave
- Too low (ε=0.01): filter bubble persists, exploration too rare to matter
- ε=0.1: 10% chance per session → ~every 10 sessions user gets one serendipitous paper

**Scholar's simulation result:**
```
Without exploration: Topic entropy@session5 = ~0.8, session20 = ~0.6 (collapsed)
With MMR + ε-greedy: Topic entropy@session5 = ~2.4, session20 = ~2.3 (stable)

ILD improvement with ε-greedy: 0.496 → 0.576 (+16.2%)  ← actual eval result, cite this
```

---

## 11. ILD — Intra-List Diversity

How diverse is the set of 10 recommended papers?

```python
def compute_ild(paper_embeddings: list[np.ndarray]) -> float:
    """
    ILD = average pairwise dissimilarity within a recommendation list.
    dissimilarity = 1 - cosine_similarity
    """
    n = len(paper_embeddings)
    if n < 2:
        return 0.0

    embeddings = np.array(paper_embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T

    # Average upper triangle (excluding diagonal)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (1 - sim_matrix[i, j])   # dissimilarity
            count += 1
    return total / count
```

**Scholar's ILD results:**

| Configuration | ILD |
|--------------|-----|
| MMR only (λ=0.5 fixed) | 0.496 |
| LTR + MMR (adaptive λ) | 0.496 |
| LTR + MMR + ε-greedy | **0.576** (+16.2%) |

ILD is corpus-mismatch-immune (unlike NDCG) because it only measures the diversity within
the 10 returned papers, not whether those papers were the "correct" ones from an external dataset.

---

## 12. Training Data and Practical LTR

For real LTR training you need labeled (query, document, relevance) triples. Scholar uses
CiteULike interaction data as implicit feedback:

```
CiteULike: 16,980 users × 204,986 papers, recorded from 2004-2016
For a given user+query:
  - Papers they saved → relevance=2 (clicked and bookmarked)
  - Papers they viewed but didn't save → relevance=1
  - Papers not shown → relevance=0 (assumed)
```

**The corpus mismatch problem:** Only 158 of 16,980 CiteULike paper IDs appear in Scholar's
200K ArXiv corpus (different ID formats + different subsets). This makes NDCG ≈ 0 on the
CiteULike test set because the "relevant" papers are almost never in the corpus.

ILD is the meaningful metric because it measures diversity of the returned set, not whether
a specific external paper was retrieved.

---

## 13. Alternatives Considered

| Approach | Why Not Used |
|----------|-------------|
| **Neural CF (NCF)** | No latent distribution over users, can't quantify uncertainty, slower inference |
| **Two-Tower Models** | Separate query/document encoders trained on clicks — better quality but requires real click logs |
| **EASE** | No user embedding for personalized FAISS queries |
| **Sequential RecSys (BERT4Rec)** | Interaction order doesn't matter for Scholar's use case |
| **Bandit algorithms (UCB)** | More principled exploration but requires online feedback loop — Scholar is batch |

---

## 14. Interview Questions and Answers

**Q: Walk me through Scholar's recommendation pipeline end-to-end.**

A: Six stages. First, a user embedding from Mult-VAE: the user's reading history (binary
interaction vector over 200K papers) is encoded to a 200-dim latent vector via a trained
variational autoencoder. Second, dense retrieval using that vector as the query — `embeddings @ user_emb`
finds the 200 papers whose content embeddings are closest to the user's taste. Third, LightGBM
LambdaRank: we extract 7 features (RRF score, BM25 rank, dense rank, user-paper cosine,
citation count, recency, category overlap) and score each candidate. Fourth, MMR diversity
reranking on the top-50: balances relevance vs novelty using adaptive λ tuned to how diverse
the user's history is. Fifth, ε-greedy: with 10% probability, replace the lowest-ranked paper
with a random paper from outside the top-k. Sixth, fetch and return top-10 papers.

**Q: Why use LambdaRank specifically? What's wrong with pointwise regression?**

A: Pointwise regression predicts a relevance score for each document independently. The model
can get zero loss by predicting every document as 0.5 even if the ordering is completely wrong.
Ranking metrics like NDCG@10 only care about the relative ORDER of documents — whether document
A ranks above document B, not their exact scores. LambdaRank computes pseudo-gradients proportional
to the NDCG change from swapping each pair, weighted by whether they're in the wrong order. This
directly optimizes NDCG instead of a proxy loss.

**Q: Why does Scholar use the `group` parameter in LightGBM, and what happens without it?**

A: LambdaRank ranks documents WITHIN a query — it needs to know which documents belong to the
same query to compute pairwise comparisons and ΔNDCG. The `group` parameter is a list of counts:
[10, 8, 12, ...] means query 1 has 10 candidates, query 2 has 8, etc. Without it, LightGBM
treats all rows as one huge query and tries to rank 200K papers against each other — the loss
is meaningless because rank position in a 200K list has nothing to do with rank position in a
10-paper query result.

**Q: What is MMR and why post-LTR?**

A: Maximum Marginal Relevance iteratively selects papers that maximize relevance minus redundancy
with already-selected papers. The formula: `MMR(d) = λ × sim(d, query) - (1-λ) × max_{s ∈ selected} sim(d, s)`.
Post-LTR means we apply it after LambdaRank has already done quality filtering (top-50 from 200).
Applying MMR to all 200 would be O(200² × 384) = 15M multiplications per request. On top-50:
O(50² × 384) ≈ 1M multiplications — 25× faster and the top-50 are already high quality.

**Q: How does filter bubble prevention work and what did you measure?**

A: Two mechanisms. MMR's adaptive λ: when a user has diverse history (low avg_sim between
history embeddings), we lower λ, increasing the weight on novelty vs relevance. ε-greedy:
with probability ε=0.1 per session, we replace the lowest-ranked paper with a random paper
from outside the top-k — breaking the feedback loop. We measured ILD (Intra-List Diversity,
average pairwise dissimilarity within the 10 results) on a simulation of 20 sessions. Without
exploration, topic entropy collapsed from ~2.4 to ~0.6 by session 20. With MMR + ε-greedy,
it stayed at ~2.3. The ILD improved from 0.496 to 0.576 (+16.2%) — that's the number to cite.

**Q: How would you improve the recommendation system with more resources?**

A: Three areas. First, real click data: Scholar uses CiteULike (2016 data) which has 158/16,980
paper ID matches with the ArXiv corpus — near-zero usable labels. With production clicks, LTR
quality would improve dramatically. Second, two-tower models: train separate user/paper encoders
end-to-end on clicks, replacing the Mult-VAE + SBERT combination with a unified model. Third,
online learning: currently Mult-VAE is retrained offline. With Kafka or real-time feedback,
the user model could update continuously.

---

## 15. World-Class Resources

- [LambdaRank paper: Burges et al. (2006) — From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [LightGBM paper: Ke et al. (2017)](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [MMR paper: Carbonell & Goldstein (1998)](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_SIGIR_1998.pdf)
- [Mult-VAE paper: Liang et al. (2018)](https://arxiv.org/abs/1802.05814)
- [RecSys textbook: Aggarwal (2016) — Recommender Systems: The Textbook](https://link.springer.com/book/10.1007/978-3-319-29659-3)
- [LightGBM docs](https://lightgbm.readthedocs.io/en/stable/) — LambdaRank parameters explained
- [ε-Greedy in RecSys: Exploration-Exploitation in Collaborative Filtering](https://dl.acm.org/doi/10.1145/2507157.2507173)
- [Reinforcement Learning (Sutton & Barto) Ch. 2](http://incompleteideas.net/book/the-book-2nd.html) — Multi-armed bandits and ε-greedy theory
