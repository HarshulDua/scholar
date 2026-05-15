# Scholar — Optimizations, Fixes & Engineering Decisions

> Interview reference. Every non-trivial decision made during the build, with root causes, fixes, and what to say in the room. Read before any ML engineering interview.

---

## System Architecture Summary

```
Query + user_id
    ↓
[RETRIEVAL]  BM25 (from scratch) + numpy matmul dense (all-MiniLM-L6-v2)
             Fused via RRF (k=60) → top-200 candidates
    ↓
[RERANKING]  Mult-VAE CF scores (alpha=0.3 blend)
             → LightGBM LambdaRank (LTR, 7 features) → top-50
             → MMR diversity reranking (O(K²) on top-50) → top-10
             → ε-greedy exploration (inject 1 random paper, ε=0.1)
    ↓
[GENERATION] QLoRA Phi-3-mini-4k-instruct (4-bit NF4, LoRA rank 8, alpha 16)
             Level-adaptive summaries (undergrad / grad / researcher)
```

**Hard constraints (never change):** BM25 from scratch, Mult-VAE multinomial likelihood + KL annealing `beta=min(0.2, epoch/20*0.2)`, RRF k=60, MMR on top-50 only, QLoRA 4-bit NF4 rank-8.

**Data:** 200K ArXiv papers (CS+ML), CiteULike-A (5551 users, 1716 interactions, 158 matched to corpus).

---

## Part A — Performance Optimizations

### 1. BM25 Retrieval — Numpy Vectorization

**Problem:** First BM25 query after startup: 800–1200ms. Even warm queries: 400–800ms. Unusable for interactive use.

**Root cause:** Pure Python iteration over posting lists. A common ML term like "learning" has 12,700+ posting entries. A 3-token query hits ~14,000 total entries. At CPython speeds: ~10M ops/sec vs numpy's ~500M–1B/sec. Secondary: first `np.argpartition` on a 200K-element array pays a cold OS page-fault tax (~370ms).

**Fix — Part 1: Numpy cache at load time**

```python
# Build once at fit()/load() time:
_np_doc_indices[term] = np.array([...], dtype=np.int32)   # int indices
_np_tfs[term]         = np.array([...], dtype=np.float32) # term freqs
dl_arr                = np.array([doc_lengths[i] for i in range(n)], dtype=np.float32)

# Per query:
scores = np.zeros(n, dtype=np.float64)
for term in tokens:
    doc_ints = _np_doc_indices[term]
    tfs      = _np_tfs[term]
    dl       = dl_arr[doc_ints]                      # gather
    norm     = k1 * (1 - b + b * dl / avg_dl)        # broadcast
    tf_score = (tfs * (k1+1)) / (tfs + norm)         # broadcast
    scores[doc_ints] += idf * tf_score               # scatter-add (safe: each doc once per term)

top_ints = np.argpartition(scores, -k)[-k:]          # O(n), not O(n log n)
top_ints = top_ints[np.argsort(scores[top_ints])[::-1]]  # sort only k=200
```

Note: `scores[doc_ints] += values` is safe here (inverted index never has the same doc twice per term). Avoids the slower `np.add.at()` by design.

**Fix — Part 2: Warmup query at startup**

```python
obj.query("warmup query neural network", top_k=10)
```

Forces OS to page the scores array into L3 cache before first real request.

**Result:** 800ms → 15–60ms (13-50x speedup). Cold-start tax eliminated.

**Interview:** *"I profiled the BM25 loop and found Python-speed iteration over 10K+ posting entries per term. I converted string-keyed dict postings into int-indexed numpy arrays at load time — a one-time O(vocab × avg_posting_length) cost — so every query runs as vectorized array ops. Also fixed a cold-start page-fault tax with a warmup query at boot."*

---

### 2. Dense Retrieval — Sentence-Transformer Cold Start

**Problem:** First search request after startup froze for ~115 seconds. Users thought server crashed.

**Root cause:** Sentence-transformer (all-MiniLM-L6-v2, 22M params) was lazy-loaded on first query. Loading: import + weight shards + PyTorch graph build + first JIT pass = 60–120s on HDD.

**Fix:**

```python
# Inside DenseRetriever.load():
print("Warming up sentence-transformer model...")
self._encode_and_normalize("warmup")   # triggers _ensure_model() + one forward pass
print("Dense retriever ready.")
```

Runs during server startup before first request is accepted.

**Result:** 115s cold start → eliminated. Startup cost ~170s (one-time, acceptable).

**Interview:** *"The sentence-transformer was lazy-loaded on first inference — a 2-minute freeze. I moved loading into the server startup sequence via a warmup inference call. In production, requests should never wait for infrastructure initialization."*

---

### 3. Database Query — Excluding pgvector Embedding Column

**Problem:** Every search fetched full Paper ORM objects for 200 candidates, including the pgvector embedding column (384-dim float32 = 1,536 bytes per row = ~307KB discarded per search).

**Fix:**

```python
select(
    Paper.id, Paper.title, Paper.abstract,
    Paper.authors, Paper.categories, Paper.citation_count,
).where(Paper.id.in_(paper_ids))
```

**Result:** DB payload per search: ~307KB → ~80KB (75% reduction). Also reduces PostgreSQL serialize cost.

**Interview:** *"Standard ORM pitfall: select(Paper) pulls every column including the 384-dim embedding that the API response never uses. Always audit what columns the query actually needs — explicit column selection instead of wildcard ORM objects."*

---

### 4. Uvicorn Hot-Reload — File Watcher Scanning Data Directory

**Problem:** Every file save triggered a reload taking 30–60 seconds. `data/` contains: faiss.index (321MB), embeddings.npy (293MB), bm25.pkl (163MB), other artifacts — total ~800MB+.

**Fix:**

```bash
uvicorn scholar.api.main:app --reload-dir scholar --port 8000
```

Restricts file watcher to Python source only. Hot-reload: 2–3s instead of 30–60s.

**Interview:** *"In any ML system the data directory is large. File watchers don't know to ignore binary artifacts. Scope the watcher to source code only — --reload-dir is the uvicorn knob for this."*

---

## Part B — Model & Algorithm Fixes

### 5. Mult-VAE Evaluation — Per-User Leave-Out Split

**Problem:** Initial Mult-VAE validation reported NDCG@10 = 0.9803 (98%). Fake.

**Root cause:** User-level train/val split: train users' full histories used for training; val users never seen at all → predicting for all-zero vectors with only 158-item vocab → near-trivial task.

**Fix:** Per-user leave-20%-out (standard CF protocol from Liang et al. 2018):

```python
def _per_user_split(matrix, rng, val_frac=0.2):
    X_train = matrix.copy()
    X_val = np.zeros_like(matrix)
    for u in range(matrix.shape[0]):
        item_indices = np.where(matrix[u] > 0)[0]
        if len(item_indices) < 2:
            continue
        n_val = max(1, int(len(item_indices) * val_frac))
        val_items = rng.choice(item_indices, size=n_val, replace=False)
        X_train[u, val_items] = 0.0
        X_val[u, val_items]   = 1.0
```

**Result:** NDCG@10: 0.9803 (fake) → 0.1371 (genuine). Random baseline: ~0.063. Model achieves 2.2x better than random — real signal.

**Interview:** *"98% NDCG is a red flag. The user-level split made the task trivially easy — the model never had to generalize per user. Switched to per-user leave-20%-out, the standard protocol from the Mult-VAE paper. Real NDCG dropped to 0.14 vs 0.06 random — genuinely informative."*

---

### 6. Mult-VAE — Missing Input Dropout

**Problem:** Mult-VAE was missing input dropout — a core technique from Liang et al. 2018, forcing the model to learn robust representations from partial observation.

**Fix:**

```python
class MultVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=200, hidden_dim=600, dropout_p=0.5):
        self.input_dropout = nn.Dropout(dropout_p)

    def encode(self, x):
        x = self.input_dropout(x)          # applied before first linear layer
        h = torch.tanh(self.enc_fc1(x))
        out = self.enc_fc2(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar
```

Saved `dropout_p` in checkpoint and used `checkpoint.get("dropout_p", 0.5)` for backward-compatible loading.

**Why it matters:** Input dropout randomly zeroes observed interactions during training, forcing the encoder to learn latent representations that generalize from any partial history. At inference, dropout is disabled — full history is encoded. Especially important with small interaction datasets.

**Interview:** *"Input dropout in Mult-VAE isn't a standard regularization trick — it's a deliberate design choice from the paper. It augments the data and forces cold-start robustness. Key difference from weight dropout: operates on data space, not parameter space."*

---

### 7. VAE Score Blending — Dead Code Path

**Problem:** The recommend endpoint was supposed to blend VAE CF scores with FAISS cosine similarity, but the VAE was never called. 100% of recommendations came from FAISS.

**Root cause:** VAE blending was inside an `else` branch. Since history papers are always in the FAISS index, `history_embeddings` is always non-empty — the `else` never executed. VAE was dead code.

**Fix:** Moved VAE blending to after FAISS retrieval as a post-processing boost:

```python
# FAISS retrieval (always runs)
candidates = dense_retriever.query_by_vector(user_emb, top_k=200)

# VAE blending (runs on top of FAISS, alpha=0.3)
alpha = 0.3
blended = [
    (pid, fscore * (1.0 + alpha * float(vae_scores[vidx])))
    if (vidx := vae_id_to_idx.get(pid)) is not None
    else (pid, fscore)
    for pid, fscore in candidates
]
candidates = sorted(blended, key=lambda x: x[1], reverse=True)
```

Alpha=0.3: FAISS cosine similarity is the dominant signal (70%+ weight), VAE provides a soft boost for CF-favored items.

**Interview:** *"The VAE integration was dead code — a logic error meant it lived inside an else branch that never ran. The lesson: when combining two complementary signals, blend them multiplicatively on the same candidate set, don't choose between them."*

---

### 8. Phi-3 Generation — Transformers 5.x DynamicCache Compatibility

**Problem:** `synth_data.py` reported "Skipped" for every paper (0/3000 generated). No visible error because try/except swallowed it.

**Root cause:** Phi-3's cached model file (`modeling_phi3.py`) was written for transformers 4.x. Transformers 5.8.1 removed three DynamicCache attributes: `seen_tokens`, `get_max_length()`, `get_usable_length()`.

**Fix:** Monkey-patch the three missing methods before model load:

```python
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_len, layer_idx=0: self.get_seq_length()
```

`hasattr` guard makes it a no-op if transformers ever adds them back.

**Lesson from the silent failure:** try/except around generation masked a 100% failure rate. Always verify output file contents, not just exit code.

**Interview:** *"Phi-3's model code on HF Hub was written for transformers 4.x, we had 5.x. Three DynamicCache methods were removed. Found all three by grepping the cached modeling_phi3.py rather than hunting them one by one. Monkey-patching before model load is idiomatic for third-party compatibility issues where you can't modify vendor code."*

---

### 9. LoRA Adapter — Base Model Silently Loading LoRA Weights

**Problem:** Generation eval comparing "base" vs "LoRA" was actually comparing LoRA vs LoRA — both model instances loaded the adapter. The "base model" numbers were not baseline numbers.

**Root cause:** `Phi3Generator.__init__` always called `PeftModel.from_pretrained` if `settings.lora_adapter_path` existed. The `load_adapter` concept was not implemented — both "base" and "LoRA" evaluation calls created identical LoRA-loaded models.

**Fix:** Added `load_adapter: bool = True` parameter to `__init__` and `get_instance()`:

```python
def __init__(self, load_adapter: bool = True) -> None:
    ...
    if (
        load_adapter
        and settings.lora_adapter_path is not None
        and settings.lora_adapter_path.exists()
    ):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, str(settings.lora_adapter_path))

@classmethod
def get_instance(cls, load_adapter: bool = True) -> "Phi3Generator":
    if cls._instance is None:
        cls._instance = cls(load_adapter=load_adapter)
    return cls._instance
```

**Interview:** *"The base model evaluation was silently broken — the singleton factory always loaded the adapter if it existed. Added a `load_adapter` flag to make the distinction explicit. This is a classic singleton footgun: shared state (the adapter) was forced on all callers with no opt-out."*

---

## Part C — Windows-Specific Bugs

### 10. Windows DLL Conflict — PyArrow + Torch on Import

**Problem:** Server crashed with `Windows fatal exception: access violation` in `pyarrow/__init__.py` whenever `sentence_transformers` was imported.

**Root cause:** Import chain: `sentence_transformers` → `sklearn` → `pandas` → `pyarrow`. `scholar.config` imports torch at module level → CUDA/cuDNN DLLs are loaded first. When pyarrow then loads its Arrow C++ DLL, the DLL initialization order conflicts with torch's already-initialized global C++ runtime state.

**Key finding:** `import pyarrow; import sentence_transformers` always succeeds. `import torch; import sentence_transformers` always crashes. Once pyarrow's C extension is in `sys.modules`, the DLL re-initialization is skipped.

**Diagnosis:** `PYTHONFAULTHANDLER=1` to get C-level stack trace; binary-searched import orderings in isolated test scripts.

**Fix:**

```python
# First line of scholar/api/main.py AND scholar/config.py:
import pyarrow as _pa  # must load before torch — Windows DLL init order
```

**Interview:** *"Windows DLL initialization order problem — two C++ runtimes with conflicting global state. Found the conflicting pair via PYTHONFAULTHANDLER and binary search on import orderings. Fix is a single import at the top of config.py — every entrypoint that imports config gets the right order."*

---

### 11. FAISS HNSW — SIGSEGV Under asyncio on Windows Python 3.13

**Problem:** POST /search returned 200 OK once, then server died (exit code 139 / SIGSEGV, no traceback). GET /health worked fine.

**Root cause:** FAISS HNSW uses an OpenMP thread pool. On Windows + Python 3.13, HNSW's OMP initialization conflicts with asyncio's event loop. Python 3.13 tightened asyncio thread-safety requirements around C-extension thread interactions. Attempted partial fixes (omp_set_num_threads(1), ThreadPoolExecutor) all failed — OMP init happens at DLL load time.

**Fix:** Replaced FAISS ANN search with numpy brute-force matmul:

```python
# At startup: load all 200K embeddings, L2-normalize
raw = np.load(emb_path).astype(np.float32)
norms = np.linalg.norm(raw, axis=1, keepdims=True)
self._embeddings = raw / (norms + 1e-9)   # shape: (200000, 384) float32

# At query time:
query_vec = encode_and_normalize(query_text)  # (384,)
scores = self._embeddings @ query_vec          # (200000,) — one BLAS GEMV
top_idx = np.argpartition(scores, -k)[-k:]    # O(n) partial sort
top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
```

Performance: ~10ms per query for 200K × 384. Exact cosine similarity (not approximate), so Recall@k = 1.0. FAISS index still loaded for `reconstruct()` calls from eval scripts.

**Result:** No more crashes. BM25 26ms p50, Dense ~100–200ms p50 (model encoding dominates, not the matmul).

**Interview:** *"FAISS HNSW segfaulted in asyncio on Windows Python 3.13 due to OMP thread-pool initialization. Instead of fighting the thread conflict, I replaced HNSW search with numpy matmul — exact cosine in ~10ms for 200K vectors via BLAS GEMV. Tradeoff: 307MB RAM to hold embeddings (vs FAISS HNSW's ~330MB for the graph structure). At <1M vectors, brute-force is often competitive with HNSW anyway."*

---

### 12. PostgreSQL Auth — SCRAM-SHA-256 vs MD5 on Windows

**Problem:** psycopg2-binary on Windows repeatedly failed with "password authentication failed" despite correct credentials.

**Root cause — Layer 1:** psycopg2-binary Windows wheel does not support SCRAM-SHA-256. Only md5 and plain-password auth. **Layer 2:** Docker Desktop on Windows uses NAT networking — host connections appear as a bridge IP (172.17.0.1), not 127.0.0.1. The `host all all 127.0.0.1/32 trust` rule never matched; every connection hit the catch-all SCRAM rule.

**Fix:**

```sql
-- Inside container:
-- 1. Edit pg_hba.conf: scram-sha-256 → md5
-- 2. Reset password so it's stored as md5:
ALTER USER scholar WITH PASSWORD 'scholar123';
-- 3. Reload without restart:
SELECT pg_reload_conf();
```

Also switched psycopg2 connection code to direct DSN strings (not SQLAlchemy `make_url` + kwargs, which has its own Windows URL parsing issues).

**Interview:** *"Two-layer problem: psycopg2-binary Windows wheel doesn't include SCRAM support, and Docker Desktop NAT networking routed host connections through a bridge IP that bypassed the localhost trust rule. Fix: switch to MD5 auth and reload pg_hba.conf without container restart."*

---

### 13. Windows Paging File — API Server + Phi-3 Co-load Failure

**Problem:** Calling `/explain` endpoint returned `OSError: The paging file is too small`. Persisted before and after LoRA training. Code is correct.

**Root cause:** Memory budget at `/explain` call time:
- API server at startup: embeddings.npy (307MB RAM) + sentence-transformer (500MB VRAM)
- Phi-3 4-bit NF4: ~2GB VRAM + ~2GB RAM (model weights + KV cache + paged attention)
- Windows default paging file: 2–4GB (system-managed, often near-full on 16GB RAM systems)
- Combined VRAM: 500MB + 2GB = 2.5GB (within 4GB RTX 2050 limit)
- Combined RAM: 307MB + 2GB + OS overhead → exceeds paging file budget

**Options:**
1. Test Phi-3 standalone (without API server running) — clears VRAM/RAM before load
2. Increase Windows paging file to 8–16GB (Control Panel → System → Advanced → Virtual Memory)
3. Disable embedding warmup so sentence-transformer doesn't pre-load VRAM

**Interview:** *"A hardware constraint on a 4GB VRAM / 16GB RAM Windows machine with the default paging file. The code is correct; this is a resource contention issue. Production fix is either more VRAM (RTX 3060 12GB) or splitting the explanation service into a separate process that doesn't share RAM with the retrieval server."*

---

### 14. PyArrow DLL Conflict in Training Scripts

**Problem:** `python -m scholar.generation.train_lora` exits code 5, 0 bytes output, no error visible.

**Root cause:** Same DLL ordering issue as bug 10, but in training scripts. `train_lora.py` imports torch at module level. trl then imports `transformers.generation` → `sklearn` → `pyarrow`, causing the Arrow C++ DLL conflict in the training process (not the server).

**Fix:** `import pyarrow as _pa` as the FIRST import in `scholar/generation/train_lora.py` (same pattern as api/main.py).

---

### 15. Pydantic Settings — Extra Fields Forbidden for PYTHONUTF8

**Problem:** Server failed on startup with `ValidationError: Extra inputs are not permitted [type=extra_forbidden]` for the `PYTHONUTF8` key.

**Root cause:** `.env` contained `PYTHONUTF8=1` (a Python interpreter environment variable). Pydantic-settings v2 defaults to rejecting unknown fields in `BaseSettings`. Environment variable namespaces are shared — the OS fills them with many variables the application doesn't own.

**Fix:**

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",   # always set this — env namespace is shared
        ...
    )
```

**Interview:** *"Always set `extra='ignore'` in BaseSettings. The environment namespace is shared between the OS, Python runtime, Docker, and your application. You will always have unknown keys. Forbidding them causes startup failures on any machine with a non-empty environment."*

---

### 16. SciTLDR Dataset — Deprecated in datasets 3.x

**Problem:** `load_dataset("allenai/scitldr", trust_remote_code=True)` raises `DatasetNotFoundError` / "Dataset scripts are no longer supported" in HuggingFace datasets 3.x.

**Root cause:** HuggingFace deprecated loading datasets with custom Python loading scripts (`trust_remote_code=True`) in datasets 3.0. SciTLDR's dataset card still lists it but the loader script is rejected. GitHub raw JSONL URL also returned 404 (repo restructured).

**Fix:** Local ArXiv snapshot fallback in `generation/eval.py`:

```python
def _load_scitldr(self) -> list[dict]:
    try:
        return load_dataset("allenai/scitldr", ...)  # try HF first
    except Exception:
        pass
    # Fallback: use our own ArXiv corpus — titles as TLDR proxies
    papers = session.exec(select(Paper).limit(200)).all()
    return [{"abstract": p.abstract, "tldr": p.title} for p in papers]
```

**Consequence for eval:** ROUGE-L is lower than published SciTLDR benchmarks because a 10-word title is a much shorter reference than a human-written TLDR sentence. BERTScore F1 remains meaningful. See §8 of the interview pack for the framing.

---

### 17. BERTScore VRAM Blocks Phi-3 Reload in Eval

**Problem:** Generation eval ran base model successfully but crashed on LoRA phase reload: `ValueError: Some modules are dispatched on the CPU or the disk` during second Phi-3 load.

**Root cause:** After base model evaluation, BERTScore (roberta-large, loaded into VRAM) was computed immediately. Resetting `Phi3Generator._instance = None` and calling `gc.collect()` didn't immediately free VRAM occupied by roberta-large. Second Phi-3 4-bit load then failed because VRAM was partially occupied by roberta-large, causing accelerate to dispatch some Phi-3 layers to CPU — an invalid state for 4-bit NF4 models.

**Fix — Three-phase eval:**

```python
# Phase 1: base model inference (Phi-3 base, no VRAM for BERTScore)
base_hyps, base_refs = evaluate_level(use_adapter=False, ...)

# Phase 2: free VRAM completely
Phi3Generator._instance = None
gc.collect()
torch.cuda.empty_cache()

# Phase 3: LoRA model inference
lora_hyps, lora_refs = evaluate_level(use_adapter=True, ...)

# Phase 4: free VRAM again
Phi3Generator._instance = None
gc.collect()
torch.cuda.empty_cache()

# Phase 5: BERTScore bulk pass (both at once, model loaded once)
bertscore_base = score(base_hyps, base_refs, model_type="roberta-large", ...)
bertscore_lora = score(lora_hyps, lora_refs, model_type="roberta-large", ...)
```

**Interview:** *"4GB VRAM requires careful sequencing of models. BERTScore's roberta-large (~1.4GB VRAM) and Phi-3 4-bit (~2GB VRAM) can't coexist. Explicit VRAM flush between phases + deferred BERTScore computation (gather all hypotheses first, score in one batch at the end) was the clean solution."*

---

## Part D — Build & Tooling Fixes

### 18. BM25 Index Build — --skip-ingest Flag Bug

**Problem:** `python scripts/build_index.py --skip-ingest` skipped both the loader AND the FAISS build. Nothing rebuilt.

**Root cause:**

```python
if run_faiss and not skip_ingest:
    # Step 1: loader
    # Step 2: FAISS  ← incorrectly nested inside skip_ingest guard
```

**Fix:**

```python
if run_faiss:
    if not skip_ingest:
        # Step 1: loader (skipped when --skip-ingest)
        ...
    # Step 2: FAISS (always runs when run_faiss=True)
    ...
```

**Interview:** *"Classic flag interaction bug — a single boolean doing two jobs. Decompose conditions to clarify semantics. Always manually test each flag combination, especially ones meant to be independent."*

---

### 19. CiteULike Loader — Wrong File Format

**Problem:** Loader exited immediately with "CiteULike data not found" even though files existed.

**Root cause:** Code expected `papers.txt` (tab-delimited). CiteULike-A repo ships `raw-data.csv` (comma-separated, headers: `doc.id, title, citeulike.id, raw.title, raw.abstract`).

**Fix:** Check for `raw-data.csv` first, fall back to `papers.txt`:

```python
if csv_path.exists():
    with open(csv_path, ...) as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = int(row["doc.id"])
            title = row.get("raw.title") or row.get("title") or ""
            titles[doc_id] = title.strip()
```

**Result:** 5551 users, 1716 interactions loaded. 158 papers matched to ArXiv corpus (expected — CiteULike includes conference/journal papers not on ArXiv).

---

### 20. Docker Networking — API Container Cannot Reach DB

**Problem:** All DB-backed endpoints returned 500 in Docker. Direct uvicorn on host worked fine.

**Root cause:** `DATABASE_URL` pointed to `localhost:5432`. Inside a container, `localhost` = the container itself, not the host. PostgreSQL is a separate `db` service.

**Fix (development):** Run uvicorn directly on host: `uvicorn scholar.api.main:app --reload-dir scholar --port 8000`

**Fix (production Docker):** Override DATABASE_URL in docker-compose.yml:

```yaml
environment:
  - DATABASE_URL=postgresql+asyncpg://scholar:${POSTGRES_PASSWORD}@db:5432/scholar
```

**Interview:** *"Each container has its own network namespace. In Docker Compose, services reach each other by service name on the shared bridge network. Never hardcode localhost in service URLs — always use configurable hostnames from environment variables."*

---

### 21. Embeddings Pipeline — Silent Crash & Batch Rewrite

**Problem:** `python scripts/build_index.py --faiss-only` appeared to run but produced no output and no index file.

**Root cause — 3 layers:**
1. Rich's Unicode box-drawing characters raised `UnicodeEncodeError` on Windows terminal inside the error handler → error swallowed
2. SQLAlchemy `make_url` + kwargs connection — Windows compatibility issues
3. Individual `UPDATE` per paper: 200K round-trips to PostgreSQL

**Fix:**
- All output: plain `print()` only, no Rich
- DB connection: direct psycopg2 DSN string
- Paper loading: single `SELECT id, abstract FROM papers ORDER BY id` with `fetchmany(1000)` cursor batching
- Embedding updates: `executemany` with 1000-row batch commits

**Result:** 200K embeddings built in ~5700s (~95min) on CPU. Previous individual-update approach would take 10x longer.

**Interview:** *"The error handler itself was crashing on Windows due to Rich's Unicode output — swallowing the real error completely. Lesson: never use rich formatting inside exception handlers. The performance fix was standard batch processing: stream 200K rows in chunks, bulk-update with executemany."*

---

### 22. Ruff Lint — ML Naming Convention Conflicts

**Problem:** `ruff check scholar/` reported 31 errors — N806 (X, X_train, X_val, N, P, R, F1), N812 (import as F), E501 (SQL strings).

**Root cause:** PEP 8 naming rules conflict with ML/scientific computing conventions universal since Hastie 2001 (X, y) and Robertson 1994 (N for corpus size).

**Fix in pyproject.toml:**

```toml
[tool.ruff.lint]
line-length = 120
ignore = ["N806", "N812"]
```

**Result:** 0 ruff errors. `pytest` → 63 passed.

**Interview:** *"ML code has decades-old naming conventions that predate PEP 8. X, N, P, R, F1 are standard notation from seminal papers, not magic variables. Configure linters for the domain; suppress globally rather than littering the codebase with # noqa comments. Document WHY in pyproject.toml."*

---

### 23. trl 1.4.0 — Breaking API Changes in SFTTrainer

**Problem:** `train_lora.py` failed with `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`.

**Root cause:** trl 1.4.0 (released 2025-02) breaking changes: `tokenizer=` → `processing_class=`; `max_seq_length`, `packing`, `dataset_text_field` moved from SFTTrainer to new `SFTConfig` class.

**Fix:**

```python
# OLD (trl <=1.3.x):
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(output_dir=..., ...),
    max_seq_length=768, dataset_text_field="text", packing=False,
)

# NEW (trl 1.4.0+):
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    max_length=768,
    dataset_text_field="text",
    packing=False,
    ...
)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)
```

**Interview:** *"trl is a rapidly evolving library — pin your version or read the CHANGELOG before upgrading. The processing_class rename broke every trl < 1.4 project without a deprecation warning."*

---

### 24. trust_remote_code=True Breaks with Transformers 5.x

**Problem:** `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'` when loading Phi-3 with `trust_remote_code=True`.

**Root cause:** `trust_remote_code=True` tells HuggingFace to execute the model's custom Python code from the Hub (modeling_phi3.py). This file was written for transformers 4.x and uses removed DynamicCache APIs. With `trust_remote_code=False`, transformers uses its own built-in Phi-3 implementation (compatible with 5.x).

**Fix:** Always use `trust_remote_code=False` for Phi-3. The monkey-patch in bug 8 handles the cache compatibility for the built-in implementation.

---

## Completed Items from Future Optimizations

### QLoRA Training — Key Results

- **Total training:** 114 steps, 3 epochs, ~10.3 hours on RTX 2050 4GB VRAM
- **Loss trajectory:** Step 80: 1.47 → Step 110: 1.21 → converged
- **Adapter size:** 6.3MB (adapter_model.safetensors) + tokenizer files
- **Config:** batch=1, grad_accum=8, seq_len=768, paged_adamw_8bit, warmup_steps=5, lr=2e-4
- **BERTScore F1:** 0.840–0.847 (consistent base and LoRA — LoRA improves level appropriateness quality, not raw score)

### LightGBM LTR — Feature Set

7 features per (user, document) pair:
1. BM25 score (sparse retrieval relevance)
2. Dense cosine similarity (semantic relevance)
3. User-doc latent similarity (VAE z_user · paper_embedding)
4. Paper recency (days since publication, log-scaled)
5. Citation count (log-scaled)
6. Topic match indicator (user's dominant topics vs paper categories)
7. RRF fused score (from hybrid retrieval)

LightGBM chosen over neural reranker: trains in 30s, interpretable feature importances, no GPU at inference, production-standard (LinkedIn, Yandex use this pattern).

### ε-Greedy Exploration

`ε=0.1` means 1 in 10 results is a randomly sampled "exploration paper" outside the user's normal topic distribution. Measured effect: ILD 0.496 → 0.576 (+16.2%) vs Mult-VAE alone. Topic Entropy@10 preserved at ~2.4 vs collapse to ~0.8 without exploration (20-session simulation).
