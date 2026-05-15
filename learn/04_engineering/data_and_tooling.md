# Data Layer and Python Tooling — Complete Deep Dive

> Covers: PostgreSQL + pgvector, asyncpg, SQLModel, connection pooling, Scholar's schema,
> plus uv, ruff, mypy, pytest, pydantic-settings — every tool in Scholar's dev stack.

---

## 1. Why PostgreSQL for Scholar

The data layer has two distinct needs:

**Relational metadata:** paper titles, authors, categories, dates, citation counts, user histories —
structured, queryable, filterable by SQL WHERE clauses.

**Vector embeddings:** 384-dimensional sentence-BERT embeddings for 200K papers (~307MB) —
needs nearest-neighbor search, not SQL conditions.

The traditional answer is "two systems": PostgreSQL for metadata + Pinecone/Qdrant for vectors.
Scholar uses **one system**: PostgreSQL 16 + pgvector extension.

| Decision | Reasoning |
|----------|-----------|
| One DB for relational + vector | One credential, one backup, one query language, zero extra infrastructure |
| asyncpg driver | Non-blocking I/O matching FastAPI's async architecture, 3-5× faster than psycopg2 |
| SQLModel ORM | Define schema once as Python class, get both DB table AND Pydantic validation |
| pgvector HNSW | ANN search within SQL queries, enabling filtered vector search |

---

## 2. Relational Model and SQL

### Schema Design

```sql
-- scholar/db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id             TEXT PRIMARY KEY,        -- ArXiv ID: "2301.00001"
    title          TEXT NOT NULL,
    abstract       TEXT NOT NULL,
    authors        TEXT NOT NULL,           -- comma-separated string
    categories     TEXT NOT NULL,           -- "cs.LG cs.AI stat.ML"
    update_date    DATE,
    citation_count INTEGER NOT NULL DEFAULT 0,
    embedding      vector(384)              -- all-MiniLM-L6-v2 output
);

CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,            -- UUID string
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_history (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    paper_id   TEXT NOT NULL REFERENCES papers(id),
    clicked_at TIMESTAMP,
    PRIMARY KEY (user_id, paper_id)         -- prevents duplicate clicks
);

CREATE TABLE IF NOT EXISTS click_logs (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT,                     -- nullable (anonymous allowed)
    paper_id      TEXT,
    query         TEXT,
    rank_position INTEGER,
    clicked_at    TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers (categories);
CREATE INDEX IF NOT EXISTS idx_papers_update_date ON papers (update_date);
CREATE INDEX IF NOT EXISTS idx_history_user ON user_history (user_id);
CREATE INDEX IF NOT EXISTS idx_papers_embedding
    ON papers USING hnsw (embedding vector_cosine_ops);
```

**Design decisions:**
- `id TEXT PRIMARY KEY`: ArXiv IDs are natural unique keys ("2301.00001"). No surrogate int key needed.
- `authors TEXT` not `TEXT[]`: Simpler storage and ORM handling. Application splits on commas.
- `user_history PRIMARY KEY (user_id, paper_id)`: Schema-enforced deduplication — inserting
  the same click twice fails gracefully via ON CONFLICT.
- `click_logs` nullable fields: Tolerant logging — records even if user_id or paper_id is unavailable.
- `ON DELETE CASCADE` on user_history: Deleting a user automatically removes all their history.

### Key SQL Patterns

**UPSERT (used in ArXiv loader):**
```sql
INSERT INTO papers (id, title, abstract, categories, update_date, citation_count)
VALUES ('2301.00001', 'Attention Is All You Need', '...', 'cs.LG cs.AI', '2023-01-01', 10000)
ON CONFLICT (id) DO UPDATE
SET title = EXCLUDED.title,
    abstract = EXCLUDED.abstract,
    citation_count = GREATEST(papers.citation_count, EXCLUDED.citation_count);
-- EXCLUDED = the row we tried to insert (the "excluded" conflicting row)
-- GREATEST: only increase citation count, never decrease
```

**Batch fetch by IDs (N+1 prevention):**
```sql
SELECT * FROM papers WHERE id IN ('id1', 'id2', ..., 'id200');
-- One query for 200 papers — NOT 200 queries of 1 paper each
```

**pgvector ANN query:**
```sql
SELECT id, title, embedding <=> $1 AS distance
FROM papers
ORDER BY distance
LIMIT 200;
-- <=> = cosine distance operator (1 - cosine_similarity)
-- With HNSW index: O(log N) approximate search
```

### ACID Properties

| Property | Meaning | Scholar Example |
|----------|---------|-----------------|
| Atomicity | Transaction completes entirely or not at all | Insert user + history: both succeed or neither |
| Consistency | DB stays in valid state | FK constraints: can't add history for nonexistent user |
| Isolation | Concurrent transactions don't interfere | Two /search requests don't corrupt each other |
| Durability | Committed data survives crashes | Click logged even if server restarts immediately after |

PostgreSQL implements isolation via MVCC (Multi-Version Concurrency Control): each transaction
sees a consistent snapshot. Readers don't block writers.

---

## 3. pgvector — Vector Search in PostgreSQL

```sql
-- Three distance operators
embedding <->  $1   -- L2 (Euclidean) distance
embedding <=>  $1   -- Cosine distance (1 - cosine_similarity)
embedding <#>  $1   -- Negative inner product

-- For unit-normalized embeddings (sentence-BERT): cosine == inner product → same ordering
-- Scholar uses <=> (cosine distance)
```

### HNSW vs IVFFlat in pgvector

```sql
-- HNSW (Scholar's choice)
CREATE INDEX idx_papers_embedding ON papers
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
-- m = bidirectional connections per node (higher = better recall, more memory)
-- ef_construction = search width during build (higher = better quality, slower build)

SET hnsw.ef_search = 100;   -- query-time recall/speed tradeoff

-- IVFFlat (alternative)
CREATE INDEX idx_papers_embedding_ivf ON papers
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);   -- number of clusters from k-means
```

| | HNSW | IVFFlat |
|--|------|---------|
| Build time | Slower (no training needed) | Faster (k-means training) |
| Query time | ~1-5ms | Slightly faster when tuned |
| Recall@10 | 95-99% | 90-95% |
| Memory | ~2× flat | Same as flat |
| Best for | Default — no training required | Memory-constrained systems |

### Why pgvector vs Dedicated Vector DBs

| | pgvector | Pinecone | Qdrant | Weaviate |
|--|---------|----------|--------|---------|
| Ops complexity | Low (already have Postgres) | Medium (SaaS) | Medium | High |
| Cost | $0 extra | $$$+ | $0 self-hosted | $0 self-hosted |
| Recall | 95%+ HNSW | 99% | 99% | 99% |
| Scale | ~10M vectors comfortable | Billions | ~100M | ~100M |
| SQL joins + vector | Yes | No | No | No |
| Scholar use case | ✓ | Overkill | Overkill | Overkill |

At 200K papers, pgvector is trivially fast. The SQL join capability
(filter by date + vector similarity in one query) is a real advantage.

---

## 4. SQLModel — ORM for FastAPI

SQLModel combines SQLAlchemy Core with Pydantic. One class = database table + API schema.

```python
# scholar/db/models.py
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from datetime import date, datetime
from typing import Optional, List

class Paper(SQLModel, table=True):
    __tablename__ = "papers"

    id: str = Field(primary_key=True)
    title: str
    abstract: str
    authors: str
    categories: str
    update_date: Optional[date] = Field(default=None)
    citation_count: int = Field(default=0)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(384)),  # pgvector type — SQLModel doesn't know it natively
    )

class UserHistory(SQLModel, table=True):
    __tablename__ = "user_history"
    user_id: str = Field(foreign_key="users.id", primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    clicked_at: Optional[datetime] = Field(default=None)
```

**`table=True`** makes the class both an ORM table definition AND a Pydantic model.
Without it, it's a pure Pydantic model (useful for API response schemas with no DB backing).

**`sa_column=Column(Vector(384))`**: when SQLModel doesn't support a column type natively
(like pgvector's vector), pass a raw SQLAlchemy column object.

---

## 5. asyncpg — Non-Blocking Database Calls

```python
# scholar/api/main.py — connection setup
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "postgresql+asyncpg://scholar:pass@db:5432/scholar"
# "+asyncpg" suffix tells SQLAlchemy to use asyncpg driver
engine = create_async_engine(settings.database_url, echo=False)

async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # expire_on_commit=False: don't mark objects stale after commit
    # Without this, accessing obj.title after commit triggers a new SELECT — bad in async
)
```

### Making Async Queries

```python
# SELECT with ORM
async with async_session_factory() as session:
    result = await session.execute(
        select(Paper).where(Paper.id.in_(candidate_ids))
    )
    papers = result.scalars().all()   # list of Paper objects
    papers_map = {p.id: p for p in papers}   # O(1) lookup dict

# INSERT
async with async_session_factory() as session:
    session.add(ClickLog(user_id=uid, paper_id=pid, query=q, rank_position=rank))
    await session.commit()
```

**`expire_on_commit=False`:** by default, SQLAlchemy marks all ORM objects "expired" after
commit — accessing any attribute triggers a new SELECT. In async code, this causes implicit
queries in unexpected places. Setting to False keeps attribute values intact after commit.

**asyncpg vs psycopg2:**

| | asyncpg | psycopg2 |
|--|---------|----------|
| Protocol | Binary (Cython-compiled) | Text (default) |
| Throughput | ~50k QPS | ~15k QPS |
| Event loop | Zero blocking (native async) | Blocks entire loop |
| Use in Scholar | All runtime queries | Only startup DDL |

asyncpg is used for all FastAPI requests. psycopg2 is used only in `_create_tables_sync()`
because that runs before the async event loop starts.

---

## 6. Connection Pooling

Opening a Postgres connection: TCP handshake + auth + Postgres allocating a process (~1MB RAM).
This takes ~50-100ms. A connection pool reuses open connections.

```python
engine = create_async_engine(
    settings.database_url,
    pool_size=10,         # 10 persistent connections always open
    max_overflow=20,      # up to 20 extra in bursts
    pool_timeout=30,      # wait up to 30s for free connection before error
    pool_recycle=1800,    # recycle connections every 30min (prevent stale sockets)
)
# Scholar uses defaults (pool_size=5, max_overflow=10) — fine for prototype
```

**Postgres `max_connections` default = 100.** With Scholar's 3 services + each having
a small pool, this is comfortably within limits.

---

## 7. Python Tooling — uv, ruff, mypy, pytest, pydantic-settings

### uv — Fast Package Manager

```bash
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create venv and install (10-100x faster than pip)
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"    # editable install with dev extras
# pip equivalent takes 8+ minutes; uv takes ~45 seconds

# Add dependencies
uv add torch                   # adds to [dependencies]
uv add --dev pytest ruff mypy  # adds to [dev-dependencies]
```

**pyproject.toml — central config:**
```toml
[project]
name = "scholar"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111", "uvicorn[standard]>=0.29", "sqlmodel>=0.0.18",
    "asyncpg>=0.29", "pgvector>=0.2", "sentence-transformers>=2.7",
    "torch>=2.2", "transformers>=4.40", "peft>=0.11", "bitsandbytes>=0.43",
    "lightgbm>=4.3", "gradio>=4.31", "pydantic-settings>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.10"]

[tool.setuptools.packages.find]
where = ["."]
include = ["scholar*"]
```

### ruff — Linter + Formatter

ruff replaces flake8 + pylint + isort + black in one Rust binary, ~100× faster.

```toml
[tool.ruff]
line-length = 120     # Scholar extended from 100 (SQL strings, Markdown tables)
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "B", "UP"]
ignore = [
    "N806",   # uppercase vars in functions (ML convention: X_train, N, P, R, F1)
    "N812",   # lowercase module as non-lowercase (torch.nn.functional as F)
]
```

```bash
ruff check scholar/          # find issues (read-only)
ruff check --fix scholar/    # auto-fix what's fixable
ruff format scholar/         # format like black
```

**Common Scholar catches:**
- `F401` unused imports (during active development, leftover imports)
- `I001` import ordering (stdlib before third-party before local)
- `B006` mutable default arguments (`def f(x=[])` → shared instance bug)

### mypy — Static Type Checker

```toml
[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true    # don't fail on untyped 3rd-party libs
warn_return_any = true
```

```bash
mypy scholar/    # check entire package (0 errors in Scholar's final state)
```

**Type annotation patterns in Scholar:**
```python
def extract_features(
    user_emb: np.ndarray,
    candidate_ids: list[str],
    candidate_scores: dict[str, float],
    paper_meta: dict[str, "Paper"],
    history_ids: set[str] | None = None,
) -> np.ndarray: ...

async def get_paper(paper_id: str) -> Optional[Paper]:
    result = await session.execute(select(Paper).where(Paper.id == paper_id))
    return result.scalars().first()   # Optional[Paper] — mypy knows this
```

**What mypy catches in Scholar:**
```python
# Wrong return type (list vs dict)
def get_scores() -> dict[str, float]:
    return [...]  # mypy: error: incompatible return type

# None not handled
paper = await get_paper("id")
paper.title  # mypy: error: Item "None" has no attribute "title"
if paper:
    paper.title  # OK — mypy narrows type to Paper
```

### pytest — Test Framework

```
tests/
├── test_bm25.py     # BM25 scoring correctness, save/load roundtrip
├── test_dense.py    # Dense retrieval, embedding cosine properties
├── test_hybrid.py   # RRF fusion ordering
├── test_recsys.py   # Mult-VAE, LTR features, MMR, ε-greedy
└── test_api.py      # FastAPI endpoints via ASGI test client
```

**Test patterns:**
```python
# Fixtures — reusable setup
@pytest.fixture
def sample_bm25():
    bm25 = BM25()
    bm25.fit(["attention transformers neural", "diffusion image generation"], ["p1", "p2"])
    return bm25

# Parametrize — same test, many inputs
@pytest.mark.parametrize("query,expected_top", [
    ("attention transformers", "p1"),
    ("diffusion image", "p2"),
])
def test_bm25_ranking(sample_bm25, query, expected_top):
    results = sample_bm25.query(query, top_k=1)
    assert results[0][0] == expected_top

# Async API tests
@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"     # all async tests auto-detected
testpaths = ["tests"]
addopts = "-v --tb=short"
```

```bash
pytest                    # all 63 tests
pytest -x                 # stop at first failure
pytest -k "not integration"  # skip tests requiring DB/GPU
pytest --cov=scholar      # with coverage
```

**Scholar's test strategy:**
```
Unit (fast, no I/O): BM25 scoring, RRF ordering, MMR diversity, ε-greedy injection,
                     KL annealing β, LTR feature extraction, VAE forward pass shape
Integration (require infrastructure): API endpoints, DB queries, real FAISS index
Marked slow: Phi-3 generation (60s+) — @pytest.mark.slow
```

### pydantic-settings — Typed Configuration

```python
# scholar/config.py
import pyarrow as _pa   # FIRST import (Windows DLL fix)
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str                          # required — fails fast if missing
    database_url_sync: str
    faiss_index_path: Path = Path("./data/faiss.index")
    bm25_path: Path = Path("./data/bm25.pkl")
    hf_home: str = "./data/hf_cache"
    lora_adapter_path: Optional[Path] = None
    device: str = "cuda"
    generation_available: bool = True
    warmup_generation: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",   # IMPORTANT: ignore PYTHONUTF8 and other env vars not in schema
    )

settings = Settings()   # singleton — import this everywhere
```

**Priority (highest first):**
1. Environment variables (`export DATABASE_URL=...`)
2. `.env` file contents
3. Default values in the model

`extra="ignore"` is required because Windows adds environment variables like `PYTHONUTF8=1`
that pydantic-settings would otherwise reject as unexpected fields.

---

## 8. The Complete Development Workflow

```bash
# ── Initial setup ─────────────────────────────────────────
uv venv
.venv\Scripts\activate    # Windows
uv pip install -e ".[dev]"
cp .env.example .env      # fill in DATABASE_URL etc.

# ── Before every commit ───────────────────────────────────
ruff check --fix scholar/ && ruff format scholar/
mypy scholar/
pytest -x --tb=short

# ── Run full stack ────────────────────────────────────────
docker compose up

# ── Build indexes (one-time after data ingestion) ─────────
python -m scholar.ingest.arxiv_loader --limit 200000
python scripts/build_index.py    # BM25 + embeddings

# ── Full pipeline ─────────────────────────────────────────
python -m scholar.ingest.citeulike_loader --data-dir ./data/citeulike-a
python scholar/recsys/train_vae.py
python scripts/train_ltr.py
python scholar/generation/synth_data.py
python scholar/generation/train_lora.py
python -m scholar.retrieval.eval
python -m scholar.recsys.eval
python -m scholar.generation.eval
```

---

## 9. Interview Questions and Answers

**Q: Why store embeddings in both Postgres (pgvector) and a FAISS/numpy file?**

A: Different access patterns. The numpy embeddings.npy (~307MB) is loaded into RAM at startup
and used for every API search request: `scores = embeddings @ query_vec` — one matrix multiply,
~10ms. This is the hot path. pgvector stores the canonical embeddings in Postgres for cold
operations: LTR training (need to retrieve arbitrary paper embeddings), batch embedding jobs,
and SQL queries combining metadata filters with similarity search. Running both means 2×
storage (614MB total) but zero extra infrastructure — just one Postgres instance.

**Q: What is MVCC and why does Postgres use it?**

A: Multi-Version Concurrency Control. Instead of locking rows on write, Postgres creates a new
version of the row. Readers always see the old version until the writer commits. This means
reads never block writes and writes never block reads — Scholar can serve search requests while
ingesting new papers without any locking. The trade-off: old row versions accumulate and need
VACUUM to reclaim space. For Scholar's read-heavy workload (mostly reads, occasional ingestion
batches), MVCC is ideal.

**Q: How does asyncpg differ from psycopg2, and why does it matter?**

A: Three differences. First, asyncpg is async-native — it never blocks the event loop. psycopg2
is synchronous and would block FastAPI's event loop during every DB query, preventing other
requests from being processed. Second, asyncpg uses Cython-compiled binary protocol parsing
(3-5× faster than psycopg2's text protocol). Third, asyncpg enables genuine concurrency: while
waiting for DB results, other requests run. Scholar uses asyncpg for all runtime queries and
psycopg2 only in `_create_tables_sync()` at startup, before the event loop initializes.

**Q: What does `uv pip install -e ".[dev]"` do, and why editable?**

A: `-e` (editable) installs the package by creating a `.pth` file pointing to your project
directory instead of copying files to site-packages. Changes to source code are immediately
visible without reinstalling — critical during development. `[dev]` installs optional
dependencies from `[project.optional-dependencies] dev = [...]` in pyproject.toml (pytest,
ruff, mypy). `".[dev]"` means "the current directory (.) plus the dev extras". uv is 10-100×
faster than pip for the same installation because it's written in Rust with parallel metadata
downloads and a SAT-solver-based dependency resolver.

**Q: What is `ON CONFLICT DO UPDATE` in SQL and why does Scholar use it?**

A: It's an upsert (update-or-insert). If an INSERT would violate a UNIQUE constraint, instead
of raising an error, Postgres updates the conflicting row. `EXCLUDED` refers to the values we
tried to insert. Scholar uses this in the ArXiv loader: running the loader twice won't create
duplicate papers, and it updates metadata (title, abstract) if the paper has been revised.
`GREATEST(papers.citation_count, EXCLUDED.citation_count)` ensures we only increase citation
count, never decrease it from a stale snapshot.

---

## 10. World-Class Resources

**PostgreSQL:**
- [PostgreSQL 16 Docs](https://www.postgresql.org/docs/16/) — official manual, well-written
- [Use the Index, Luke](https://use-the-index-luke.com/) — SQL performance and index design
- [pgvector GitHub](https://github.com/pgvector/pgvector) — HNSW/IVFFlat options and benchmarks
- [HNSW paper (Malkov & Yashunin 2018)](https://arxiv.org/abs/1603.09320)

**Python Tooling:**
- [uv docs](https://docs.astral.sh/uv/) — complete reference including workspaces
- [ruff docs](https://docs.astral.sh/ruff/) — all lint rules and config options
- [mypy docs](https://mypy.readthedocs.io/en/stable/) — type system reference
- [pytest docs](https://docs.pytest.org/en/stable/) — fixtures, marks, plugins
- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Hypermodern Python (Claudio Jolowicz)](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/) — complete modern Python project setup
