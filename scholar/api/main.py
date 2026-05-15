"""FastAPI application entry point with lifespan management."""
# ruff: noqa: I001  (pyarrow must be imported before torch — see DLL fix below)
from __future__ import annotations

# Windows Python 3.13 fix: pyarrow C extension DLL must load before any package
# (torch, faiss) that initializes its own OMP/threading runtime.  If pyarrow
# loads second it hits an access-violation at pyarrow.lib load time.  Importing
# pyarrow here — before scholar.config (which imports torch) — fixes it.
import pyarrow as _pa  # noqa: F401 E402 I001

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from scholar.config import settings


def _create_tables_sync() -> None:
    import psycopg2

    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    sql = schema_path.read_text()

    conn = psycopg2.connect(str(settings.database_url_sync))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


def _load_bm25():
    bm25_path = Path("./data/bm25.pkl")
    if not bm25_path.exists():
        return None
    try:
        from scholar.retrieval.bm25 import BM25

        return BM25.load(bm25_path)
    except Exception as e:
        print(f"Warning: could not load BM25 index: {e}")
        return None


def _load_dense():
    try:
        from scholar.retrieval.dense import DenseRetriever

        retriever = DenseRetriever()
        retriever.load()
        return retriever
    except Exception as e:
        print(f"Warning: could not load FAISS index: {e}")
        return None


def _load_ltr():
    ltr_path = Path("./data/ltr_model.lgb")
    if not ltr_path.exists():
        return None
    try:
        from scholar.recsys.ltr import LTRModel

        ltr = LTRModel()
        if ltr.load(ltr_path):
            return ltr
        return None
    except Exception as e:
        print(f"Warning: could not load LTR model: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Starting Scholar API ...")

    try:
        _create_tables_sync()
        print("DB schema applied.")
    except Exception as e:
        print(f"Warning: could not apply DB schema: {e}")

    engine = create_async_engine(settings.database_url, echo=False)
    async_session_factory = sessionmaker(  # type: ignore[call-overload]  # async engine typing
        engine, class_=AsyncSession, expire_on_commit=False
    )
    app.state.async_session_factory = async_session_factory
    app.state.engine = engine

    bm25 = _load_bm25()
    app.state.bm25 = bm25
    if bm25:
        print("BM25 index loaded.")
    else:
        print("BM25 index not found — run scripts/build_index.py first.")

    print("Loading FAISS index ...")
    dense = _load_dense()
    app.state.dense_retriever = dense
    if dense:
        print("FAISS index loaded.")
    else:
        print("FAISS index not found — run scripts/build_index.py first.")

    if bm25 and dense:
        from scholar.retrieval.hybrid import HybridRetriever

        app.state.hybrid_retriever = HybridRetriever(bm25=bm25, dense=dense)
        print("HybridRetriever ready.")
    else:
        app.state.hybrid_retriever = None

    if settings.warmup_retrieval and dense is not None:
        print("Warming up sentence-transformer model...")
        dense._encode_and_normalize("warmup")
        print("Sentence-transformer warm.")

    # Cache id→faiss_index lookup once at startup (O(N) dict, amortised over all requests)
    if dense is not None:
        app.state.id_to_idx = {pid: i for i, pid in enumerate(dense._paper_ids)}
        print(f"id_to_idx cache built ({len(app.state.id_to_idx)} entries).")
    else:
        app.state.id_to_idx = {}

    ltr = _load_ltr()
    app.state.ltr_model = ltr
    if ltr:
        print("LTR model loaded.")
    else:
        print("LTR model not found — using FAISS ranking (run scripts/train_ltr.py to build).")

    # Optional: pre-warm Phi-3 generator to avoid 2-4 min cold start on first /explain
    if settings.warmup_generation and settings.generation_available:
        print("Pre-warming Phi-3 generator (WARMUP_GENERATION=true)...")
        try:
            from scholar.generation.inference import Phi3Generator
            Phi3Generator.get_instance()
            print("Phi-3 generator ready.")
        except Exception as e:
            print(f"Warning: generator warmup failed: {e}")

    yield

    print("Shutting down Scholar API ...")
    await engine.dispose()


app = FastAPI(
    title="Scholar API",
    description="Personalized academic paper discovery",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from scholar.api.routes import explain, recommend, search, user  # noqa: E402

app.include_router(search.router, tags=["search"])
app.include_router(explain.router, tags=["explain"])
app.include_router(user.router, tags=["user"])
app.include_router(recommend.router, tags=["recommend"])


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
