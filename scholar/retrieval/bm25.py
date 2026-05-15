"""BM25 implementation from scratch — no rank_bm25 dependency."""
from __future__ import annotations

import argparse
import asyncio
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _tokenize(text: str) -> list[str]:
    return re.split(r"[^a-z0-9]+", text.lower())


class InvertedIndex:
    """Inverted index: term -> list of (doc_id, term_freq)."""

    def __init__(self) -> None:
        self.postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_lengths: Dict[str, int] = {}
        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0

    def build(self, docs: list[str], doc_ids: list[str]) -> None:
        self.num_docs = len(docs)
        total_length = 0

        for doc_id, doc in zip(doc_ids, docs):
            tokens = _tokenize(doc)
            tokens = [t for t in tokens if t]
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len

            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1

            for term, freq in term_counts.items():
                self.postings[term].append((doc_id, freq))

        self.avg_doc_length = total_length / self.num_docs if self.num_docs else 0.0

    def doc_freq(self, term: str) -> int:
        return len(self.postings.get(term, []))


class BM25:
    """BM25 retrieval model built on top of InvertedIndex."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.index: InvertedIndex = InvertedIndex()
        # Numpy cache — built after fit() or load()
        self._np_doc_indices: Dict[str, np.ndarray] = {}  # term -> int32 doc indices
        self._np_tfs: Dict[str, np.ndarray] = {}          # term -> float32 tf values
        self._np_doc_lengths: Optional[np.ndarray] = None  # float32 array [n_docs]
        self._pid_to_int: Dict[str, int] = {}
        self._int_to_pid: List[str] = []
        self._cache_ready: bool = False

    def _build_numpy_cache(self) -> None:
        """Pre-compute integer-indexed numpy arrays from string postings for fast scoring."""
        idx = self.index
        n = idx.num_docs
        if n == 0:
            return

        # Build pid <-> int mapping in stable order
        # Use sorted order for reproducibility; order only affects array layout
        all_pids: list[str] = list(idx.doc_lengths.keys())
        self._int_to_pid = all_pids
        self._pid_to_int = {pid: i for i, pid in enumerate(all_pids)}

        # doc_lengths as float32 array indexed by int doc id
        dl_arr = np.empty(n, dtype=np.float32)
        for pid, dl in idx.doc_lengths.items():
            i = self._pid_to_int[pid]
            dl_arr[i] = dl
        self._np_doc_lengths = dl_arr

        # Convert each posting list to numpy arrays
        self._np_doc_indices = {}
        self._np_tfs = {}
        for term, posting in idx.postings.items():
            if not posting:
                continue
            doc_ints = np.empty(len(posting), dtype=np.int32)
            tfs = np.empty(len(posting), dtype=np.float32)
            for j, (pid, tf) in enumerate(posting):
                doc_ints[j] = self._pid_to_int[pid]
                tfs[j] = tf
            self._np_doc_indices[term] = doc_ints
            self._np_tfs[term] = tfs

        self._cache_ready = True

    def fit(self, docs: list[str], doc_ids: list[str]) -> None:
        self.index.build(docs, doc_ids)
        self._build_numpy_cache()

    def _query_numpy(self, tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        """Vectorized BM25 scoring using numpy arrays."""
        n = self.index.num_docs
        scores = np.zeros(n, dtype=np.float64)
        N = n
        avg_dl = self.index.avg_doc_length
        k1 = self.k1
        b = self.b

        dl_arr = self._np_doc_lengths
        assert dl_arr is not None  # always set by _build_numpy_cache before _cache_ready=True

        for term in tokens:
            doc_ints = self._np_doc_indices.get(term)
            if doc_ints is None:
                continue
            tfs = self._np_tfs[term]
            df = len(doc_ints)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            dl = dl_arr[doc_ints]
            norm = k1 * (1.0 - b + b * dl / avg_dl) if avg_dl else 1.0
            tf_score = (tfs * (k1 + 1.0)) / (tfs + norm)
            # Each doc appears at most once per term's posting list, so direct
            # fancy-index addition is safe (no duplicate-index accumulation needed).
            scores[doc_ints] += idf * tf_score

        if top_k >= n:
            # Return all in sorted order
            sorted_ints = np.argsort(scores)[::-1]
            return [(self._int_to_pid[i], float(scores[i])) for i in sorted_ints if scores[i] > 0]

        # O(n) partial sort — much faster than full sort for large n
        k = min(top_k, n)
        top_ints = np.argpartition(scores, -k)[-k:]
        top_ints = top_ints[np.argsort(scores[top_ints])[::-1]]
        return [(self._int_to_pid[i], float(scores[i])) for i in top_ints if scores[i] > 0]

    def query(self, q: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = [t for t in _tokenize(q) if t]
        if not tokens:
            return []

        if self._cache_ready:
            return self._query_numpy(tokens, top_k)

        # Fallback pure-Python path (used before cache is built)
        scores: Dict[str, float] = defaultdict(float)
        N = self.index.num_docs
        avg_dl = self.index.avg_doc_length

        for term in tokens:
            df = self.index.doc_freq(term)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in self.index.postings[term]:
                dl = self.index.doc_lengths.get(doc_id, 0)
                norm = self.k1 * (1 - self.b + self.b * dl / avg_dl) if avg_dl else 1.0
                tf_score = (tf * (self.k1 + 1)) / (tf + norm)
                scores[doc_id] += idf * tf_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "postings": dict(self.index.postings),
            "doc_lengths": self.index.doc_lengths,
            "num_docs": self.index.num_docs,
            "avg_doc_length": self.index.avg_doc_length,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k1=data["k1"], b=data["b"])
        obj.index.postings = defaultdict(list, data["postings"])
        obj.index.doc_lengths = data["doc_lengths"]
        obj.index.num_docs = data["num_docs"]
        obj.index.avg_doc_length = data["avg_doc_length"]
        print("Building BM25 numpy cache...")
        obj._build_numpy_cache()
        # Warmup: force memory pages into cache so first real query isn't slow
        obj.query("warmup query neural network", top_k=10)
        print("BM25 numpy cache ready.")
        return obj


async def _build_from_db(output_path: Path) -> None:
    from sqlmodel import Session, create_engine, select

    from scholar.config import settings
    from scholar.db.models import Paper

    engine = create_engine(settings.database_url_sync, echo=False)
    paper_ids: list[str] = []
    docs: list[str] = []
    offset = 0
    batch_size = 1000

    print("Loading papers from DB ...")
    while True:
        with Session(engine) as session:
            rows = session.exec(
                select(Paper.id, Paper.title, Paper.abstract)
                .offset(offset)
                .limit(batch_size)
            ).all()
        if not rows:
            break
        for pid, title, abstract in rows:
            paper_ids.append(pid)
            docs.append(f"{title} {abstract}")
        offset += batch_size
        if len(rows) < batch_size:
            break

    print(f"Fitting BM25 on {len(paper_ids)} documents ...")
    bm25 = BM25()
    bm25.fit(docs, paper_ids)
    bm25.save(output_path)
    print(f"Saved BM25 index to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index from DB")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/bm25.pkl"),
        help="Path to save BM25 index",
    )
    args = parser.parse_args()
    asyncio.run(_build_from_db(args.output))
