"""Hybrid retrieval via Reciprocal Rank Fusion of BM25 and dense results."""
from __future__ import annotations

from scholar.retrieval.bm25 import BM25
from scholar.retrieval.dense import DenseRetriever


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, bm25: BM25, dense: DenseRetriever) -> None:
        self.bm25 = bm25
        self.dense = dense

    def query(self, q: str, top_k: int = 200) -> list[tuple[str, float]]:
        bm25_results = self.bm25.query(q, top_k=top_k)
        dense_results = self.dense.query(q, top_k=top_k)
        fused = _reciprocal_rank_fusion([bm25_results, dense_results], k=60)
        return fused[:top_k]
