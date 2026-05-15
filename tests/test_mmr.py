"""Tests for MMR reranking and adaptive_lambda."""
from __future__ import annotations

import numpy as np
import pytest

from scholar.recsys.mmr import adaptive_lambda, mmr_rerank


def _make_embs(n: int, dim: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {f"p{i}": rng.randn(dim).astype(np.float32) for i in range(n)}


class TestMMRRerank:
    def test_returns_top_k(self):
        embs = _make_embs(10)
        candidates = [(pid, float(10 - i)) for i, pid in enumerate(embs)]
        result = mmr_rerank(candidates, embs, lam=0.5, top_k=5)
        assert len(result) == 5

    def test_returns_fewer_when_not_enough(self):
        embs = _make_embs(3)
        candidates = [(pid, 1.0) for pid in embs]
        result = mmr_rerank(candidates, embs, lam=0.5, top_k=10)
        assert len(result) <= 3

    def test_empty_candidates(self):
        result = mmr_rerank([], {}, lam=0.5, top_k=5)
        assert result == []

    def test_lam1_returns_most_relevant_first(self):
        embs = {
            "best": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "second": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
            "third": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        }
        candidates = [("best", 3.0), ("second", 2.0), ("third", 1.0)]
        result = mmr_rerank(candidates, embs, lam=1.0, top_k=3)
        assert result[0][0] == "best"

    def test_lam0_maximises_diversity(self):
        # Two papers nearly identical, one very different — with lam=0 the diverse one should appear
        embs = {
            "clone_a": np.array([1.0, 0.0], dtype=np.float32),
            "clone_b": np.array([0.99, 0.01], dtype=np.float32),
            "diverse": np.array([0.0, 1.0], dtype=np.float32),
        }
        candidates = [("clone_a", 3.0), ("clone_b", 2.5), ("diverse", 1.0)]
        result = mmr_rerank(candidates, embs, lam=0.0, top_k=2)
        result_ids = [pid for pid, _ in result]
        assert "diverse" in result_ids

    def test_no_duplicates(self):
        embs = _make_embs(8)
        candidates = [(pid, float(8 - i)) for i, pid in enumerate(embs)]
        result = mmr_rerank(candidates, embs, lam=0.5, top_k=6)
        ids = [pid for pid, _ in result]
        assert len(ids) == len(set(ids))

    def test_scores_are_finite(self):
        embs = _make_embs(5)
        candidates = [(pid, float(5 - i)) for i, pid in enumerate(embs)]
        result = mmr_rerank(candidates, embs, lam=0.5, top_k=5)
        for _, score in result:
            assert np.isfinite(score)


class TestAdaptiveLambda:
    def test_cold_start_returns_low_lambda(self):
        embs = [np.array([1.0, 0.0], dtype=np.float32)]
        lam = adaptive_lambda(embs)
        assert lam == pytest.approx(0.3)

    def test_tight_cluster_returns_low_lambda(self):
        # Very similar vectors → high avg_sim > 0.7 → lambda = 0.3
        embs = [
            np.array([1.0, 0.01 * i], dtype=np.float32) for i in range(5)
        ]
        lam = adaptive_lambda(embs)
        assert lam <= 0.4

    def test_diverse_history_returns_high_lambda(self):
        # Orthogonal vectors → avg_sim ≈ 0 < 0.4 → lambda = 0.7
        embs = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        lam = adaptive_lambda(embs)
        assert lam >= 0.6

    def test_lambda_in_range(self):
        rng = np.random.RandomState(7)
        for _ in range(20):
            n = rng.randint(3, 10)
            embs = [rng.randn(4).astype(np.float32) for _ in range(n)]
            lam = adaptive_lambda(embs)
            assert 0.0 <= lam <= 1.0
