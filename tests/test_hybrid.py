"""Tests for the HybridRetriever and RRF fusion logic."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scholar.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion


class TestRRFFusion:
    def test_rrf_basic(self):
        list1 = [("docA", 1.0), ("docB", 0.8), ("docC", 0.6)]
        list2 = [("docB", 1.0), ("docA", 0.9), ("docD", 0.7)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)

        ids = [d for d, _ in fused]
        assert "docA" in ids
        assert "docB" in ids
        assert "docC" in ids
        assert "docD" in ids

    def test_rrf_unique_docs(self):
        list1 = [("docA", 1.0), ("docB", 0.8), ("docC", 0.6)]
        list2 = [("docB", 1.0), ("docA", 0.9), ("docD", 0.7)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)

        ids = [d for d, _ in fused]
        assert len(ids) == len(set(ids)), "Fused results contain duplicates"

    def test_rrf_fusion_cross_rank(self):
        list1 = [("docA", 1.0), ("docB", 0.8)]
        list2 = [("docB", 1.0), ("docA", 0.9)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)

        score_A = next(s for d, s in fused if d == "docA")
        score_B = next(s for d, s in fused if d == "docB")

        expected_A = 1.0 / (60 + 1) + 1.0 / (60 + 2)
        expected_B = 1.0 / (60 + 2) + 1.0 / (60 + 1)
        assert abs(score_A - expected_A) < 1e-9
        assert abs(score_B - expected_B) < 1e-9
        assert abs(score_A - score_B) < 1e-9

    def test_rrf_doc_only_in_one_list(self):
        list1 = [("docA", 1.0), ("docX", 0.5)]
        list2 = [("docA", 1.0), ("docY", 0.5)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)

        score_X = next((s for d, s in fused if d == "docX"), None)
        score_Y = next((s for d, s in fused if d == "docY"), None)
        score_A = next((s for d, s in fused if d == "docA"), None)

        assert score_X is not None and score_Y is not None
        assert score_A > score_X
        assert score_A > score_Y

    def test_rrf_empty_lists(self):
        fused = _reciprocal_rank_fusion([[], []], k=60)
        assert fused == []

    def test_rrf_single_list(self):
        list1 = [("docA", 1.0), ("docB", 0.5)]
        fused = _reciprocal_rank_fusion([list1], k=60)
        assert len(fused) == 2


class TestHybridRetriever:
    def _make_retriever(self, bm25_results, dense_results):
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = bm25_results

        mock_dense = MagicMock()
        mock_dense.query.return_value = dense_results

        return HybridRetriever(bm25=mock_bm25, dense=mock_dense)

    def test_rrf_fusion_both_docs_appear(self):
        bm25_results = [("docA", 1.0), ("docB", 0.8), ("docC", 0.6)]
        dense_results = [("docB", 1.0), ("docA", 0.9), ("docD", 0.5)]

        retriever = self._make_retriever(bm25_results, dense_results)
        results = retriever.query("test query", top_k=10)

        result_ids = [pid for pid, _ in results]
        assert "docA" in result_ids
        assert "docB" in result_ids

    def test_rrf_cross_rank_both_score(self):
        bm25_results = [("docA", 1.0), ("docB", 0.8)]
        dense_results = [("docB", 1.0), ("docA", 0.7)]

        retriever = self._make_retriever(bm25_results, dense_results)
        results = retriever.query("q", top_k=5)

        result_ids = [pid for pid, _ in results]
        assert "docA" in result_ids
        assert "docB" in result_ids

    def test_rrf_unique_docs_no_duplicates(self):
        bm25_results = [("docA", 1.0), ("docB", 0.9), ("docC", 0.8)]
        dense_results = [("docA", 1.0), ("docB", 0.85), ("docE", 0.7)]

        retriever = self._make_retriever(bm25_results, dense_results)
        results = retriever.query("test", top_k=10)

        result_ids = [pid for pid, _ in results]
        assert len(result_ids) == len(set(result_ids)), "Duplicates in hybrid results"

    def test_calls_both_retrievers(self):
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = [("d1", 0.9)]
        mock_dense = MagicMock()
        mock_dense.query.return_value = [("d2", 0.8)]

        retriever = HybridRetriever(bm25=mock_bm25, dense=mock_dense)
        retriever.query("q", top_k=5)

        mock_bm25.query.assert_called_once()
        mock_dense.query.assert_called_once()

    def test_top_k_respected(self):
        bm25_results = [(f"doc{i}", 1.0 - i * 0.05) for i in range(20)]
        dense_results = [(f"doc{i}", 1.0 - i * 0.04) for i in range(20)]

        retriever = self._make_retriever(bm25_results, dense_results)
        results = retriever.query("q", top_k=5)

        assert len(results) <= 5
