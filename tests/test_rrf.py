"""Focused tests for Reciprocal Rank Fusion algorithm."""
from __future__ import annotations

import pytest

from scholar.retrieval.hybrid import _reciprocal_rank_fusion


class TestRRF:
    def test_k60_score_formula(self):
        list1 = [("docA", 1.0)]
        fused = _reciprocal_rank_fusion([list1], k=60)
        score = dict(fused)["docA"]
        assert score == pytest.approx(1.0 / (60 + 1))

    def test_doc_in_both_lists_accumulates(self):
        list1 = [("docA", 1.0), ("docB", 0.5)]
        list2 = [("docA", 1.0), ("docC", 0.5)]
        fused = dict(_reciprocal_rank_fusion([list1, list2], k=60))
        # docA appears in both at rank 1 → 1/61 + 1/61
        assert fused["docA"] == pytest.approx(2.0 / 61.0)

    def test_unique_doc_only_in_one_list(self):
        list1 = [("docA", 1.0), ("docX", 0.5)]
        list2 = [("docA", 1.0), ("docY", 0.5)]
        fused = dict(_reciprocal_rank_fusion([list1, list2], k=60))
        # docX and docY each appear in only one list
        assert fused["docX"] == pytest.approx(1.0 / 62.0)
        assert fused["docY"] == pytest.approx(1.0 / 62.0)

    def test_output_sorted_descending(self):
        list1 = [("docA", 1.0), ("docB", 0.8), ("docC", 0.5)]
        list2 = [("docB", 1.0), ("docA", 0.7), ("docD", 0.4)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicates_in_output(self):
        list1 = [("docA", 1.0), ("docB", 0.8)]
        list2 = [("docA", 1.0), ("docB", 0.9)]
        fused = _reciprocal_rank_fusion([list1, list2], k=60)
        ids = [d for d, _ in fused]
        assert len(ids) == len(set(ids))

    def test_empty_input(self):
        assert _reciprocal_rank_fusion([], k=60) == []
        assert _reciprocal_rank_fusion([[], []], k=60) == []

    def test_single_doc_single_list(self):
        fused = _reciprocal_rank_fusion([[("docZ", 0.5)]], k=60)
        assert len(fused) == 1
        assert fused[0][0] == "docZ"

    def test_k_value_affects_score(self):
        list1 = [("docA", 1.0)]
        fused_k60 = dict(_reciprocal_rank_fusion([list1], k=60))
        fused_k10 = dict(_reciprocal_rank_fusion([list1], k=10))
        assert fused_k10["docA"] > fused_k60["docA"]

    def test_rank_position_affects_score(self):
        list1 = [("first", 1.0), ("second", 0.5)]
        fused = dict(_reciprocal_rank_fusion([list1], k=60))
        assert fused["first"] > fused["second"]

    def test_merges_three_lists(self):
        list1 = [("docA", 1.0)]
        list2 = [("docB", 1.0)]
        list3 = [("docA", 1.0)]
        fused = dict(_reciprocal_rank_fusion([list1, list2, list3], k=60))
        # docA in list1 (rank1) + list3 (rank1) = 2/61
        # docB in list2 (rank1) = 1/61
        assert fused["docA"] > fused["docB"]
