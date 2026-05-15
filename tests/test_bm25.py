"""Tests for the custom BM25 implementation."""
from __future__ import annotations

import os
import tempfile

import pytest

from scholar.retrieval.bm25 import BM25, InvertedIndex, _tokenize


SAMPLE_DOCS = [
    "machine learning neural networks deep learning",
    "natural language processing text classification",
    "computer vision image recognition object detection",
    "reinforcement learning reward policy optimization",
    "transformer attention mechanism self-attention",
]
SAMPLE_IDS = ["doc0", "doc1", "doc2", "doc3", "doc4"]


def _make_fitted_bm25() -> BM25:
    bm25 = BM25()
    bm25.fit(SAMPLE_DOCS, SAMPLE_IDS)
    return bm25


class TestTokenize:
    def test_lowercase(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_splits_on_non_alphanumeric(self):
        tokens = _tokenize("foo-bar_baz.qux")
        for t in tokens:
            assert t.isalnum() or t == ""


class TestInvertedIndex:
    def test_build_populates_postings(self):
        idx = InvertedIndex()
        idx.build(["hello world", "world peace"], ["d0", "d1"])
        assert "world" in idx.postings
        assert len(idx.postings["world"]) == 2

    def test_doc_freq(self):
        idx = InvertedIndex()
        idx.build(["foo bar", "bar baz", "qux"], ["d0", "d1", "d2"])
        assert idx.doc_freq("bar") == 2
        assert idx.doc_freq("foo") == 1
        assert idx.doc_freq("notexist") == 0

    def test_avg_doc_length(self):
        idx = InvertedIndex()
        idx.build(["a b c", "d e"], ["d0", "d1"])
        assert idx.avg_doc_length == pytest.approx(2.5)


class TestBM25:
    def test_basic_retrieval(self):
        bm25 = _make_fitted_bm25()
        results = bm25.query("deep learning neural", top_k=5)
        assert len(results) > 0
        top_id, top_score = results[0]
        assert top_id == "doc0", f"Expected doc0 at rank-1, got {top_id}"
        assert top_score > 0

    def test_idf_weights_rare_terms(self):
        docs = [
            "common word here and common word",
            "common word and common",
            "common word common word common",
            "rare_unique_term only appears here once",
            "common word once more common",
        ]
        ids = ["d0", "d1", "d2", "d3", "d4"]
        bm25 = BM25()
        bm25.fit(docs, ids)

        results_rare = bm25.query("rare_unique_term", top_k=5)
        results_common = bm25.query("common", top_k=5)

        assert len(results_rare) > 0
        assert results_rare[0][0] == "d3"

        top_rare_score = results_rare[0][1]
        top_common_score = results_common[0][1]
        assert top_rare_score > 0

    def test_empty_query(self):
        bm25 = _make_fitted_bm25()
        results = bm25.query("", top_k=10)
        assert results == []

    def test_query_unknown_term(self):
        bm25 = _make_fitted_bm25()
        results = bm25.query("xyzzy_nonexistent_term_abc", top_k=10)
        assert results == []

    def test_returns_at_most_top_k(self):
        bm25 = _make_fitted_bm25()
        results = bm25.query("learning", top_k=2)
        assert len(results) <= 2

    def test_scores_sorted_descending(self):
        bm25 = _make_fitted_bm25()
        results = bm25.query("learning", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_load(self):
        bm25 = _make_fitted_bm25()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_bm25.pkl")
            bm25.save(save_path)

            loaded = BM25.load(save_path)

        original_results = bm25.query("neural networks", top_k=5)
        loaded_results = loaded.query("neural networks", top_k=5)

        assert len(original_results) == len(loaded_results)
        for (oid, oscore), (lid, lscore) in zip(original_results, loaded_results):
            assert oid == lid
            assert abs(oscore - lscore) < 1e-9

    def test_save_load_same_top_k(self):
        bm25 = _make_fitted_bm25()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "bm25.pkl")
            bm25.save(save_path)
            loaded = BM25.load(save_path)

        for q in ["deep", "transformer attention", "image recognition"]:
            r1 = bm25.query(q, top_k=3)
            r2 = loaded.query(q, top_k=3)
            assert [x[0] for x in r1] == [x[0] for x in r2]
