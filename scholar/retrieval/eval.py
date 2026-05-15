"""Retrieval evaluation on NFCorpus (BM25 vs Dense vs Hybrid).

Metrics: NDCG@10, MRR@10, MAP@10.
Writes results to docs/benchmarks.md.

Run:
    python -m scholar.retrieval.eval
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

BENCHMARKS_PATH = Path("./docs/benchmarks.md")


def _download_nfcorpus():
    try:
        import ir_datasets

        dataset = ir_datasets.load("nfcorpus/test")
        return dataset
    except Exception as e:
        print(f"ir_datasets not available or nfcorpus download failed: {e}")
        print("Install: pip install ir_datasets")
        return None


def _dcg_at_k(relevances: list[float], k: int) -> float:
    gains = relevances[:k]
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    rels = [float(relevant.get(pid, 0)) for pid in retrieved_ids[:k]]
    dcg = _dcg_at_k(rels, k)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = _dcg_at_k([float(v) for v in ideal], k)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if relevant.get(pid, 0) > 0:
            return 1.0 / rank
    return 0.0


def _map_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int) -> float:
    num_relevant = sum(1 for v in relevant.values() if v > 0)
    if num_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if relevant.get(pid, 0) > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(num_relevant, k)


def evaluate_retriever(name: str, retriever_fn, queries_with_qrels: list, k: int = 10):
    ndcgs, mrrs, maps, latencies = [], [], [], []

    for query_text, relevant in queries_with_qrels:
        t0 = time.perf_counter()
        try:
            results = retriever_fn(query_text, top_k=100)
        except Exception:
            continue
        lat = (time.perf_counter() - t0) * 1000

        retrieved_ids = [pid for pid, _ in results]
        ndcgs.append(_ndcg_at_k(retrieved_ids, relevant, k))
        mrrs.append(_mrr_at_k(retrieved_ids, relevant, k))
        maps.append(_map_at_k(retrieved_ids, relevant, k))
        latencies.append(lat)

    return {
        "name": name,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "map": float(np.mean(maps)) if maps else 0.0,
        "lat_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "lat_p99": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "n_queries": len(ndcgs),
    }


def _update_benchmarks(results: list[dict]) -> None:
    content = BENCHMARKS_PATH.read_text(encoding="utf-8")

    header = "## Layer 1: Retrieval (BM25 + Dense + Hybrid)\n\n"
    table = (
        "Evaluation dataset: NFCorpus/test. Metric: NDCG@10, MRR@10, MAP@10.\n\n"
        "| Retriever         | NDCG@10 | MRR@10 | MAP@10 | Latency p50 (ms) | Latency p99 (ms) | Queries |\n"
        "|-------------------|---------|--------|--------|------------------|------------------|---------|\n"
    )
    for r in results:
        table += (
            f"| {r['name']:<17} | {r['ndcg']:.4f}  | {r['mrr']:.4f} | {r['map']:.4f} "
            f"| {r['lat_p50']:>16.1f} | {r['lat_p99']:>16.1f} | {r['n_queries']:>7} |\n"
        )

    new_section = header + table
    if "## Layer 1: Retrieval" in content:
        import re
        content = re.sub(
            r"## Layer 1: Retrieval.*?(?=\n---|\Z)", new_section, content, flags=re.DOTALL
        )
    else:
        content = new_section + "\n---\n\n" + content

    BENCHMARKS_PATH.write_text(content, encoding="utf-8")
    print(f"Benchmarks updated at {BENCHMARKS_PATH}")


def main() -> None:
    from pathlib import Path

    print("Loading retrieval components...")
    bm25_path = Path("./data/bm25.pkl")
    faiss_path = Path("./data/faiss.index")

    bm25 = None
    if bm25_path.exists():
        from scholar.retrieval.bm25 import BM25

        bm25 = BM25.load(bm25_path)
        print("BM25 loaded.")

    dense = None
    if faiss_path.exists():
        from scholar.retrieval.dense import DenseRetriever

        dense = DenseRetriever()
        dense.load()
        print("Dense retriever loaded.")

    if bm25 is None and dense is None:
        print("No index found. Build indexes first with scripts/build_index.py")
        return

    print("Loading NFCorpus queries and qrels...")
    dataset = _download_nfcorpus()
    if dataset is None:
        print("Could not load NFCorpus. Skipping evaluation.")
        return

    # Build query -> relevant docs mapping
    qrels: dict[str, dict[str, int]] = {}
    try:
        for qrel in dataset.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    except Exception as e:
        print(f"Could not read qrels: {e}")
        return

    queries_with_qrels = []
    try:
        for query in dataset.queries_iter():
            if query.query_id in qrels:
                queries_with_qrels.append((query.text, qrels[query.query_id]))
    except Exception as e:
        print(f"Could not read queries: {e}")
        return

    print(f"Evaluating on {len(queries_with_qrels)} queries...")

    results = []

    if bm25 is not None:
        print("Evaluating BM25...")
        r = evaluate_retriever("BM25 (scratch)", bm25.query, queries_with_qrels)
        results.append(r)
        print(f"  BM25: NDCG@10={r['ndcg']:.4f}, MRR@10={r['mrr']:.4f}, MAP@10={r['map']:.4f}")

    if dense is not None:
        print("Evaluating Dense (MiniLM)...")
        r = evaluate_retriever("Dense (MiniLM)", dense.query, queries_with_qrels)
        results.append(r)
        print(f"  Dense: NDCG@10={r['ndcg']:.4f}, MRR@10={r['mrr']:.4f}, MAP@10={r['map']:.4f}")

    if bm25 is not None and dense is not None:
        from scholar.retrieval.hybrid import HybridRetriever

        hybrid = HybridRetriever(bm25=bm25, dense=dense)
        print("Evaluating Hybrid RRF...")
        r = evaluate_retriever("Hybrid (RRF k=60)", hybrid.query, queries_with_qrels)
        results.append(r)
        print(f"  Hybrid: NDCG@10={r['ndcg']:.4f}, MRR@10={r['mrr']:.4f}, MAP@10={r['map']:.4f}")

    if results:
        _update_benchmarks(results)


if __name__ == "__main__":
    main()
