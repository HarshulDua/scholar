"""Recsys evaluation: NDCG@10, HR@10, ILD on CiteULike held-out test users.

Run:
    python -m scholar.recsys.eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _ild(paper_ids: list[str], paper_embeddings: dict[str, np.ndarray]) -> float:
    """Intra-List Diversity: mean pairwise cosine distance."""
    embs = [paper_embeddings[pid] for pid in paper_ids if pid in paper_embeddings]
    if len(embs) < 2:
        return 0.0
    dists = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dists.append(_cosine_dist(embs[i], embs[j]))
    return float(np.mean(dists))


def _ndcg_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, pid in enumerate(recommended[:k])
        if pid in relevant
    )
    ideal = [1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))]
    idcg = sum(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _hr_at_k(recommended: list[str], relevant: set[str], k: int = 10) -> float:
    return float(any(pid in relevant for pid in recommended[:k]))


def _load_all_interactions() -> dict[str, set[str]]:
    from sqlmodel import Session, create_engine, select

    from scholar.config import settings
    from scholar.db.models import UserHistory

    engine = create_engine(settings.database_url_sync, echo=False)
    with Session(engine) as session:
        rows = session.exec(select(UserHistory.user_id, UserHistory.paper_id)).all()

    user_papers: dict[str, set[str]] = {}
    for uid, pid in rows:
        user_papers.setdefault(str(uid), set()).add(str(pid))
    return user_papers


def _get_paper_embeddings(faiss_retriever) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    if faiss_retriever is None or faiss_retriever._index is None:
        return embeddings
    for i, pid in enumerate(faiss_retriever._paper_ids):
        vec = faiss_retriever._index.reconstruct(i)  # type: ignore[attr-defined]
        embeddings[pid] = np.array(vec, dtype=np.float32)
    return embeddings


def evaluate(ablation: str = "mmr") -> dict:
    """
    ablation: 'mmr' | 'ltr+mmr' | 'ltr+mmr+exploration'
    """
    from scholar.recsys.mmr import adaptive_lambda, mmr_rerank

    print(f"Loading data for ablation: {ablation}")
    user_papers = _load_all_interactions()
    if not user_papers:
        print("No interactions found. Load CiteULike data first.")
        return {}

    # 80/10/10 split by user — take test 10%
    all_users = sorted(user_papers.keys())
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_users))
    n_train = int(0.8 * len(all_users))
    n_val = int(0.1 * len(all_users))
    test_users = [all_users[i] for i in perm[n_train + n_val:]]
    print(f"Test users: {len(test_users)}")

    faiss_path = Path("./data/faiss.index")
    if not faiss_path.exists():
        print("FAISS index not found. Build first.")
        return {}

    from scholar.retrieval.dense import DenseRetriever

    dense = DenseRetriever()
    dense.load()
    print("FAISS loaded.")

    paper_embeddings = _get_paper_embeddings(dense)
    id_to_idx = {pid: i for i, pid in enumerate(dense._paper_ids)}

    ltr_model = None
    if "ltr" in ablation:
        ltr_path = Path("./data/ltr_model.lgb")
        if ltr_path.exists():
            from scholar.recsys.ltr import LTRModel

            ltr_model = LTRModel()
            ltr_model.load(ltr_path)
            print("LTR model loaded.")

    ndcgs, hrs, ilds = [], [], []

    for uid in test_users:
        all_papers = user_papers[uid]
        # 80% train / 20% test split per user
        papers_list = sorted(all_papers)
        n_user_train = max(1, int(0.8 * len(papers_list)))
        train_papers = set(papers_list[:n_user_train])
        test_papers = set(papers_list[n_user_train:])

        if not test_papers:
            continue

        # Build user embedding from train papers
        history_embs = []
        for pid in train_papers:
            idx = id_to_idx.get(pid)
            if idx is not None:
                vec = dense._index.reconstruct(idx)  # type: ignore[attr-defined]
                history_embs.append(np.array(vec, dtype=np.float32))

        if not history_embs:
            continue

        user_emb = np.mean(history_embs, axis=0).astype(np.float32)
        candidates = dense.query_by_vector(user_emb, top_k=200)
        candidates = [(pid, s) for pid, s in candidates if pid not in train_papers]

        if ltr_model is not None and ltr_model._model is not None:
            candidates = ltr_model.score(
                user_emb=user_emb,
                candidates=candidates,
                paper_meta={},
                paper_embeddings=paper_embeddings,
                history_cats=[],
            )[:50]
        else:
            candidates = candidates[:50]

        lam = adaptive_lambda(history_embs)
        reranked = mmr_rerank(candidates, paper_embeddings, lam=lam, top_k=10)

        if "exploration" in ablation:
            from scholar.recsys.exploration import epsilon_greedy_inject

            reranked = epsilon_greedy_inject(
                ranked=reranked,
                all_paper_ids=dense._paper_ids,
                history_ids=train_papers,
                epsilon=0.1,
            )

        recommended_ids = [pid for pid, _ in reranked]
        ndcgs.append(_ndcg_at_k(recommended_ids, test_papers))
        hrs.append(_hr_at_k(recommended_ids, test_papers))
        ilds.append(_ild(recommended_ids, paper_embeddings))

    result = {
        "ablation": ablation,
        "n_users": len(ndcgs),
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hr_at_10": float(np.mean(hrs)) if hrs else 0.0,
        "ild": float(np.mean(ilds)) if ilds else 0.0,
    }

    print(
        f"[{ablation}] NDCG@10={result['ndcg_at_10']:.4f} | "
        f"HR@10={result['hr_at_10']:.4f} | ILD={result['ild']:.4f} "
        f"({result['n_users']} users)"
    )
    return result


def _update_benchmarks(all_results: list[dict]) -> None:
    benchmarks_path = Path("./docs/benchmarks.md")
    content = benchmarks_path.read_text(encoding="utf-8")

    table = (
        "## Layer 2: Reranking\n\n"
        "Evaluation dataset: CiteULike-A test split (10% of users).\n\n"
        "| Ablation                | NDCG@10 | HR@10  | ILD    | Users |\n"
        "|-------------------------|---------|--------|--------|-------|\n"
    )
    for r in all_results:
        table += (
            f"| {r['ablation']:<23} | {r['ndcg_at_10']:.4f}  | {r['hr_at_10']:.4f} "
            f"| {r['ild']:.4f} | {r['n_users']:>5} |\n"
        )

    import re

    if "## Layer 2: Reranking" in content:
        content = re.sub(
            r"## Layer 2: Reranking.*?(?=\n---|\Z)", table, content, flags=re.DOTALL
        )
    else:
        content += "\n---\n\n" + table

    benchmarks_path.write_text(content, encoding="utf-8")
    print("Benchmarks updated.")


def main() -> None:
    ablations = ["mmr", "ltr+mmr", "ltr+mmr+exploration"]
    all_results = []
    for ablation in ablations:
        result = evaluate(ablation)
        if result:
            all_results.append(result)

    if all_results:
        _update_benchmarks(all_results)


if __name__ == "__main__":
    main()
