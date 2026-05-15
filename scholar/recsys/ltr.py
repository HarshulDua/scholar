"""LightGBM LambdaRank for learning-to-rank paper recommendations."""
from __future__ import annotations

from datetime import date as date_cls
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from scholar.db.models import Paper

MODEL_PATH = Path("./data/ltr_model.lgb")

FEATURE_NAMES = [
    "faiss_score",
    "bm25_score",
    "citation_count",
    "days_since_pub",
    "category_overlap",
    "user_paper_cosine",
    "is_in_history",
]


def _days_since(update_date: Optional[date_cls]) -> float:
    if update_date is None:
        return 3650.0
    delta = date_cls.today() - update_date
    return float(max(delta.days, 0))


def _category_overlap(paper_cats: str, history_cats: list[str]) -> float:
    if not history_cats:
        return 0.0
    paper_set = set(paper_cats.split())
    history_set: set[str] = set()
    for cats in history_cats:
        history_set.update(cats.split())
    union = paper_set | history_set
    if not union:
        return 0.0
    return len(paper_set & history_set) / len(union)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_features(
    user_emb: np.ndarray,
    candidate_ids: list[str],
    candidate_scores: dict[str, float],
    paper_meta: dict[str, "Paper"],
    paper_embeddings: dict[str, np.ndarray],
    history_cats: list[str],
    history_ids: set[str] | None = None,
) -> np.ndarray:
    """Build (N, 7) feature matrix for a set of candidates."""
    history_ids = history_ids or set()
    rows: list[list[float]] = []

    for pid in candidate_ids:
        paper = paper_meta.get(pid)
        faiss_score = float(candidate_scores.get(pid, 0.0))
        citation_count = float(paper.citation_count) if paper else 0.0
        days_since = _days_since(paper.update_date) if paper else 3650.0
        cat_overlap = _category_overlap(paper.categories if paper else "", history_cats)

        paper_emb = paper_embeddings.get(pid)
        user_paper_cos = (
            _cosine(user_emb, paper_emb)
            if paper_emb is not None and user_emb is not None
            else 0.0
        )

        rows.append([
            faiss_score,
            0.0,                         # bm25_score — filled by caller
            citation_count,
            days_since,
            cat_overlap,
            user_paper_cos,
            1.0 if pid in history_ids else 0.0,
        ])

    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), dtype=np.float32)


class LTRModel:
    def __init__(self) -> None:
        self._model = None

    def load(self, path: Path = MODEL_PATH) -> bool:
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(path))  # type: ignore[assignment]
            return True
        except Exception:
            return False

    def score(
        self,
        user_emb: np.ndarray,
        candidates: list[tuple[str, float]],
        paper_meta: dict[str, "Paper"],
        paper_embeddings: dict[str, np.ndarray],
        history_cats: list[str],
    ) -> list[tuple[str, float]]:
        if self._model is None or not candidates:
            return candidates

        candidate_ids = [pid for pid, _ in candidates]
        candidate_scores = {pid: score for pid, score in candidates}

        X = extract_features(  # noqa: N806
            user_emb=user_emb,
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            paper_meta=paper_meta,
            paper_embeddings=paper_embeddings,
            history_cats=history_cats,
        )
        if X.shape[0] == 0:
            return candidates

        preds: np.ndarray = self._model.predict(X)
        return sorted(
            zip(candidate_ids, preds.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

    def train(
        self,
        interaction_data: list[tuple[str, str]],
        faiss_retriever,
        bm25,
        paper_meta: dict[str, "Paper"],
        save_path: Path = MODEL_PATH,
    ) -> None:
        import lightgbm as lgb

        # Build paper embeddings from FAISS
        paper_embeddings: dict[str, np.ndarray] = {}
        if faiss_retriever is not None and faiss_retriever._index is not None:
            id_to_faiss = {pid: i for i, pid in enumerate(faiss_retriever._paper_ids)}
            for pid in paper_meta:
                idx = id_to_faiss.get(pid)
                if idx is not None:
                    vec = faiss_retriever._index.reconstruct(idx)
                    paper_embeddings[pid] = np.array(vec, dtype=np.float32)

        # Group by user
        user_papers: dict[str, set[str]] = {}
        for uid, pid in interaction_data:
            user_papers.setdefault(uid, set()).add(pid)

        all_paper_ids = list(paper_meta.keys())

        X_all: list[np.ndarray] = []  # noqa: N806
        y_all: list[int] = []
        groups: list[int] = []

        rng = np.random.RandomState(42)

        for uid, pos_papers in user_papers.items():
            pos_papers_list = [p for p in pos_papers if p in paper_meta]
            if not pos_papers_list:
                continue

            history_cats = [paper_meta[p].categories for p in pos_papers_list]

            pos_embs = [paper_embeddings[p] for p in pos_papers_list if p in paper_embeddings]
            user_emb = (
                np.mean(pos_embs, axis=0).astype(np.float32) if pos_embs
                else np.zeros(384, dtype=np.float32)
            )

            neg_pool = [p for p in all_paper_ids if p not in pos_papers]
            n_neg = min(len(pos_papers_list) * 4, len(neg_pool), 20)
            if n_neg == 0:
                continue
            neg_papers = list(rng.choice(neg_pool, n_neg, replace=False))

            candidate_ids = pos_papers_list + neg_papers
            labels = [3] * len(pos_papers_list) + [0] * len(neg_papers)
            candidate_scores = {
                **{p: 1.0 for p in pos_papers_list},
                **{p: 0.0 for p in neg_papers},
            }

            X = extract_features(  # noqa: N806
                user_emb=user_emb,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                paper_meta=paper_meta,
                paper_embeddings=paper_embeddings,
                history_cats=history_cats,
            )

            X_all.append(X)
            y_all.extend(labels)
            groups.append(len(candidate_ids))

        if not X_all:
            print("No training data for LTR — skipping.")
            return

        X_train = np.vstack(X_all)  # noqa: N806
        y_train = np.array(y_all, dtype=np.int32)

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups,
            feature_name=FEATURE_NAMES,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 1,
            "verbose": -1,
        }

        print(f"Training LTR on {len(X_all)} users, {X_train.shape[0]} (query, doc) pairs...")
        model = lgb.train(params, train_data, num_boost_round=100)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save_path))
        self._model = model  # type: ignore[assignment]
        print(f"LTR model saved to {save_path}")
