"""Maximum Marginal Relevance reranking."""
from __future__ import annotations

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_rerank(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, np.ndarray],
    lam: float = 0.5,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Standard MMR reranking.

    candidates: (paper_id, relevance_score) list sorted by relevance descending.
    embeddings: paper_id -> embedding array.
    lam: trade-off between relevance and diversity (higher = more relevance).
    Returns top_k (paper_id, mmr_score) pairs.
    """
    if not candidates:
        return []

    candidates_with_emb = [
        (pid, rel) for pid, rel in candidates if pid in embeddings
    ]

    if not candidates_with_emb:
        return candidates[:top_k]

    selected: list[tuple[str, float]] = []
    remaining = list(candidates_with_emb)

    while remaining and len(selected) < top_k:
        best_pid: str | None = None
        best_score = float("-inf")

        for pid, rel in remaining:
            emb = embeddings[pid]
            if not selected:
                diversity_penalty = 0.0
            else:
                sims = [_cosine_sim(emb, embeddings[s_pid]) for s_pid, _ in selected]
                diversity_penalty = max(sims)

            mmr_score = lam * rel - (1 - lam) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_pid = pid

        if best_pid is None:
            break

        selected.append((best_pid, best_score))
        remaining = [(pid, rel) for pid, rel in remaining if pid != best_pid]

    return selected


def adaptive_lambda(user_history_embeddings: list[np.ndarray]) -> float:
    """Compute adaptive lambda based on user history diversity.

    Cold start (< 3 items): 0.3 (prefer diversity).
    Tight cluster (avg_sim > 0.7): 0.3 (specialist — inject diversity).
    Generalist (avg_sim < 0.4): 0.7 (prefer relevance).
    Otherwise: linear interpolation.
    """
    if len(user_history_embeddings) < 3:
        return 0.3

    n = len(user_history_embeddings)
    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += _cosine_sim(user_history_embeddings[i], user_history_embeddings[j])
            count += 1

    avg_sim = total_sim / count if count > 0 else 0.0

    if avg_sim > 0.7:
        return 0.3
    if avg_sim < 0.4:
        return 0.7

    t = (avg_sim - 0.4) / (0.7 - 0.4)
    return 0.7 - t * (0.7 - 0.3)
