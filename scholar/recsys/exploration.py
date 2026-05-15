"""Epsilon-greedy exploration to prevent filter bubbles."""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholar.db.models import Paper


def epsilon_greedy_inject(
    ranked: list[tuple[str, float]],
    all_paper_ids: list[str],
    history_ids: set[str],
    epsilon: float = 0.1,
    quality_percentile: float = 0.8,
    paper_meta: dict[str, "Paper"] | None = None,
) -> list[tuple[str, float]]:
    """Replace ~epsilon fraction of ranked results with random high-citation papers.

    Replaces at most 1–2 tail slots so results don't feel random.
    """
    if not ranked or not all_paper_ids or epsilon <= 0:
        return ranked

    paper_meta = paper_meta or {}
    n_inject = max(1, math.ceil(epsilon * len(ranked)))
    n_inject = min(n_inject, 2)

    ranked_ids = {pid for pid, _ in ranked}
    excluded = history_ids | ranked_ids

    candidates = [
        pid for pid in all_paper_ids
        if pid not in excluded and pid in paper_meta
    ]

    if not candidates:
        return ranked

    candidates.sort(
        key=lambda pid: paper_meta[pid].citation_count,
        reverse=True,
    )
    cutoff = max(1, int(len(candidates) * quality_percentile))
    high_quality = candidates[:cutoff]

    n_sample = min(n_inject, len(high_quality))
    injected = random.sample(high_quality, n_sample)

    min_score = min(score for _, score in ranked) if ranked else 0.0
    result = list(ranked[:-n_inject]) if n_inject < len(ranked) else list(ranked)
    for pid in injected:
        result.append((pid, min_score * 0.9))

    return result
