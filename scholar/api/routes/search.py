"""Search endpoint: hybrid retrieval + optional MMR reranking."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from scholar.api.schemas import PaperResult, SearchRequest, SearchResponse
from scholar.db.models import ClickLog, Paper, UserHistory
from scholar.recsys.mmr import adaptive_lambda, mmr_rerank

router = APIRouter()


async def _log_search_clicks(
    session_factory,
    user_id: Optional[str],
    query: str,
    results: list[PaperResult],
) -> None:
    now = datetime.now()
    async with session_factory() as session:
        for rank, paper in enumerate(results, start=1):
            session.add(
                ClickLog(
                    user_id=user_id,
                    paper_id=paper.id,
                    query=query,
                    rank_position=rank,
                    clicked_at=now,
                )
            )
        await session.commit()


async def _get_papers_by_ids(
    session: AsyncSession, paper_ids: list[str]
) -> dict[str, Paper]:
    if not paper_ids:
        return {}
    # Exclude embedding column (384-dim pgvector) — ~1.5KB per row, not needed here
    result = await session.execute(
        select(  # type: ignore[call-overload]  # SQLModel attrs typed as str, not Column
            Paper.id,
            Paper.title,
            Paper.abstract,
            Paper.authors,
            Paper.categories,
            Paper.citation_count,
        ).where(Paper.id.in_(paper_ids))  # type: ignore[attr-defined]
    )
    rows = result.all()
    papers: dict[str, Paper] = {}
    for row in rows:
        p = Paper(
            id=row.id,
            title=row.title,
            abstract=row.abstract,
            authors=row.authors,
            categories=row.categories,
            citation_count=row.citation_count,
        )
        papers[row.id] = p
    return papers


async def _get_user_history(
    session: AsyncSession, user_id: str
) -> list[str]:
    result = await session.execute(
        select(UserHistory.paper_id).where(UserHistory.user_id == user_id)
    )
    return list(result.scalars().all())


def _get_embeddings_for_ids(
    paper_ids: list[str], dense_retriever, id_to_idx: dict[str, int]
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    if dense_retriever is None:
        return embeddings

    # Prefer numpy embeddings array (avoids FAISS reconstruct OMP issues).
    emb_array = getattr(dense_retriever, "_embeddings", None)
    if emb_array is not None:
        for pid in paper_ids:
            idx = id_to_idx.get(pid)
            if idx is not None:
                embeddings[pid] = emb_array[idx]
        return embeddings

    # Fallback: FAISS reconstruct (only used if embeddings.npy not loaded).
    if dense_retriever._index is not None:
        for pid in paper_ids:
            idx = id_to_idx.get(pid)
            if idx is not None:
                vec = dense_retriever._index.reconstruct(idx)
                embeddings[pid] = np.array(vec, dtype=np.float32)

    return embeddings


@router.post("/search", response_model=SearchResponse)
async def search(
    request_body: SearchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> SearchResponse:
    hybrid_retriever = getattr(request.app.state, "hybrid_retriever", None)
    dense_retriever = getattr(request.app.state, "dense_retriever", None)
    async_session_factory = request.app.state.async_session_factory

    if hybrid_retriever is None:
        raise HTTPException(status_code=503, detail="Search index not loaded.")

    candidates = hybrid_retriever.query(request_body.query, top_k=200)
    total_candidates = len(candidates)

    candidate_ids = [pid for pid, _ in candidates]


    async with async_session_factory() as session:
        papers_map = await _get_papers_by_ids(session, candidate_ids)

        history_ids: list[str] = []
        if request_body.user_id:
            history_ids = await _get_user_history(session, request_body.user_id)

    id_to_idx: dict[str, int] = getattr(request.app.state, "id_to_idx", {})

    if request_body.user_id and history_ids and dense_retriever is not None:
        lam = request_body.diversity_lambda
        if lam == 0.5:
            emb_array = getattr(dense_retriever, "_embeddings", None)
            history_embeddings = []
            for hid in history_ids:
                idx = id_to_idx.get(hid)
                if idx is not None:
                    if emb_array is not None:
                        history_embeddings.append(emb_array[idx])
                    elif dense_retriever._index is not None:
                        vec = dense_retriever._index.reconstruct(idx)
                        history_embeddings.append(np.array(vec, dtype=np.float32))
            if history_embeddings:
                lam = adaptive_lambda(history_embeddings)

        paper_embeddings = _get_embeddings_for_ids(candidate_ids, dense_retriever, id_to_idx)
        reranked = mmr_rerank(
            candidates=candidates,
            embeddings=paper_embeddings,
            lam=lam,
            top_k=request_body.top_k,
        )
        final_candidates = reranked
    else:
        final_candidates = candidates[: request_body.top_k]

    results: list[PaperResult] = []
    for paper_id, score in final_candidates:
        paper = papers_map.get(paper_id)
        if paper is None:
            continue
        results.append(
            PaperResult(
                id=paper.id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                categories=paper.categories,
                citation_count=paper.citation_count,
                score=score,
            )
        )

    response = SearchResponse(
        query=request_body.query,
        results=results,
        total_candidates=total_candidates,
    )

    background_tasks.add_task(
        _log_search_clicks,
        async_session_factory,
        request_body.user_id,
        request_body.query,
        results,
    )

    return response
