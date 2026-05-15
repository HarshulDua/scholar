"""Recommendation endpoint: personalized recommendations using user history."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from sqlmodel import select

from scholar.api.schemas import PaperResult, SearchResponse
from scholar.db.models import ClickLog, Paper, User, UserHistory
from scholar.recsys.mmr import adaptive_lambda, mmr_rerank

router = APIRouter()


async def _log_clicks(
    session_factory,
    user_id: str,
    results: list[tuple[str, float]],
) -> None:
    now = datetime.now()
    async with session_factory() as session:
        for rank, (paper_id, _) in enumerate(results, start=1):
            session.add(
                ClickLog(
                    user_id=user_id,
                    paper_id=paper_id,
                    query=None,
                    rank_position=rank,
                    clicked_at=now,
                )
            )
        await session.commit()


@router.get("/recommend/{user_id}", response_model=SearchResponse)
async def recommend(
    user_id: str,
    level: str = "grad",
    request: Request = None,  # type: ignore[assignment]  # FastAPI-injected
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]  # FastAPI-injected
) -> SearchResponse:
    async_session_factory = request.app.state.async_session_factory
    dense_retriever = getattr(request.app.state, "dense_retriever", None)
    ltr_model = getattr(request.app.state, "ltr_model", None)

    if dense_retriever is None:
        raise HTTPException(status_code=503, detail="Dense index not loaded.")

    async with async_session_factory() as session:
        user_result = await session.execute(select(User).where(User.id == user_id))
        user = user_result.scalars().first()
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

        history_result = await session.execute(
            select(UserHistory.paper_id).where(UserHistory.user_id == user_id)
        )
        history_ids = list(history_result.scalars().all())


    if not history_ids:
        raise HTTPException(
            status_code=422,
            detail="User has no history. Add papers to history before requesting recommendations.",
        )

    id_to_idx: dict[str, int] = getattr(request.app.state, "id_to_idx", {})
    emb_array = getattr(dense_retriever, "_embeddings", None)
    history_embeddings: list[np.ndarray] = []
    for hid in history_ids:
        idx = id_to_idx.get(hid)
        if idx is not None:
            if emb_array is not None:
                history_embeddings.append(emb_array[idx])
            elif dense_retriever._index is not None:
                vec = dense_retriever._index.reconstruct(idx)
                history_embeddings.append(np.array(vec, dtype=np.float32))

    if history_embeddings:
        user_emb = np.mean(history_embeddings, axis=0).astype(np.float32)
    else:
        raise HTTPException(
            status_code=503,
            detail="Could not build user embedding: history papers not in FAISS index.",
        )

    candidates = dense_retriever.query_by_vector(user_emb, top_k=200)

    # Blend VAE collaborative-filtering scores into FAISS scores.
    # The VAE is trained on the 158-paper interaction vocab; for candidates that
    # appear in that vocab we boost/dampen their score using the VAE's reconstruction
    # probability. Alpha=0.3 keeps FAISS as the dominant signal.
    try:
        import json
        from pathlib import Path

        import torch

        from scholar.recsys.mult_vae import _load_trained_model

        vae_model = _load_trained_model()
        vocab_path = Path("./data/vae_vocab.json")
        if vae_model is not None and vocab_path.exists():
            with open(vocab_path) as f:
                vae_paper_ids: list[str] = json.load(f)
            vae_id_to_idx = {pid: i for i, pid in enumerate(vae_paper_ids)}
            x_hist = torch.zeros(1, vae_model.vocab_size, dtype=torch.float32)
            for hid in history_ids:
                if hid in vae_id_to_idx:
                    x_hist[0, vae_id_to_idx[hid]] = 1.0
            if x_hist.sum() > 0:
                vae_model.eval()
                with torch.no_grad():
                    recon, _, _ = vae_model(x_hist)
                vae_scores = torch.exp(recon).squeeze(0).numpy()  # softmax probs
                alpha = 0.3
                blended: list[tuple[str, float]] = []
                for pid, fscore in candidates:
                    vidx = vae_id_to_idx.get(pid)
                    if vidx is not None:
                        blended.append((pid, fscore * (1.0 + alpha * float(vae_scores[vidx]))))
                    else:
                        blended.append((pid, fscore))
                candidates = sorted(blended, key=lambda x: x[1], reverse=True)
    except Exception:
        pass  # VAE blending is optional; fall back to pure FAISS order
    history_set = set(history_ids)
    candidates = [(pid, score) for pid, score in candidates if pid not in history_set]

    # Fetch paper metadata for LTR / exploration
    candidate_ids = [pid for pid, _ in candidates]
    async with async_session_factory() as session:
        result = await session.execute(select(Paper).where(Paper.id.in_(candidate_ids)))  # type: ignore[attr-defined]
        papers = {p.id: p for p in result.scalars().all()}

    # Collect paper embeddings using startup-cached id_to_idx
    paper_embeddings: dict[str, np.ndarray] = {}
    for pid in candidate_ids:
        idx = id_to_idx.get(pid)
        if idx is not None:
            if emb_array is not None:
                paper_embeddings[pid] = emb_array[idx]
            elif dense_retriever._index is not None:
                vec = dense_retriever._index.reconstruct(idx)
                paper_embeddings[pid] = np.array(vec, dtype=np.float32)

    # LTR re-rank (top-50 input to MMR), fall back to FAISS order if no model
    if ltr_model is not None and ltr_model._model is not None:
        history_cats = [papers[hid].categories for hid in history_ids if hid in papers]
        ltr_ranked = ltr_model.score(
            user_emb=user_emb,
            candidates=candidates,
            paper_meta=papers,
            paper_embeddings=paper_embeddings,
            history_cats=history_cats,
        )
        mmr_input = ltr_ranked[:50]
    else:
        mmr_input = candidates[:50]

    lam = adaptive_lambda(history_embeddings) if history_embeddings else 0.3

    reranked = mmr_rerank(
        candidates=mmr_input,
        embeddings=paper_embeddings,
        lam=lam,
        top_k=10,
    )

    # Epsilon-greedy exploration
    try:
        from scholar.recsys.exploration import epsilon_greedy_inject

        reranked = epsilon_greedy_inject(
            ranked=reranked,
            all_paper_ids=dense_retriever._paper_ids,
            history_ids=history_set,
            epsilon=0.1,
            paper_meta=papers,
        )
    except Exception:
        pass

    # Fetch any injected papers not yet in papers dict
    injected_ids = [pid for pid, _ in reranked if pid not in papers]
    if injected_ids:
        async with async_session_factory() as session:
            result = await session.execute(select(Paper).where(Paper.id.in_(injected_ids)))  # type: ignore[attr-defined]
            for p in result.scalars().all():
                papers[p.id] = p

    results: list[PaperResult] = []
    for paper_id, score in reranked:
        paper = papers.get(paper_id)
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
                score=float(score),
            )
        )

    if background_tasks is not None:
        background_tasks.add_task(
            _log_clicks,
            async_session_factory,
            user_id,
            reranked,
        )

    return SearchResponse(
        query=f"Personalized recommendations for user {user_id}",
        results=results,
        total_candidates=len(candidates),
    )
