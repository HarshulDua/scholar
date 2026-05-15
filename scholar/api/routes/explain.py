"""Explain endpoint: generate AI explanation for a paper."""
from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from scholar.api.schemas import ExplainRequest
from scholar.db.models import Paper

router = APIRouter()


@router.post("/explain/{paper_id}")
async def explain_paper(
    paper_id: str,
    body: ExplainRequest,
    request: Request,
) -> dict:
    async_session_factory = request.app.state.async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(Paper).where(Paper.id == paper_id))
        paper = result.scalars().first()

    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found.")

    from scholar.config import settings
    if not settings.generation_available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generation is disabled: Phi-3-mini requires a CUDA GPU. "
                "Install a CUDA-enabled PyTorch build and retry."
            ),
        )

    try:
        from scholar.generation.inference import Phi3Generator
        explanation = Phi3Generator.get_instance().generate(
            title=paper.title,
            abstract=paper.abstract,
            why=body.why,
            level=body.level,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Generation failed: {str(e)}",
        )

    return {
        "paper_id": paper_id,
        "title": paper.title,
        "level": body.level,
        **explanation,
    }
