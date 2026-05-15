"""User management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from sqlmodel import select

from scholar.api.schemas import UserHistoryItem, UserProfile
from scholar.db.models import User, UserHistory

router = APIRouter()


@router.post("/user", response_model=dict)
async def create_user(request: Request) -> dict:
    async_session_factory = request.app.state.async_session_factory

    user_id = str(uuid.uuid4())
    user = User(id=user_id, created_at=datetime.now())

    async with async_session_factory() as session:
        session.add(user)
        await session.commit()

    return {"user_id": user_id}


@router.get("/user/{user_id}", response_model=UserProfile)
async def get_user(user_id: str, request: Request) -> UserProfile:
    async_session_factory = request.app.state.async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalars().first()
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

        history_result = await session.execute(
            select(UserHistory).where(UserHistory.user_id == user_id)
        )
        history = history_result.scalars().all()

    return UserProfile(user_id=user_id, history_count=len(history))


@router.post("/user/{user_id}/history", response_model=dict)
async def add_to_history(
    user_id: str,
    body: UserHistoryItem,
    request: Request,
) -> dict:
    async_session_factory = request.app.state.async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalars().first()
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

        existing_result = await session.execute(
            select(UserHistory).where(
                UserHistory.user_id == user_id,
                UserHistory.paper_id == body.paper_id,
            )
        )
        existing = existing_result.scalars().first()

        if existing is None:
            history_item = UserHistory(
                user_id=user_id,
                paper_id=body.paper_id,
                clicked_at=datetime.now(),
            )
            session.add(history_item)
            await session.commit()

    return {"user_id": user_id, "paper_id": body.paper_id, "added": existing is None}
