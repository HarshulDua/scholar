from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel


class Paper(SQLModel, table=True):
    __tablename__ = "papers"

    id: str = Field(primary_key=True)
    title: str
    abstract: str
    authors: str
    categories: str
    update_date: Optional[date] = Field(default=None)
    citation_count: int = Field(default=0)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(384)),
    )


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(primary_key=True)
    created_at: Optional[datetime] = Field(default=None)


class UserHistory(SQLModel, table=True):
    __tablename__ = "user_history"

    user_id: str = Field(foreign_key="users.id", primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", primary_key=True)
    clicked_at: Optional[datetime] = Field(default=None)


class ClickLog(SQLModel, table=True):
    __tablename__ = "click_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[str] = Field(default=None)
    paper_id: Optional[str] = Field(default=None)
    query: Optional[str] = Field(default=None)
    rank_position: Optional[int] = Field(default=None)
    clicked_at: Optional[datetime] = Field(default=None)
