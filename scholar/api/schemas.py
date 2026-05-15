"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PaperResult(BaseModel):
    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    citation_count: int = 0
    score: float
    summary: Optional[str] = None
    glossary: List[str] = Field(default_factory=list)
    why_recommended: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    level: Literal["undergrad", "grad", "researcher"] = "grad"
    diversity_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: List[PaperResult]
    total_candidates: int


class UserHistoryItem(BaseModel):
    user_id: str
    paper_id: str


class UserProfile(BaseModel):
    user_id: str
    history_count: int


class ExplainRequest(BaseModel):
    user_id: Optional[str] = None
    level: Literal["undergrad", "grad", "researcher"] = "grad"
    why: str = "This paper matched your search query."
