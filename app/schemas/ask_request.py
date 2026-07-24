from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request payload for the question-answering endpoint."""

    question: str = Field(..., description="User question to answer using the local retrieval pipeline.", min_length=1)
    top_k: int = Field(default=5, description="Maximum number of retrieved chunks to use as context.", ge=1, le=20)
