from __future__ import annotations

from pydantic import BaseModel, Field


class AskSource(BaseModel):
    """Metadata for a source chunk used as retrieval context."""

    document_id: str = Field(..., description="Identifier of the source document.")
    chunk_number: int = Field(..., description="Chunk sequence number used in the answer generation context.")
    similarity_score: float = Field(..., description="Similarity score of the retrieved source chunk.")


class AskResponse(BaseModel):
    """Response model returned after question answering with a mock LLM."""

    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Deterministic answer produced by the mock LLM implementation.")
    sources: list[AskSource] = Field(default_factory=list, description="Source chunk metadata used to ground the answer.")
