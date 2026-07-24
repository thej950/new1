from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """Request payload for the retrieval endpoint."""

    question: str = Field(..., description="User question to retrieve relevant chunks for.", min_length=1)
    top_k: int = Field(default=5, description="Maximum number of similar chunks to return.", ge=1, le=20)


class RetrievalResult(BaseModel):
    """A single retrieved chunk result."""

    chunk_id: str = Field(..., description="Identifier for the retrieved chunk.")
    document_id: str = Field(..., description="Identifier of the source document.")
    chunk_number: int = Field(..., description="Chunk sequence number.")
    similarity_score: float = Field(..., description="Similarity score between the query embedding and the stored chunk embedding.")
    chunk_text: str = Field(..., description="Chunk content text.")


class RetrievalResponse(BaseModel):
    """Response model returned after query retrieval from the local FAISS vector store."""

    question: str = Field(..., description="Original user question.")
    results: list[RetrievalResult] = Field(default_factory=list, description="Top-k retrieved chunks ordered by similarity.")
