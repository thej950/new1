from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Response model returned after a document has been embedded into FAISS."""

    document_id: str = Field(..., description="UUID identifier of the source document.")
    chunks_embedded: int = Field(..., description="Number of chunks embedded into the vector store.")
    embedding_model: str = Field(..., description="Sentence Transformers model used for embedding.")
    vector_store: str = Field(..., description="Path to the saved FAISS index file.")
    status: str = Field(default="completed", description="Embedding status.")
