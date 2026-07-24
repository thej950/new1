from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkResponse(BaseModel):
    """Metadata returned after chunk generation is complete."""

    document_id: str = Field(..., description="UUID identifier of the source document.")
    total_chunks: int = Field(..., description="Total number of generated chunks.")
    average_chunk_size: float = Field(..., description="Average size of generated chunks in characters.")
    output_file: str = Field(..., description="Path to the JSON file containing the saved chunks.")
    status: str = Field(default="chunked", description="Chunking status.")
