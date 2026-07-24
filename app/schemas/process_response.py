from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessResponse(BaseModel):
    """Response model returned after PDF text extraction and cleanup."""

    document_id: str = Field(..., description="UUID identifier of the uploaded document.")
    original_pdf_filename: str = Field(..., description="Original uploaded PDF filename.")
    processed_text_filename: str = Field(..., description="Generated text file name in the processed directory.")
    total_pages: int = Field(..., description="Total number of pages in the PDF.")
    extracted_character_count: int = Field(..., description="Number of characters in the cleaned extracted text.")
    processing_timestamp: str = Field(..., description="Processing timestamp in ISO 8601 format.")
    processing_status: str = Field(default="processed", description="Processing status.")
