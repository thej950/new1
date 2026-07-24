from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response model returned after a successful document upload."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document.")
    original_filename: str = Field(..., description="Original uploaded file name.")
    stored_filename: str = Field(..., description="Unique storage filename.")
    file_size: int = Field(..., description="Uploaded file size in bytes.")
    upload_timestamp: str = Field(..., description="Upload timestamp in ISO 8601 format.")
    storage_path: str = Field(..., description="Local filesystem path for the stored file.")
    status: str = Field(default="uploaded", description="Upload status.")
