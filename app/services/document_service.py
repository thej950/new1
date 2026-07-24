from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.logging import setup_logging
from app.schemas.upload_response import UploadResponse

logger = setup_logging()


class DocumentService:
    """Handles local PDF upload validation and persistence."""

    def __init__(self, uploads_dir: Path | None = None, max_size_bytes: int = 20 * 1024 * 1024):
        self.uploads_dir = uploads_dir or Path(__file__).resolve().parents[2] / "uploads"
        self.max_size_bytes = max_size_bytes
        self.allowed_extension = ".pdf"
        self.allowed_content_type = "application/pdf"

    def ensure_upload_directory(self) -> Path:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        return self.uploads_dir

    def _is_empty(self, file: UploadFile) -> bool:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        return size == 0

    def _validate_filename(self, filename: str | None) -> str:
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A filename is required.",
            )

        if not filename.lower().endswith(self.allowed_extension):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed.",
            )

        return filename

    def _validate_size(self, file: UploadFile) -> int:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        if file_size > self.max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 20 MB limit.",
            )

        return file_size

    def _validate_content_type(self, file: UploadFile) -> None:
        content_type = (file.content_type or "").lower()
        if content_type and content_type != self.allowed_content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed.",
            )

    async def save_document(self, file: UploadFile) -> UploadResponse:
        original_filename = self._validate_filename(file.filename)
        self._validate_content_type(file)
        file_size = self._validate_size(file)

        if self._is_empty(file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        self.ensure_upload_directory()

        document_id = uuid4()
        stored_filename = f"{document_id}{self.allowed_extension}"
        storage_path = self.uploads_dir / stored_filename
        metadata_path = self.uploads_dir / f"{document_id}.json"

        file_content = await file.read()
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        with storage_path.open("wb") as destination:
            destination.write(file_content)

        metadata = {
            "document_id": str(document_id),
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "file_size": file_size,
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "storage_path": str(storage_path),
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        logger.info(
            "Document upload successful: document_id=%s original_filename=%s stored_filename=%s file_size=%s",
            document_id,
            original_filename,
            stored_filename,
            file_size,
        )

        return UploadResponse(
            document_id=str(document_id),
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_size=file_size,
            upload_timestamp=datetime.now(timezone.utc).isoformat(),
            storage_path=str(storage_path),
            status="uploaded",
        )
