from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz
from fastapi import HTTPException, status

from app.core.logging import setup_logging
from app.schemas.process_response import ProcessResponse

logger = setup_logging()


class PDFProcessor:
    """Reusable PDF text extraction and local processing service."""

    def __init__(
        self,
        uploads_dir: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.uploads_dir = uploads_dir or base_dir / "uploads"
        self.processed_dir = processed_dir or base_dir / "processed"

    def ensure_processed_directory(self) -> Path:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return self.processed_dir

    def _find_pdf(self, document_id: str) -> Path:
        pdf_path = self.uploads_dir / f"{document_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF document not found for the provided document_id.",
            )
        return pdf_path

    def _load_metadata(self, document_id: str) -> dict[str, str] | None:
        metadata_path = self.uploads_dir / f"{document_id}.json"
        if not metadata_path.exists():
            return None

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _clean_text(self, text: str) -> str:
        paragraphs: list[str] = []
        current_paragraph: list[str] = []

        for raw_line in text.splitlines():
            cleaned_line = re.sub(r"\s+", " ", raw_line).strip()
            if not cleaned_line:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(cleaned_line)

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        cleaned_text = "\n\n".join(paragraphs)
        return cleaned_text.strip()

    def process_document(self, document_id: str, original_filename: str | None = None) -> ProcessResponse:
        pdf_path = self._find_pdf(document_id)
        metadata = self._load_metadata(document_id)
        original_filename = original_filename or (metadata or {}).get("original_filename") or pdf_path.name
        self.ensure_processed_directory()

        try:
            with fitz.open(pdf_path) as document:
                if document.page_count == 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="The PDF is empty and cannot be processed.",
                    )

                page_texts: list[str] = []
                for page in document:
                    page_text = page.get_text("text")
                    if page_text:
                        page_texts.append(page_text)

                combined_text = "\n".join(page_texts)
                cleaned_text = self._clean_text(combined_text)

                if not cleaned_text:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No readable text could be extracted from the PDF.",
                    )

                txt_filename = f"{document_id}.txt"
                output_path = self.processed_dir / txt_filename
                output_path.write_text(cleaned_text, encoding="utf-8")

                logger.info(
                    "PDF processing completed: document_id=%s original_filename=%s pages=%s chars=%s output=%s",
                    document_id,
                    original_filename,
                    document.page_count,
                    len(cleaned_text),
                    output_path,
                )

                return ProcessResponse(
                    document_id=document_id,
                    original_pdf_filename=original_filename,
                    processed_text_filename=txt_filename,
                    total_pages=document.page_count,
                    extracted_character_count=len(cleaned_text),
                    processing_timestamp=datetime.now(timezone.utc).isoformat(),
                    processing_status="processed",
                )
        except HTTPException:
            raise
        except fitz.FileDataError as exc:
            logger.exception("Corrupted PDF detected for document_id=%s", document_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The PDF is corrupted and could not be processed.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected PDF processing failure for document_id=%s", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process the PDF document.",
            ) from exc
