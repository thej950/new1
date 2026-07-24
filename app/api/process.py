from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status

from app.core.logging import setup_logging
from app.schemas.process_response import ProcessResponse
from app.services.pdf_processor import PDFProcessor

router = APIRouter()
logger = setup_logging()
pdf_processor = PDFProcessor()


@router.post(
    "/process/{document_id}",
    response_model=ProcessResponse,
    tags=["Processing"],
)
async def process_document(document_id: str = Path(..., description="UUID-based document identifier.")) -> ProcessResponse:
    """Process a previously uploaded PDF document and extract its readable text."""

    try:
        response = pdf_processor.process_document(document_id=document_id)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in process endpoint for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the PDF document.",
        ) from exc
