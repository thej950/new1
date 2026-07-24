from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.logging import setup_logging
from app.schemas.upload_response import UploadResponse
from app.services.document_service import DocumentService

router = APIRouter()
logger = setup_logging()
document_service = DocumentService()


@router.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF document to the local uploads directory."""

    try:
        response = await document_service.save_document(file)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected upload failure for file=%s", getattr(file, "filename", "unknown"))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the uploaded document.",
        ) from exc
