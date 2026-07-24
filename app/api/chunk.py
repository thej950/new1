from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status

from app.core.logging import setup_logging
from app.schemas.chunk_response import ChunkResponse
from app.services.chunk_service import ChunkService

router = APIRouter()
logger = setup_logging()
chunk_service = ChunkService()


@router.post(
    "/chunk/{document_id}",
    response_model=ChunkResponse,
    tags=["Chunking"],
)
async def chunk_document(document_id: str = Path(..., description="UUID-based document identifier.")) -> ChunkResponse:
    """Generate JSON chunks from a processed document text file."""

    try:
        response = chunk_service.build_chunks(document_id=document_id)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in chunk endpoint for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to chunk the document.",
        ) from exc
