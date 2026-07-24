from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, status

from app.core.logging import setup_logging
from app.schemas.embedding_response import EmbeddingResponse
from app.services.embedding_service import EmbeddingService

router = APIRouter()
logger = setup_logging()
embedding_service = EmbeddingService()


@router.post(
    "/embed/{document_id}",
    response_model=EmbeddingResponse,
    tags=["Embedding"],
)
async def embed_document(document_id: str = Path(..., description="UUID-based document identifier.")) -> EmbeddingResponse:
    """Generate embeddings for a document's chunk JSON and persist a local FAISS index."""

    try:
        response = embedding_service.build_embeddings(document_id=document_id)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in embed endpoint for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to embed the document.",
        ) from exc
