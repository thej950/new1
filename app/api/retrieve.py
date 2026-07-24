from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import setup_logging
from app.schemas.retrieval_response import RetrieveRequest, RetrievalResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter()
logger = setup_logging()
retrieval_service = RetrievalService()


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    tags=["Retrieval"],
)
async def retrieve_chunks(request: RetrieveRequest) -> RetrievalResponse:
    """Retrieve the most relevant document chunks for a user question using FAISS and the local sentence-transformer model."""

    try:
        response = retrieval_service.retrieve(
            question=request.question,
            top_k=request.top_k,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in retrieve endpoint for question=%s", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve relevant chunks.",
        ) from exc
