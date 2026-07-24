from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.config.settings import get_settings
from app.core.logging import setup_logging
from app.schemas.ask_request import AskRequest
from app.schemas.ask_response import AskResponse
from app.services.bedrock_llm_service import BedrockLLMService
from app.services.llm_service import LLMService, MockLLMService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService

router = APIRouter()
logger = setup_logging()
settings = get_settings()
retrieval_service = RetrievalService()
prompt_builder = PromptBuilder()
llm_service: LLMService = MockLLMService() if settings.use_mock_llm else BedrockLLMService()


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["Ask"],
)
async def ask_question(request: AskRequest) -> AskResponse:
    """Answer a user question using the local retrieval pipeline and a mock LLM abstraction."""

    try:
        retrieval_response = retrieval_service.search(
            question=request.question,
            top_k=request.top_k,
        )
        prompt = prompt_builder.build(
            question=request.question,
            retrieval_response=retrieval_response,
        )
        answer = llm_service.generate(prompt)

        sources = [
            {
                "document_id": result.document_id,
                "chunk_number": result.chunk_number,
                "similarity_score": result.similarity_score,
            }
            for result in retrieval_response.results
        ]

        logger.info(
            "Answer generation completed: question=%s sources=%s",
            request.question,
            len(sources),
        )

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in ask endpoint for question=%s", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to answer the question.",
        ) from exc
