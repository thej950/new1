from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.config.settings import get_settings
from app.core.logging import setup_logging
from app.schemas.agent_chat_request import AgentChatRequest
from app.schemas.agent_chat_response import AgentChatResponse
from app.schemas.session_response import ConversationSessionResponse, DeleteSessionResponse
from app.services.bedrock_llm_service import BedrockLLMService
from app.services.conversation_memory import ConversationMemory
from app.services.llm_service import LLMService, MockLLMService
from app.services.orchestrator import AgentOrchestrator
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["Agent Chat"])
logger = setup_logging()
settings = get_settings()

retrieval_service = RetrievalService()
prompt_builder = PromptBuilder()
llm_service: LLMService = MockLLMService() if settings.use_mock_llm else BedrockLLMService()
conversation_memory = ConversationMemory(session_timeout_minutes=settings.session_timeout_minutes)
agent_orchestrator = AgentOrchestrator(
    retrieval_service=retrieval_service,
    prompt_builder=prompt_builder,
    llm_service=llm_service,
    multi_agent_enabled=settings.multi_agent_enabled,
    multi_agent_threshold=settings.multi_agent_threshold,
)


def get_agent_orchestrator() -> AgentOrchestrator:
    return agent_orchestrator


def get_conversation_memory() -> ConversationMemory:
    return conversation_memory


@router.post(
    "/agent-chat",
    response_model=AgentChatResponse,
)
async def agent_chat(
    request: AgentChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    memory: ConversationMemory = Depends(get_conversation_memory),
) -> AgentChatResponse:
    """Route a question to the appropriate enterprise agent and return a grounded answer."""

    logger.info("Incoming agent-chat request: question=%s", request.question)

    try:
        session = memory.get_or_create_session(request.session_id)
        conversation_history = list(session.messages)
        result = orchestrator.handle(
            question=request.question,
            conversation_history=conversation_history,
        )
        memory.append_message(
            session_id=session.session_id,
            role="user",
            content=request.question,
        )
        memory.append_message(
            session_id=session.session_id,
            role="assistant",
            content=result.answer,
        )
        response = AgentChatResponse(
            session_id=session.session_id,
            selected_agents=result.selected_agents,
            question=result.question,
            answer=result.answer,
            sources=[
                {
                    "document_id": source.document_id,
                    "chunk_number": source.chunk_number,
                    "similarity_score": source.similarity_score,
                }
                for source in result.sources
            ],
        )
        logger.info(
            "Response returned: endpoint=/agent-chat session_id=%s selected_agents=%s sources=%s",
            response.session_id,
            response.selected_agents,
            len(response.sources),
        )
        logger.info(
            "Agent chat completed: session_id=%s question=%s selected_agents=%s",
            response.session_id,
            request.question,
            response.selected_agents,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected failure in agent-chat endpoint for question=%s", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete agent chat request.",
        ) from exc


@router.get(
    "/sessions/{session_id}",
    response_model=ConversationSessionResponse,
    tags=["Sessions"],
)
async def get_session(
    session_id: str,
    memory: ConversationMemory = Depends(get_conversation_memory),
) -> ConversationSessionResponse:
    """Return chat history for an active conversation session."""

    session = memory.get_session(session_id)
    return ConversationSessionResponse(
        session_id=session.session_id,
        messages=[
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
            }
            for message in session.messages
        ],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["Sessions"],
)
async def delete_session(
    session_id: str,
    memory: ConversationMemory = Depends(get_conversation_memory),
) -> DeleteSessionResponse:
    """Delete an active conversation session and its chat history."""

    memory.clear_session(session_id)
    return DeleteSessionResponse(session_id=session_id, deleted=True)
