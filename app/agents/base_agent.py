from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.logging import setup_logging
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService
from app.services.conversation_memory import ConversationMessage

logger = setup_logging()


@dataclass(frozen=True)
class AgentClassification:
    """Confidence-based routing decision returned by each agent."""

    can_handle: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class AgentSource:
    """Source metadata returned by an agent after retrieval."""

    document_id: str
    chunk_number: int
    similarity_score: float


@dataclass(frozen=True)
class AgentResult:
    """Agent execution result containing the grounded answer and source metadata."""

    answer: str
    sources: list[AgentSource]


class BaseAgent(ABC):
    """Base class for confidence-routed enterprise agents backed by the shared RAG pipeline."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
        top_k: int = 5,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.top_k = top_k

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def classify(self, question: str) -> AgentClassification:
        """Return the agent's confidence and rationale for handling the incoming question."""

    def handle(self, question: str, conversation_history: list[ConversationMessage] | None = None) -> AgentResult:
        """Execute the shared retrieval, prompt building, and LLM flow for this agent."""

        retrieval_response = self.retrieval_service.retrieve(
            question=question,
            top_k=self.top_k,
        )
        logger.info(
            "Retrieval completed: agent=%s question=%s results=%s",
            self.name,
            question,
            len(retrieval_response.results),
        )

        prompt = self.prompt_builder.build(
            question=question,
            retrieval_response=retrieval_response,
            conversation_history=conversation_history,
        )
        logger.info(
            "History appended: agent=%s question=%s history_messages=%s",
            self.name,
            question,
            len(conversation_history or []),
        )
        logger.info(
            "Prompt generated: agent=%s question=%s prompt_length=%s",
            self.name,
            question,
            len(prompt),
        )

        answer = self.llm_service.generate(prompt)
        logger.info("LLM completed: agent=%s question=%s", self.name, question)

        sources = [
            AgentSource(
                document_id=result.document_id,
                chunk_number=result.chunk_number,
                similarity_score=result.similarity_score,
            )
            for result in retrieval_response.results
        ]

        return AgentResult(answer=answer, sources=sources)
