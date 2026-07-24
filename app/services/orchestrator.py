from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status

from app.agents.base_agent import AgentClassification, AgentSource, BaseAgent
from app.agents.default_agent import DefaultAgent
from app.agents.finance_agent import FinanceAgent
from app.agents.hr_agent import HRAgent
from app.agents.it_agent import ITAgent
from app.core.logging import setup_logging
from app.services.collaboration_service import AgentCollaborationResponse, CollaborationService
from app.services.conversation_memory import ConversationMessage
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.retrieval_service import RetrievalService

logger = setup_logging()


@dataclass(frozen=True)
class OrchestratorResult:
    """Result returned after agent selection and execution."""

    selected_agents: list[str]
    question: str
    answer: str
    sources: list[AgentSource]


class AgentOrchestrator:
    """Confidence-based orchestrator for selecting and executing enterprise agents."""

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
        top_k: int = 5,
        multi_agent_enabled: bool = True,
        multi_agent_threshold: float = 0.70,
        collaboration_service: CollaborationService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service
        self.top_k = top_k
        self.multi_agent_enabled = multi_agent_enabled
        self.multi_agent_threshold = multi_agent_threshold
        self.collaboration_service = collaboration_service or CollaborationService()
        self.agents = self._register_agents()
        self.default_agent = DefaultAgent(
            retrieval_service=self.retrieval_service,
            prompt_builder=self.prompt_builder,
            llm_service=self.llm_service,
            top_k=self.top_k,
        )

    def _register_agents(self) -> list[BaseAgent]:
        agents: list[BaseAgent] = [
            HRAgent(
                retrieval_service=self.retrieval_service,
                prompt_builder=self.prompt_builder,
                llm_service=self.llm_service,
                top_k=self.top_k,
            ),
            ITAgent(
                retrieval_service=self.retrieval_service,
                prompt_builder=self.prompt_builder,
                llm_service=self.llm_service,
                top_k=self.top_k,
            ),
            FinanceAgent(
                retrieval_service=self.retrieval_service,
                prompt_builder=self.prompt_builder,
                llm_service=self.llm_service,
                top_k=self.top_k,
            ),
        ]
        logger.info("Agent registration completed: agents=%s", [agent.name for agent in agents])
        return agents

    def classify_agents(self, question: str) -> list[tuple[BaseAgent, AgentClassification]]:
        classifications = [(agent, agent.classify(question)) for agent in self.agents]

        for agent, classification in classifications:
            logger.info(
                "Agent confidence: agent=%s can_handle=%s confidence=%.2f reason=%s question=%s",
                agent.name,
                classification.can_handle,
                classification.confidence,
                classification.reason,
                question,
            )

        return sorted(
            classifications,
            key=lambda item: item[1].confidence,
            reverse=True,
        )

    def select_agent(self, question: str) -> tuple[BaseAgent, AgentClassification]:
        ranked_classifications = self.classify_agents(question)
        selected_agent, classification = ranked_classifications[0]

        if classification.confidence < self.CONFIDENCE_THRESHOLD:
            default_classification = self.default_agent.classify(question)
            logger.info(
                "Confidence below threshold; using default agent=%s highest_agent=%s highest_confidence=%.2f threshold=%.2f question=%s",
                self.default_agent.name,
                selected_agent.name,
                classification.confidence,
                self.CONFIDENCE_THRESHOLD,
                question,
            )
            return self.default_agent, default_classification

        return selected_agent, classification

    def select_collaborating_agents(
        self,
        question: str,
        ranked_classifications: list[tuple[BaseAgent, AgentClassification]],
    ) -> list[tuple[BaseAgent, AgentClassification]]:
        selected_agents = [
            (agent, classification)
            for agent, classification in ranked_classifications
            if classification.confidence >= self.multi_agent_threshold
        ]

        if selected_agents:
            return selected_agents

        default_classification = self.default_agent.classify(question)
        return [(self.default_agent, default_classification)]

    def _handle_single_agent(
        self,
        question: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> OrchestratorResult:
        selected_agent, classification = self.select_agent(question)
        logger.info(
            "Selected agent: agent=%s confidence=%.2f reason=%s question=%s",
            selected_agent.name,
            classification.confidence,
            classification.reason,
            question,
        )
        logger.info("Agent execution started: agent=%s question=%s", selected_agent.name, question)

        agent_result = selected_agent.handle(
            question=question,
            conversation_history=conversation_history,
        )
        return OrchestratorResult(
            selected_agents=[selected_agent.name],
            question=question,
            answer=agent_result.answer,
            sources=agent_result.sources,
        )

    def _handle_multi_agent(
        self,
        question: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> OrchestratorResult:
        ranked_classifications = self.classify_agents(question)
        selected_agents = self.select_collaborating_agents(
            question=question,
            ranked_classifications=ranked_classifications,
        )
        logger.info(
            "Selected agents: agents=%s threshold=%.2f question=%s",
            [agent.name for agent, _classification in selected_agents],
            self.multi_agent_threshold,
            question,
        )

        successful_responses: list[AgentCollaborationResponse] = []
        for agent, classification in selected_agents:
            try:
                logger.info(
                    "Agent execution started: agent=%s confidence=%.2f reason=%s question=%s",
                    agent.name,
                    classification.confidence,
                    classification.reason,
                    question,
                )
                agent_result = agent.handle(
                    question=question,
                    conversation_history=conversation_history,
                )
                successful_responses.append(
                    AgentCollaborationResponse(
                        agent_name=agent.name,
                        result=agent_result,
                    )
                )
            except Exception:
                logger.exception("Agent execution failed; continuing: agent=%s question=%s", agent.name, question)

        if not successful_responses:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All selected agents failed to complete the request.",
            )

        collaboration_result = self.collaboration_service.merge(successful_responses)
        return OrchestratorResult(
            selected_agents=[response.agent_name for response in successful_responses],
            question=question,
            answer=collaboration_result.answer,
            sources=collaboration_result.sources,
        )

    def handle(
        self,
        question: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> OrchestratorResult:
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty.",
            )

        if self.multi_agent_enabled:
            return self._handle_multi_agent(
                question=question,
                conversation_history=conversation_history,
            )

        return self._handle_single_agent(
            question=question,
            conversation_history=conversation_history,
        )
