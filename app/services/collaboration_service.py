from __future__ import annotations

from dataclasses import dataclass

from app.agents.base_agent import AgentResult, AgentSource
from app.core.logging import setup_logging

logger = setup_logging()


@dataclass(frozen=True)
class AgentCollaborationResponse:
    """A successful response from one selected agent."""

    agent_name: str
    result: AgentResult


@dataclass(frozen=True)
class CollaborationResult:
    """Merged multi-agent answer and deduplicated source metadata."""

    answer: str
    sources: list[AgentSource]


class CollaborationService:
    """Merge selected agent responses into one ordered answer with unique source chunks."""

    def merge(self, responses: list[AgentCollaborationResponse]) -> CollaborationResult:
        answer_sections = [
            f"## {response.agent_name}\n\n{response.result.answer}"
            for response in responses
        ]

        sources: list[AgentSource] = []
        seen_sources: set[tuple[str, int]] = set()

        for response in responses:
            for source in response.result.sources:
                source_key = (source.document_id, source.chunk_number)
                if source_key in seen_sources:
                    continue

                seen_sources.add(source_key)
                sources.append(source)

        logger.info(
            "Merge completed: responses=%s unique_sources=%s",
            len(responses),
            len(sources),
        )

        return CollaborationResult(
            answer="\n\n".join(answer_sections),
            sources=sources,
        )
