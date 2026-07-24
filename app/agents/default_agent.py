from __future__ import annotations

from app.agents.base_agent import AgentClassification, BaseAgent


class DefaultAgent(BaseAgent):
    """Fallback agent used when no specialist reaches the routing confidence threshold."""

    def classify(self, question: str) -> AgentClassification:
        return AgentClassification(
            can_handle=True,
            confidence=0.50,
            reason="No specialist agent met the confidence threshold; using default enterprise assistant.",
        )
