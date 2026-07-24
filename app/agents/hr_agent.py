from __future__ import annotations

from app.agents.base_agent import AgentClassification, BaseAgent


class HRAgent(BaseAgent):
    """Agent responsible for HR-related employee policy questions."""

    KEYWORDS = (
        "leave",
        "attendance",
        "holiday",
        "salary",
        "benefits",
        "employee",
        "policy",
        "recruitment",
    )

    HIGH_CONFIDENCE_PHRASES = (
        "leave policy",
        "annual leave",
        "employee benefits",
        "attendance policy",
        "holiday policy",
        "salary policy",
    )

    def classify(self, question: str) -> AgentClassification:
        normalized_question = question.lower()
        matched_keywords = [keyword for keyword in self.KEYWORDS if keyword in normalized_question]

        if any(phrase in normalized_question for phrase in self.HIGH_CONFIDENCE_PHRASES):
            return AgentClassification(
                can_handle=True,
                confidence=0.95,
                reason="Detected HR policy intent.",
            )

        if len(matched_keywords) >= 2:
            return AgentClassification(
                can_handle=True,
                confidence=0.88,
                reason=f"Detected multiple HR terms: {', '.join(matched_keywords)}.",
            )

        if matched_keywords:
            return AgentClassification(
                can_handle=True,
                confidence=0.72,
                reason=f"Detected HR term: {matched_keywords[0]}.",
            )

        return AgentClassification(
            can_handle=False,
            confidence=0.10,
            reason="No HR intent detected.",
        )
