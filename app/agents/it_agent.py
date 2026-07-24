from __future__ import annotations

from app.agents.base_agent import AgentClassification, BaseAgent


class ITAgent(BaseAgent):
    """Agent responsible for internal IT support and systems questions."""

    KEYWORDS = (
        "password",
        "vpn",
        "login",
        "email",
        "software",
        "network",
        "laptop",
        "system",
    )

    HIGH_CONFIDENCE_PHRASES = (
        "reset password",
        "vpn password",
        "reset vpn",
        "vpn login",
        "email login",
        "network issue",
        "software access",
    )

    def classify(self, question: str) -> AgentClassification:
        normalized_question = question.lower()
        matched_keywords = [keyword for keyword in self.KEYWORDS if keyword in normalized_question]

        if any(phrase in normalized_question for phrase in self.HIGH_CONFIDENCE_PHRASES):
            return AgentClassification(
                can_handle=True,
                confidence=0.96,
                reason="Detected IT access or support intent.",
            )

        if len(matched_keywords) >= 2:
            return AgentClassification(
                can_handle=True,
                confidence=0.89,
                reason=f"Detected multiple IT terms: {', '.join(matched_keywords)}.",
            )

        if matched_keywords:
            return AgentClassification(
                can_handle=True,
                confidence=0.74,
                reason=f"Detected IT term: {matched_keywords[0]}.",
            )

        return AgentClassification(
            can_handle=False,
            confidence=0.10,
            reason="No IT intent detected.",
        )
