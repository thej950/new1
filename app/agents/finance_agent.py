from __future__ import annotations

from app.agents.base_agent import AgentClassification, BaseAgent


class FinanceAgent(BaseAgent):
    """Agent responsible for finance, purchase, and reimbursement questions."""

    KEYWORDS = (
        "invoice",
        "expense",
        "payment",
        "tax",
        "budget",
        "purchase",
        "reimbursement",
        "finance",
    )

    HIGH_CONFIDENCE_PHRASES = (
        "expense reimbursement",
        "submit expense",
        "invoice payment",
        "tax reimbursement",
        "purchase request",
        "finance approval",
    )

    def classify(self, question: str) -> AgentClassification:
        normalized_question = question.lower()
        matched_keywords = [keyword for keyword in self.KEYWORDS if keyword in normalized_question]

        if any(phrase in normalized_question for phrase in self.HIGH_CONFIDENCE_PHRASES):
            return AgentClassification(
                can_handle=True,
                confidence=0.93,
                reason="Detected finance reimbursement or payment intent.",
            )

        if len(matched_keywords) >= 2:
            return AgentClassification(
                can_handle=True,
                confidence=0.87,
                reason=f"Detected multiple finance terms: {', '.join(matched_keywords)}.",
            )

        if matched_keywords:
            return AgentClassification(
                can_handle=True,
                confidence=0.73,
                reason=f"Detected finance term: {matched_keywords[0]}.",
            )

        return AgentClassification(
            can_handle=False,
            confidence=0.10,
            reason="No finance intent detected.",
        )
