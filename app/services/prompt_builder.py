from __future__ import annotations

from collections.abc import Sequence

from app.schemas.retrieval_response import RetrievalResponse
from app.services.conversation_memory import ConversationMessage


class PromptBuilder:
    """Build a prompt for a local, context-grounded mock LLM answer."""

    def _build_conversation_history(self, conversation_history: Sequence[ConversationMessage] | None) -> str:
        if not conversation_history:
            return ""

        history_lines = [
            f"{message.role.title()}:\n{message.content}"
            for message in conversation_history
        ]
        return "\n\n".join(history_lines)

    def build(
        self,
        question: str,
        retrieval_response: RetrievalResponse,
        conversation_history: Sequence[ConversationMessage] | None = None,
    ) -> str:
        context_chunks = [
            str(result.chunk_text).strip()
            for result in retrieval_response.results
            if str(result.chunk_text).strip()
        ]
        context = "\n\n".join(context_chunks)
        history = self._build_conversation_history(conversation_history)

        return (
            "------------------------------------------------\n"
            "System:\n\n"
            "You are an enterprise AI assistant.\n\n"
            "Answer ONLY using the provided context.\n\n"
            "If the answer is not available in the context, reply exactly:\n\n"
            '"I couldn\'t find this information in the uploaded documents."\n\n'
            "Do not hallucinate.\n"
            "Do not use outside knowledge.\n\n"
            "Conversation History:\n"
            f"{history}\n\n"
            "Current Context:\n"
            f"{context}\n\n"
            "Current Question:\n"
            f"{question}\n\n"
            "Answer:\n"
            "------------------------------------------------"
        )
