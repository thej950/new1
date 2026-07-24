from __future__ import annotations

from abc import ABC, abstractmethod


class LLMService(ABC):
    """Abstract contract for any language model integration used by the platform."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a deterministic or provider-backed answer from a prompt."""


class MockLLMService(LLMService):
    """A deterministic mock LLM implementation that answers only from the provided context prompt."""

    def _extract_context(self, prompt: str) -> str:
        context_marker = "Current Context:\n" if "Current Context:\n" in prompt else "Context:\n"
        question_marker = "\n\nCurrent Question:\n" if "\n\nCurrent Question:\n" in prompt else "\n\nQuestion:\n"
        answer_marker = "\n\nAnswer:"

        if context_marker not in prompt:
            return ""

        context_section = prompt.split(context_marker, maxsplit=1)[1]
        context_section = context_section.split(question_marker, maxsplit=1)[0].strip()
        context_section = context_section.split(answer_marker, maxsplit=1)[0].strip()
        return context_section

    def generate(self, prompt: str) -> str:
        context = self._extract_context(prompt)
        if not context:
            return "I couldn't find this information in the uploaded documents."

        return f"Based on the uploaded documents:\n\n{context}"
