from __future__ import annotations

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Request payload for the agent-orchestrated chat endpoint."""

    session_id: str | None = Field(default=None, description="Optional conversation session identifier.")
    question: str = Field(..., description="User question to route to the appropriate enterprise agent.", min_length=1)
