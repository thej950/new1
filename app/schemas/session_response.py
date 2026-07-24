from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ConversationMessageResponse(BaseModel):
    """Message returned from a conversation session."""

    role: str = Field(..., description="Message role, such as user or assistant.")
    content: str = Field(..., description="Message content.")
    timestamp: datetime = Field(..., description="UTC timestamp when the message was stored.")


class ConversationSessionResponse(BaseModel):
    """Chat history returned for a conversation session."""

    session_id: str = Field(..., description="Conversation session identifier.")
    messages: list[ConversationMessageResponse] = Field(default_factory=list, description="Stored chat messages.")


class DeleteSessionResponse(BaseModel):
    """Response returned after deleting a conversation session."""

    session_id: str = Field(..., description="Deleted conversation session identifier.")
    deleted: bool = Field(..., description="Whether the session was deleted.")
