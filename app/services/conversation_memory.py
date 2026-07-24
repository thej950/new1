from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.logging import setup_logging

logger = setup_logging()


@dataclass(frozen=True)
class ConversationMessage:
    """Single message stored in an in-memory conversation session."""

    role: str
    content: str
    timestamp: datetime


@dataclass
class ConversationSession:
    """In-memory conversation state for one agent-chat session."""

    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConversationMemory:
    """In-memory conversation store with inactivity-based session expiration."""

    def __init__(self, session_timeout_minutes: int = 30) -> None:
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._sessions: dict[str, ConversationSession] = {}

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _is_expired(self, session: ConversationSession, now: datetime) -> bool:
        return now - session.last_activity_at > self.session_timeout

    def expire_inactive_sessions(self) -> None:
        now = self._now()
        expired_session_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_expired(session, now)
        ]

        for session_id in expired_session_ids:
            del self._sessions[session_id]
            logger.info("Session expired: session_id=%s", session_id)

    def create_session(self) -> ConversationSession:
        self.expire_inactive_sessions()
        session_id = uuid4().hex
        session = ConversationSession(session_id=session_id)
        self._sessions[session_id] = session
        logger.info("Session created: session_id=%s", session_id)
        return session

    def get_session(self, session_id: str) -> ConversationSession:
        self.expire_inactive_sessions()
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation session not found.",
            )

        session.last_activity_at = self._now()
        logger.info("Session loaded: session_id=%s messages=%s", session_id, len(session.messages))
        return session

    def get_or_create_session(self, session_id: str | None) -> ConversationSession:
        if session_id:
            return self.get_session(session_id)

        return self.create_session()

    def append_message(self, session_id: str, role: str, content: str) -> ConversationMessage:
        session = self.get_session(session_id)
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=self._now(),
        )
        session.messages.append(message)
        session.last_activity_at = message.timestamp
        logger.info(
            "History saved: session_id=%s role=%s total_messages=%s",
            session_id,
            role,
            len(session.messages),
        )
        return message

    def clear_session(self, session_id: str) -> None:
        self.expire_inactive_sessions()
        if session_id not in self._sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation session not found.",
            )

        del self._sessions[session_id]
        logger.info("Session cleared: session_id=%s", session_id)
