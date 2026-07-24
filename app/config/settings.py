from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration for local and AWS-backed runtime behavior."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="agentic-ai-platform")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    use_mock_llm: bool = Field(default=True)
    aws_region: str = Field(default="us-east-1")
    bedrock_model_id: str = Field(default="anthropic.claude-3-haiku-20240307-v1:0")
    multi_agent_enabled: bool = Field(default=True)
    multi_agent_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    session_timeout_minutes: int = Field(default=30, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
