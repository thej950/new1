from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException, status

from app.core.logging import setup_logging
from app.services.llm_service import LLMService
from app.config.settings import get_settings

logger = setup_logging()
settings = get_settings()


class BedrockLLMService(LLMService):
    """Bedrock-backed LLM implementation using the boto3 Bedrock Runtime client."""

    def __init__(self) -> None:
        self.region = settings.aws_region
        self.model_id = settings.bedrock_model_id
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region,
        )

    def generate(self, prompt: str) -> str:
        start_time = time.perf_counter()

        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty.",
            )

        try:
            logger.info("Bedrock request sent for model_id=%s", self.model_id)
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
            )

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            response_body = json.loads(response["body"].read())
            logger.info(
                "Bedrock response received: model_id=%s latency_ms=%s",
                self.model_id,
                latency_ms,
            )

            if "content" not in response_body:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Bedrock response did not include content.",
                )

            content_blocks = response_body["content"]
            answer = "".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )

            if not answer:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Bedrock returned an empty answer.",
                )

            return answer.strip()
        except HTTPException:
            raise
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Bedrock invocation failed for model_id=%s", self.model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to invoke Amazon Bedrock model.",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Bedrock failure for model_id=%s", self.model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate answer via Bedrock.",
            ) from exc
