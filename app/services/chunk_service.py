from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import setup_logging
from app.schemas.chunk_response import ChunkResponse

logger = setup_logging()


class ChunkService:
    """Reusable document chunking service for local text processing."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        chunks_dir: Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.processed_dir = processed_dir or base_dir / "processed"
        self.chunks_dir = chunks_dir or base_dir / "chunks"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ensure_chunks_directory(self) -> Path:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        return self.chunks_dir

    def _load_processed_text(self, document_id: str) -> str:
        text_path = self.processed_dir / f"{document_id}.txt"
        if not text_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Processed text not found for the provided document_id.",
            )

        text = text_path.read_text(encoding="utf-8")
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The processed document is empty and cannot be chunked.",
            )

        return text

    def _build_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def build_chunks(self, document_id: str) -> ChunkResponse:
        text = self._load_processed_text(document_id)
        self.ensure_chunks_directory()

        try:
            splitter = self._build_text_splitter()
            chunks = splitter.split_text(text)

            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No chunks were produced from the processed document.",
                )

            serialized_chunks: list[dict[str, object]] = []
            for index, chunk_text in enumerate(chunks, start=1):
                chunk_payload = {
                    "chunk_id": str(uuid4()),
                    "document_id": document_id,
                    "chunk_number": index,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }
                serialized_chunks.append(chunk_payload)

            output_path = self.chunks_dir / f"{document_id}.json"
            with output_path.open("w", encoding="utf-8") as file_handle:
                json.dump(serialized_chunks, file_handle, indent=2)

            average_chunk_size = sum(item["character_count"] for item in serialized_chunks) / len(serialized_chunks)

            logger.info(
                "Chunking completed: document_id=%s total_chunks=%s average_chunk_size=%s output_file=%s",
                document_id,
                len(serialized_chunks),
                round(average_chunk_size, 2),
                output_path,
            )

            return ChunkResponse(
                document_id=document_id,
                total_chunks=len(serialized_chunks),
                average_chunk_size=round(average_chunk_size, 2),
                output_file=str(output_path),
                status="chunked",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Chunking failed for document_id=%s", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to chunk the processed document.",
            ) from exc
