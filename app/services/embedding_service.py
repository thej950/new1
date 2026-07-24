from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer

from app.core.logging import setup_logging
from app.schemas.embedding_response import EmbeddingResponse

logger = setup_logging()


class EmbeddingService:
    """Reusable embedding service using Sentence Transformers and FAISS."""

    def __init__(
        self,
        chunk_dir: Path | None = None,
        vector_store_dir: Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.chunk_dir = chunk_dir or base_dir / "chunks"
        self.vector_store_dir = vector_store_dir or base_dir / "vector_store"
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def ensure_vector_store_directory(self) -> Path:
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        return self.vector_store_dir

    def _load_chunk_file(self, document_id: str) -> list[dict[str, object]]:
        chunk_path = self.chunk_dir / f"{document_id}.json"
        if not chunk_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk file not found for the provided document_id.",
            )

        with chunk_path.open("r", encoding="utf-8") as file_handle:
            chunks = json.load(file_handle)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The chunk file is empty and cannot be embedded.",
            )

        return chunks

    def _build_vector_store(self, embeddings: np.ndarray, metadata: list[dict[str, object]]) -> None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        vector_store_path = self.vector_store_dir / "index.faiss"
        metadata_path = self.vector_store_dir / "metadata.json"

        try:
            faiss.write_index(index, str(vector_store_path))
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)
        except Exception as exc:
            logger.exception("Failed to persist FAISS index for document_id=%s", metadata[0]["document_id"] if metadata else "unknown")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save the FAISS vector store.",
            ) from exc

    def build_embeddings(self, document_id: str) -> EmbeddingResponse:
        chunks = self._load_chunk_file(document_id)
        self.ensure_vector_store_directory()

        try:
            texts = [str(chunk["text"]) for chunk in chunks]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            if embeddings is None or len(embeddings) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding generation produced no vectors.",
                )

            embeddings_array = np.asarray(embeddings, dtype=np.float32)

            metadata_payload = []
            for chunk in chunks:
                metadata_payload.append(
                    {
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_number": chunk["chunk_number"],
                        "character_count": chunk["character_count"],
                    }
                )

            self._build_vector_store(embeddings_array, metadata_payload)

            logger.info(
                "Embedding generation completed: document_id=%s chunks_embedded=%s model=%s",
                document_id,
                len(chunks),
                self.model_name,
            )

            return EmbeddingResponse(
                document_id=document_id,
                chunks_embedded=len(chunks),
                embedding_model=self.model_name,
                vector_store=str(self.vector_store_dir / "index.faiss"),
                status="completed",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Embedding generation failed for document_id=%s", document_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate embeddings for the provided document.",
            ) from exc
