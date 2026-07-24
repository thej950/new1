from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer

from app.core.logging import setup_logging
from app.schemas.retrieval_response import RetrievalResponse

logger = setup_logging()


class RetrievalService:
    """Reusable retrieval service for local FAISS vector stores and chunk metadata."""

    def __init__(
        self,
        vector_store_dir: Path | None = None,
        chunk_dir: Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.vector_store_dir = vector_store_dir or base_dir / "vector_store"
        self.chunk_dir = chunk_dir or base_dir / "chunks"
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self._index: faiss.Index | None = None
        self._metadata: list[dict[str, object]] = []

    def _load_faiss_index(self) -> faiss.Index:
        index_path = self.vector_store_dir / "index.faiss"
        metadata_path = self.vector_store_dir / "metadata.json"

        if not index_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="FAISS index file not found at vector_store/index.faiss.",
            )

        if not metadata_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metadata file not found at vector_store/metadata.json.",
            )

        self._index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            self._metadata = json.load(metadata_file)

        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty for vector_store=%s", str(index_path))

        return self._index

    def _load_chunk_text(self, document_id: str, chunk_id: str) -> str:
        chunk_path = self.chunk_dir / f"{document_id}.json"
        if not chunk_path.exists():
            logger.warning("Chunk file not found for document_id=%s during retrieval", document_id)
            return ""

        with chunk_path.open("r", encoding="utf-8") as chunk_file:
            chunks = json.load(chunk_file)

        for chunk in chunks:
            if str(chunk.get("chunk_id")) == str(chunk_id):
                return str(chunk.get("text", ""))

        return ""

    def _normalise_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        return vector

    def search(self, question: str, top_k: int = 5) -> RetrievalResponse:
        if not question or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty.",
            )

        if top_k < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be greater than or equal to 1.",
            )

        try:
            index = self._load_faiss_index()
            query_embedding = self.model.encode([question], show_progress_bar=False)
            query_vector = self._normalise_query_embedding(query_embedding)

            if index.ntotal == 0:
                logger.info("Retrieval completed with no results for question=%s", question)
                return RetrievalResponse(question=question, results=[])

            k = min(top_k, index.ntotal)
            distances, indices = index.search(query_vector, k)

            results = []
            for distance, index_position in zip(distances[0], indices[0]):
                if index_position == -1:
                    continue

                metadata = self._metadata[index_position]
                chunk_text = self._load_chunk_text(
                    document_id=str(metadata.get("document_id", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                )

                similarity_score = float(1.0 / (1.0 + float(distance)))
                results.append(
                    {
                        "chunk_id": str(metadata.get("chunk_id", "")),
                        "document_id": str(metadata.get("document_id", "")),
                        "chunk_number": int(metadata.get("chunk_number", 0)),
                        "similarity_score": round(similarity_score, 6),
                        "chunk_text": chunk_text,
                    }
                )

            logger.info(
                "Retrieval completed: question=%s results_returned=%s top_k=%s model=%s",
                question,
                len(results),
                top_k,
                self.model_name,
            )

            return RetrievalResponse(question=question, results=results)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Retrieval failed for question=%s", question)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve relevant chunks.",
            ) from exc

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResponse:
        return self.search(question=question, top_k=top_k)
