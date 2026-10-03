"""ChromaDB Vector Store Service for MeetWise AI.

Manages persistent ChromaDB vector storage for meeting transcript chunks,
supporting semantic similarity search and cross-meeting knowledge retrieval.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..embeddings.chunker import TranscriptChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Exception raised when vector database operations fail."""
    pass


class ChromaService:
    """Encapsulates ChromaDB persistent client and operations."""

    COLLECTION_NAME = "meetwise_transcripts"

    def __init__(self, persist_dir: str = "./data/chromadb"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _get_client(self):
        """Lazy load persistent ChromaDB client and collection."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"Connecting to ChromaDB at: {self.persist_dir}...")

                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "MeetWise AI meeting transcript semantic embeddings"},
                )
                logger.info(f"ChromaDB collection '{self.COLLECTION_NAME}' ready.")
            except ImportError:
                raise VectorStoreError(
                    "chromadb is not installed. Please install it via 'pip install chromadb'."
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to initialize ChromaDB: {e}")
        return self._collection

    def add_chunks(
        self,
        chunks: List[TranscriptChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Insert or update transcript chunks and their BGE embeddings in ChromaDB.

        Args:
            chunks: List of TranscriptChunk objects.
            embeddings: List of embedding vectors matching chunks.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        collection = self._get_client()

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "meeting_id": c.meeting_id,
                "meeting_title": c.meeting_title,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "speakers": ", ".join(c.speakers) if c.speakers else "UNKNOWN",
            }
            for c in chunks
        ]

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Upserted {len(chunks)} chunks into ChromaDB collection '{self.COLLECTION_NAME}'.")
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks into ChromaDB: {e}")

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        meeting_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search using BGE query embedding.

        Args:
            query_embedding: Dense embedding vector of the search query.
            top_k: Number of relevant transcript chunks to return.
            meeting_id: Optional filter to restrict search to a specific meeting.

        Returns:
            List of retrieved chunks with documents, metadatas, and distances.
        """
        collection = self._get_client()

        where_clause = None
        if meeting_id:
            where_clause = {"meeting_id": meeting_id}

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
                ids = results["ids"][0] if "ids" in results else [""] * len(docs)

                for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
                    retrieved.append({
                        "chunk_id": doc_id,
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                    })

            logger.info(f"Retrieved {len(retrieved)} relevant chunks from ChromaDB.")
            return retrieved

        except Exception as e:
            raise VectorStoreError(f"ChromaDB similarity query failed: {e}")

    def delete_meeting_chunks(self, meeting_id: str) -> None:
        """Remove all chunks associated with a specific meeting."""
        collection = self._get_client()
        try:
            collection.delete(where={"meeting_id": meeting_id})
            logger.info(f"Deleted chunks for meeting {meeting_id} from ChromaDB.")
        except Exception as e:
            logger.warning(f"Failed to delete meeting chunks from ChromaDB: {e}")

