"""BGE Embedding Service for MeetWise AI.

Loads BAAI BGE (Beijing Academy of Artificial Intelligence) dense embedding models
via sentence-transformers with query prefix support.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Exception raised when generating embeddings fails."""
    pass


class BGEEmbeddingService:
    """Encapsulates BAAI BGE embedding model loading and inference."""

    # BGE query instruction as specified by model authors
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading BGE embedding model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("BGE model loaded successfully.")
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Please install it via 'pip install sentence-transformers'."
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load BGE embedding model: {e}")
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute embeddings for a batch of transcript document chunks.

        Note: BGE documents are embedded WITHOUT query prefix.
        """
        if not texts:
            return []

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate document embeddings: {e}")

    def embed_query(self, query: str) -> List[float]:
        """Compute embedding for a search query, adding the BGE query instruction prefix."""
        try:
            model = self._get_model()
            # BGE models require query instruction prefix for optimal dense retrieval
            prefixed_query = f"{self.BGE_QUERY_PREFIX}{query}" if "bge" in self.model_name.lower() else query
            embedding = model.encode(
                prefixed_query,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}")

