"""Embedding service using local GPU models."""
import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from loguru import logger


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    model_name: str
    dimension: int


class EmbeddingService:
    """GPU-accelerated embedding service using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        vector = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return EmbeddingResult(
            vector=np.array(vector),
            model_name=self.model_name,
            dimension=self.dimension
        )
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress
        )
        
        return [
            EmbeddingResult(
                vector=np.array(vec),
                model_name=self.model_name,
                dimension=self.dimension
            )
            for vec in vectors
        ]
    
    def get_dimension(self) -> int:
        return self.dimension


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(**kwargs) -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(**kwargs)
    return _embedding_service
