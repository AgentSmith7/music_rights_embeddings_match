"""Configuration settings for the music rights classifier."""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    dimension: int = 1024


@dataclass
class ChunkingConfig:
    strategy: str = "semantic"
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 100


@dataclass
class QdrantConfig:
    path: str = "/workspace/qdrant_data"
    collection_name: str = "music_rights_documents"
    distance: str = "Cosine"


@dataclass
class ClassificationConfig:
    top_k_retrieval: int = 50
    top_k_rerank: int = 20
    top_k_aggregation: int = 10
    weight_max_sim: float = 0.5
    weight_avg_sim: float = 0.3
    weight_count: float = 0.2
    min_similarity_threshold: float = 0.65
    min_margin_threshold: float = 0.1


@dataclass
class DataConfig:
    training_archive: str = "gdrive:Music_rights_train/Copy of trainData.tgz"
    local_extract_path: str = "/workspace/training_data"
    holdout_path: str = "/workspace/holdout_data"
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf", ".csv", ".txt", ".xlsx", ".xls"])


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    data: DataConfig = field(default_factory=DataConfig)


def get_config() -> Config:
    return Config()
