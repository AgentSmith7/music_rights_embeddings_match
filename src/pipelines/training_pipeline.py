"""Training pipeline for building the vector index."""
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from ..config.settings import Config, get_config
from ..data.training_loader import TrainingDataLoader, TrainingFile
from ..parsers.parser_factory import parse_file
from ..embeddings.embedding_service import EmbeddingService, get_embedding_service
from ..vectordb.qdrant_store import QdrantStore, VectorPayload


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_store: Optional[QdrantStore] = None
        self.data_loader: Optional[TrainingDataLoader] = None
    
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing training pipeline...")
        
        self.embedding_service = get_embedding_service(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size
        )
        
        self.vector_store = QdrantStore(
            path=self.config.qdrant.path,
            collection_name=self.config.qdrant.collection_name,
            dimension=self.embedding_service.get_dimension()
        )
        
        self.data_loader = TrainingDataLoader(
            source_path=self.config.data.local_extract_path,
            supported_extensions=self.config.data.supported_extensions
        )
        
        logger.info("Training pipeline initialized")
    
    def process_file(self, training_file: TrainingFile) -> int:
        """Process a single training file and add to vector store."""
        parsed = parse_file(training_file.file_bytes, training_file.file_name)
        
        if parsed is None or not parsed.representations:
            logger.warning(f"No representations for: {training_file.file_name}")
            return 0
        
        vectors = []
        payloads = []
        
        for rep in parsed.representations:
            embedding = self.embedding_service.embed_text(rep.text)
            
            payload = VectorPayload(
                class_label=training_file.class_label,
                file_name=training_file.file_name,
                file_path=training_file.file_path,
                file_type=parsed.file_type,
                representation_type=rep.representation_type.value,
                chunk_id=rep.metadata.get("chunk_id"),
                text_preview=rep.text[:500] if rep.text else None,
                source="training"
            )
            
            vectors.append(embedding.vector)
            payloads.append(payload)
        
        self.vector_store.upsert(vectors, payloads)
        
        return len(vectors)
    
    def run_from_directory(self, directory: str, max_files: Optional[int] = None) -> dict:
        """Run training from a directory of labeled files."""
        if self.embedding_service is None:
            self.initialize()
        
        logger.info(f"Processing training data from: {directory}")
        
        stats = {
            "total_files": 0,
            "total_vectors": 0,
            "files_by_class": {},
            "errors": []
        }
        
        files_iterator = self.data_loader.iterate_from_directory(directory)
        
        for training_file in tqdm(files_iterator, desc="Processing files"):
            if max_files and stats["total_files"] >= max_files:
                break
            
            try:
                num_vectors = self.process_file(training_file)
                
                stats["total_files"] += 1
                stats["total_vectors"] += num_vectors
                
                label = training_file.class_label
                stats["files_by_class"][label] = stats["files_by_class"].get(label, 0) + 1
                
            except Exception as e:
                logger.error(f"Error processing {training_file.file_name}: {e}")
                stats["errors"].append({
                    "file": training_file.file_name,
                    "error": str(e)
                })
        
        logger.info(f"Training complete: {stats['total_files']} files, {stats['total_vectors']} vectors")
        return stats
    
    def run_from_tar(self, tar_path: str, max_files: Optional[int] = None) -> dict:
        """Run training from a tar archive."""
        if self.embedding_service is None:
            self.initialize()
        
        logger.info(f"Processing training data from tar: {tar_path}")
        
        stats = {
            "total_files": 0,
            "total_vectors": 0,
            "files_by_class": {},
            "errors": []
        }
        
        files_iterator = self.data_loader.iterate_from_tar(tar_path)
        
        for training_file in tqdm(files_iterator, desc="Processing files"):
            if max_files and stats["total_files"] >= max_files:
                break
            
            try:
                num_vectors = self.process_file(training_file)
                
                stats["total_files"] += 1
                stats["total_vectors"] += num_vectors
                
                label = training_file.class_label
                stats["files_by_class"][label] = stats["files_by_class"].get(label, 0) + 1
                
            except Exception as e:
                logger.error(f"Error processing {training_file.file_name}: {e}")
                stats["errors"].append({
                    "file": training_file.file_name,
                    "error": str(e)
                })
        
        logger.info(f"Training complete: {stats['total_files']} files, {stats['total_vectors']} vectors")
        return stats
    
    def get_stats(self) -> dict:
        """Get current vector store statistics."""
        if self.vector_store is None:
            return {}
        
        info = self.vector_store.get_collection_info()
        class_counts = self.vector_store.count_by_class()
        
        return {
            "collection": info,
            "class_distribution": class_counts
        }
