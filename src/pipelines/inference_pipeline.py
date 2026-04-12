"""Inference pipeline for classifying new documents."""
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from ..config.settings import Config, get_config
from ..data.zip_reader import ZipReader
from ..parsers.parser_factory import parse_file
from ..embeddings.embedding_service import EmbeddingService, get_embedding_service
from ..vectordb.qdrant_store import QdrantStore
from ..classification.classifier import Classifier, ClassificationResult


class InferencePipeline:
    """End-to-end inference pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_store: Optional[QdrantStore] = None
        self.classifier: Optional[Classifier] = None
    
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing inference pipeline...")
        
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
        
        self.classifier = Classifier(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k_retrieval=self.config.classification.top_k_retrieval,
            top_k_aggregation=self.config.classification.top_k_aggregation,
            weight_max_sim=self.config.classification.weight_max_sim,
            weight_avg_sim=self.config.classification.weight_avg_sim,
            weight_count=self.config.classification.weight_count,
            min_similarity_threshold=self.config.classification.min_similarity_threshold,
            min_margin_threshold=self.config.classification.min_margin_threshold
        )
        
        logger.info("Inference pipeline initialized")
    
    def classify_file(self, file_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Classify a single file."""
        parsed = parse_file(file_bytes, file_name)
        
        if parsed is None:
            return {
                "file_name": file_name,
                "predicted_class": "unsupported",
                "confidence": 0.0,
                "error": "Unsupported file type"
            }
        
        if not parsed.representations:
            return {
                "file_name": file_name,
                "predicted_class": "unparseable",
                "confidence": 0.0,
                "error": "Could not extract content from file"
            }
        
        result = self.classifier.classify(parsed)
        
        return {
            "file_name": file_name,
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "class_scores": result.class_scores,
            "needs_review": result.needs_review,
            "abstain_reason": result.abstain_reason,
            "supporting_evidence": [
                {
                    "file_name": e.payload.file_name,
                    "class_label": e.payload.class_label,
                    "similarity": e.score,
                    "representation_type": e.payload.representation_type
                }
                for e in result.supporting_evidence
            ]
        }
    
    def run_on_zip(
        self,
        zip_path: str,
        output_path: Optional[str] = None,
        extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run inference on all files in a ZIP archive."""
        if self.classifier is None:
            self.initialize()
        
        if extensions is None:
            extensions = self.config.data.supported_extensions
        
        results = []
        
        logger.info(f"Processing ZIP archive: {zip_path}")
        
        with ZipReader(zip_path) as reader:
            files = reader.list_files(extensions=extensions)
            logger.info(f"Found {len(files)} files to process")
            
            for file_path, file_bytes in tqdm(
                reader.iterate_files(extensions=extensions),
                total=len(files),
                desc="Classifying files"
            ):
                try:
                    result = self.classify_file(file_bytes, file_path)
                    result["zip_internal_path"] = file_path
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error classifying {file_path}: {e}")
                    results.append({
                        "zip_internal_path": file_path,
                        "file_name": Path(file_path).name,
                        "predicted_class": "error",
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def run_on_directory(
        self,
        directory: str,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run inference on all files in a directory."""
        if self.classifier is None:
            self.initialize()
        
        results = []
        dir_path = Path(directory)
        
        files = []
        for ext in self.config.data.supported_extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files to process")
        
        for file_path in tqdm(files, desc="Classifying files"):
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                result = self.classify_file(file_bytes, file_path.name)
                result["file_path"] = str(file_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error classifying {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "predicted_class": "error",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        total = len(results)
        
        class_counts = {}
        needs_review_count = 0
        error_count = 0
        confidence_sum = 0
        
        for r in results:
            pred_class = r.get("predicted_class", "unknown")
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            if r.get("needs_review"):
                needs_review_count += 1
            
            if r.get("error"):
                error_count += 1
            
            confidence_sum += r.get("confidence", 0)
        
        return {
            "total_files": total,
            "class_distribution": class_counts,
            "needs_review_count": needs_review_count,
            "error_count": error_count,
            "average_confidence": confidence_sum / total if total > 0 else 0
        }
