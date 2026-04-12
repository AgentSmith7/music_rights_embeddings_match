"""Document classifier using embedding similarity and aggregation."""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from ..vectordb.qdrant_store import QdrantStore, SearchResult
from ..embeddings.embedding_service import EmbeddingService, EmbeddingResult
from ..parsers.base_parser import ParsedDocument


@dataclass
class ClassificationResult:
    """Result of document classification."""
    predicted_class: str
    confidence: float
    class_scores: Dict[str, float]
    supporting_evidence: List[SearchResult]
    needs_review: bool = False
    abstain_reason: Optional[str] = None


class Classifier:
    """Embedding-based document classifier with aggregation."""
    
    def __init__(
        self,
        vector_store: QdrantStore,
        embedding_service: EmbeddingService,
        top_k_retrieval: int = 50,
        top_k_aggregation: int = 10,
        weight_max_sim: float = 0.5,
        weight_avg_sim: float = 0.3,
        weight_count: float = 0.2,
        min_similarity_threshold: float = 0.65,
        min_margin_threshold: float = 0.1
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k_retrieval = top_k_retrieval
        self.top_k_aggregation = top_k_aggregation
        self.weight_max_sim = weight_max_sim
        self.weight_avg_sim = weight_avg_sim
        self.weight_count = weight_count
        self.min_similarity_threshold = min_similarity_threshold
        self.min_margin_threshold = min_margin_threshold
    
    def classify(self, document: ParsedDocument) -> ClassificationResult:
        """Classify a document using embedding similarity."""
        if not document.representations:
            return ClassificationResult(
                predicted_class="unknown",
                confidence=0.0,
                class_scores={},
                supporting_evidence=[],
                needs_review=True,
                abstain_reason="No representations extracted from document"
            )
        
        all_results = []
        
        for rep in document.representations:
            embedding = self.embedding_service.embed_text(rep.text)
            
            results = self.vector_store.search(
                query_vector=embedding.vector,
                top_k=self.top_k_retrieval
            )
            
            for r in results:
                r.payload.representation_type = rep.representation_type.value
            
            all_results.extend(results)
        
        if not all_results:
            return ClassificationResult(
                predicted_class="unknown",
                confidence=0.0,
                class_scores={},
                supporting_evidence=[],
                needs_review=True,
                abstain_reason="No similar documents found"
            )
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:self.top_k_aggregation]
        
        class_scores = self._aggregate_scores(top_results)
        
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_classes:
            return ClassificationResult(
                predicted_class="unknown",
                confidence=0.0,
                class_scores={},
                supporting_evidence=top_results,
                needs_review=True,
                abstain_reason="No class scores computed"
            )
        
        best_class, best_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        
        max_sim = max(r.score for r in top_results) if top_results else 0.0
        margin = best_score - second_score
        
        needs_review = False
        abstain_reason = None
        
        if max_sim < self.min_similarity_threshold:
            needs_review = True
            abstain_reason = f"Max similarity {max_sim:.3f} below threshold {self.min_similarity_threshold}"
        elif margin < self.min_margin_threshold:
            needs_review = True
            abstain_reason = f"Margin {margin:.3f} between top classes below threshold {self.min_margin_threshold}"
        
        return ClassificationResult(
            predicted_class=best_class if not needs_review else "needs_review",
            confidence=best_score,
            class_scores=dict(sorted_classes),
            supporting_evidence=top_results[:5],
            needs_review=needs_review,
            abstain_reason=abstain_reason
        )
    
    def _aggregate_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        """Aggregate search results into class scores."""
        class_results = defaultdict(list)
        
        for r in results:
            class_results[r.payload.class_label].append(r.score)
        
        class_scores = {}
        k = len(results)
        
        for class_label, scores in class_results.items():
            max_sim = max(scores)
            avg_sim = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
            count_ratio = len(scores) / k
            
            class_scores[class_label] = (
                self.weight_max_sim * max_sim +
                self.weight_avg_sim * avg_sim +
                self.weight_count * count_ratio
            )
        
        return class_scores
    
    def classify_batch(self, documents: List[ParsedDocument]) -> List[ClassificationResult]:
        """Classify multiple documents."""
        return [self.classify(doc) for doc in documents]
