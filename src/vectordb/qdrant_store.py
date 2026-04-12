"""Qdrant vector store implementation."""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchParams
)


@dataclass
class VectorPayload:
    """Payload stored with each vector."""
    class_label: str
    file_name: str
    file_path: str
    file_type: str
    representation_type: str
    chunk_id: Optional[int] = None
    text_preview: Optional[str] = None
    source: str = "training"


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    payload: VectorPayload


class QdrantStore:
    """Qdrant vector database wrapper."""
    
    def __init__(
        self,
        path: str = "/workspace/qdrant_data",
        collection_name: str = "music_rights_documents",
        dimension: int = 1024,
        distance: str = "Cosine"
    ):
        self.path = path
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = Distance.COSINE if distance == "Cosine" else Distance.EUCLID
        
        logger.info(f"Initializing Qdrant at {path}")
        self.client = QdrantClient(path=path)
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self.distance
                )
            )
        else:
            logger.info(f"Collection exists: {self.collection_name}")
    
    def upsert(
        self,
        vectors: List[np.ndarray],
        payloads: List[VectorPayload],
        ids: Optional[List[str]] = None
    ) -> int:
        """Insert or update vectors with payloads."""
        if not vectors:
            return 0
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        points = [
            PointStruct(
                id=id_,
                vector=vec.tolist() if isinstance(vec, np.ndarray) else vec,
                payload=asdict(payload)
            )
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.debug(f"Upserted {len(points)} vectors")
        return len(points)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        filter_obj = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            filter_obj = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            limit=top_k,
            query_filter=filter_obj
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=VectorPayload(**r.payload)
            )
            for r in results
        ]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": getattr(info, 'points_count', 0),
            "status": str(info.status)
        }
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def count_by_class(self) -> Dict[str, int]:
        """Count vectors by class label."""
        info = self.client.get_collection(self.collection_name)
        
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        counts = {}
        for point in scroll_result[0]:
            label = point.payload.get("class_label", "unknown")
            counts[label] = counts.get(label, 0) + 1
        
        return counts
