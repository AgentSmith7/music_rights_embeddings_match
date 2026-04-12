"""Base parser interface and data structures."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RepresentationType(Enum):
    SUMMARY = "summary"
    CHUNK = "chunk"
    SCHEMA = "schema"
    CONTENT = "content"
    KEYWORDS = "keywords"


@dataclass
class DocumentRepresentation:
    """A single representation of a document for embedding."""
    text: str
    representation_type: RepresentationType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Parsed document with multiple representations."""
    file_name: str
    file_type: str
    representations: List[DocumentRepresentation]
    raw_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_bytes: bytes, file_name: str) -> ParsedDocument:
        """Parse file bytes into semantic representations."""
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if parser supports this file type."""
        pass
    
    def _create_summary(self, text: str, max_chars: int = 2000) -> str:
        """Create a summary representation from text."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    def _extract_keywords(self, text: str, top_n: int = 20) -> str:
        """Extract keywords from text (simple frequency-based)."""
        import re
        from collections import Counter
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they',
                     'this', 'that', 'with', 'from', 'will', 'would', 'there', 'their', 'what',
                     'about', 'which', 'when', 'make', 'like', 'time', 'just', 'know', 'take',
                     'into', 'year', 'your', 'some', 'could', 'them', 'than', 'then', 'now',
                     'look', 'only', 'come', 'over', 'such', 'also', 'back', 'after', 'use',
                     'two', 'how', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
                     'any', 'these', 'give', 'day', 'most', 'cant', 'dont', 'does', 'didnt'}
        
        filtered = [w for w in words if w not in stopwords]
        common = Counter(filtered).most_common(top_n)
        return " ".join([word for word, _ in common])
