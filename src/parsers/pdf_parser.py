"""PDF document parser with chunking support."""
import io
from typing import List
from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from .base_parser import BaseParser, ParsedDocument, DocumentRepresentation, RepresentationType


class PDFParser(BaseParser):
    """Parser for PDF documents with multi-representation extraction."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF parsing. Install with: pip install pymupdf")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.pdf']
    
    def parse(self, file_bytes: bytes, file_name: str) -> ParsedDocument:
        """Parse PDF into multiple representations."""
        representations = []
        
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"
            
            doc.close()
            
            full_text = full_text.strip()
            
            if not full_text:
                logger.warning(f"No text extracted from PDF: {file_name}")
                return ParsedDocument(
                    file_name=file_name,
                    file_type="pdf",
                    representations=[],
                    raw_text="",
                    metadata={"error": "No text extracted"}
                )
            
            # 1. Summary representation
            summary_text = self._create_summary(full_text)
            representations.append(DocumentRepresentation(
                text=summary_text,
                representation_type=RepresentationType.SUMMARY,
                metadata={"total_pages": len(page_texts), "total_chars": len(full_text)}
            ))
            
            # 2. Chunk representations (page-based for now)
            chunks = self._create_chunks(full_text, page_texts)
            for i, chunk in enumerate(chunks):
                representations.append(DocumentRepresentation(
                    text=chunk["text"],
                    representation_type=RepresentationType.CHUNK,
                    metadata={
                        "chunk_id": i,
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end")
                    }
                ))
            
            # 3. Keywords representation
            keywords = self._extract_keywords(full_text)
            if keywords:
                representations.append(DocumentRepresentation(
                    text=keywords,
                    representation_type=RepresentationType.KEYWORDS,
                    metadata={}
                ))
            
            return ParsedDocument(
                file_name=file_name,
                file_type="pdf",
                representations=representations,
                raw_text=full_text,
                metadata={"total_pages": len(page_texts)}
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_name}: {e}")
            return ParsedDocument(
                file_name=file_name,
                file_type="pdf",
                representations=[],
                raw_text="",
                metadata={"error": str(e)}
            )
    
    def _create_chunks(self, full_text: str, page_texts: List[str]) -> List[dict]:
        """Create chunks from document text."""
        chunks = []
        
        # Simple token-window chunking
        words = full_text.split()
        
        if len(words) <= self.chunk_size:
            return [{
                "text": full_text,
                "page_start": 0,
                "page_end": len(page_texts) - 1,
                "char_start": 0,
                "char_end": len(full_text)
            }]
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_words) >= self.min_chunk_size or start + self.chunk_size >= len(words):
                chunks.append({
                    "text": chunk_text,
                    "page_start": None,
                    "page_end": None,
                    "char_start": None,
                    "char_end": None,
                    "word_start": start,
                    "word_end": end
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
            if start >= len(words) - self.min_chunk_size:
                break
        
        return chunks
