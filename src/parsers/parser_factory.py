"""Parser factory for selecting appropriate document parser."""
from pathlib import Path
from typing import Optional
from loguru import logger

from .base_parser import BaseParser, ParsedDocument, DocumentRepresentation, RepresentationType
from .pdf_parser import PDFParser
from .csv_parser import CSVParser


class TextParser(BaseParser):
    """Simple parser for plain text files."""
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.txt', '.text', '.md', '.markdown']
    
    def parse(self, file_bytes: bytes, file_name: str) -> ParsedDocument:
        """Parse text file into representations."""
        try:
            text = file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            text = file_bytes.decode('latin-1', errors='ignore')
        
        representations = []
        
        if text.strip():
            representations.append(DocumentRepresentation(
                text=self._create_summary(text),
                representation_type=RepresentationType.SUMMARY,
                metadata={"total_chars": len(text)}
            ))
            
            keywords = self._extract_keywords(text)
            if keywords:
                representations.append(DocumentRepresentation(
                    text=keywords,
                    representation_type=RepresentationType.KEYWORDS,
                    metadata={}
                ))
        
        return ParsedDocument(
            file_name=file_name,
            file_type="txt",
            representations=representations,
            raw_text=text
        )


_parsers = [
    PDFParser(),
    CSVParser(),
    TextParser(),
]


def get_parser(file_name: str) -> Optional[BaseParser]:
    """Get appropriate parser for a file based on extension."""
    ext = Path(file_name).suffix.lower()
    
    for parser in _parsers:
        if parser.supports(ext):
            return parser
    
    logger.warning(f"No parser found for file type: {ext} ({file_name})")
    return None


def parse_file(file_bytes: bytes, file_name: str) -> Optional[ParsedDocument]:
    """Parse a file using the appropriate parser."""
    parser = get_parser(file_name)
    if parser is None:
        return None
    
    return parser.parse(file_bytes, file_name)
