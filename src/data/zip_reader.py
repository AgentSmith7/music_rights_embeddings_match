"""In-memory ZIP file reader for inference."""
import zipfile
import io
from typing import Iterator, Tuple, Optional
from pathlib import Path
from loguru import logger


class ZipReader:
    """Read files from ZIP archive without extracting to disk."""
    
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self._zip_file: Optional[zipfile.ZipFile] = None
    
    def __enter__(self):
        self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._zip_file:
            self._zip_file.close()
    
    def iterate_files(
        self,
        extensions: Optional[list] = None
    ) -> Iterator[Tuple[str, bytes]]:
        """
        Iterate through files in ZIP, yielding (path, content) tuples.
        
        Args:
            extensions: List of file extensions to include (e.g., ['.pdf', '.csv'])
        
        Yields:
            Tuple of (internal_path, file_bytes)
        """
        if self._zip_file is None:
            raise RuntimeError("ZipReader must be used as context manager")
        
        for info in self._zip_file.infolist():
            if info.is_dir():
                continue
            
            file_path = info.filename
            ext = Path(file_path).suffix.lower()
            
            if extensions and ext not in extensions:
                continue
            
            try:
                content = self._zip_file.read(file_path)
                yield file_path, content
            except Exception as e:
                logger.error(f"Error reading {file_path} from ZIP: {e}")
                continue
    
    def list_files(self, extensions: Optional[list] = None) -> list:
        """List all files in the ZIP archive."""
        if self._zip_file is None:
            raise RuntimeError("ZipReader must be used as context manager")
        
        files = []
        for info in self._zip_file.infolist():
            if info.is_dir():
                continue
            
            file_path = info.filename
            ext = Path(file_path).suffix.lower()
            
            if extensions and ext not in extensions:
                continue
            
            files.append({
                "path": file_path,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "extension": ext
            })
        
        return files
    
    def read_file(self, internal_path: str) -> bytes:
        """Read a specific file from the ZIP."""
        if self._zip_file is None:
            raise RuntimeError("ZipReader must be used as context manager")
        
        return self._zip_file.read(internal_path)


def read_zip_in_memory(zip_bytes: bytes) -> ZipReader:
    """Create a ZipReader from bytes (for streaming scenarios)."""
    class BytesZipReader(ZipReader):
        def __init__(self, data: bytes):
            self._data = data
            self._zip_file = None
        
        def __enter__(self):
            self._zip_file = zipfile.ZipFile(io.BytesIO(self._data), 'r')
            return self
    
    return BytesZipReader(zip_bytes)
