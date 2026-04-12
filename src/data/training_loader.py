"""Training data loader for directory-based labeled data."""
import os
import tarfile
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrainingFile:
    """A training file with its label."""
    file_path: str
    file_name: str
    class_label: str
    file_bytes: bytes


class TrainingDataLoader:
    """Load training data from directory structure or tar archive."""
    
    def __init__(
        self,
        source_path: str,
        supported_extensions: Optional[List[str]] = None
    ):
        self.source_path = source_path
        self.supported_extensions = supported_extensions or ['.pdf', '.csv', '.txt', '.xlsx', '.xls']
    
    def iterate_from_directory(self, directory: str) -> Iterator[TrainingFile]:
        """
        Iterate through training files from a directory structure.
        
        Expected structure:
        directory/
            class_label_1/
                file1.pdf
                file2.csv
            class_label_2/
                file3.pdf
        """
        base_path = Path(directory)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Training directory not found: {directory}")
        
        for class_dir in base_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_label = class_dir.name
            
            if class_label.startswith('.'):
                continue
            
            for file_path in class_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                
                ext = file_path.suffix.lower()
                if ext not in self.supported_extensions:
                    continue
                
                try:
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    yield TrainingFile(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        class_label=class_label,
                        file_bytes=file_bytes
                    )
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue
    
    def iterate_from_tar(self, tar_path: str) -> Iterator[TrainingFile]:
        """
        Iterate through training files from a tar/tgz archive.
        
        Expected structure inside tar:
        trainData/
            class_label_1/
                file1.pdf
            class_label_2/
                file2.csv
        """
        logger.info(f"Opening tar archive: {tar_path}")
        
        with tarfile.open(tar_path, 'r:*') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                
                parts = Path(member.name).parts
                
                if len(parts) < 2:
                    continue
                
                if parts[0] in ['trainData', 'training_data', 'data']:
                    class_label = parts[1]
                    file_name = parts[-1]
                else:
                    class_label = parts[0]
                    file_name = parts[-1]
                
                ext = Path(file_name).suffix.lower()
                if ext not in self.supported_extensions:
                    continue
                
                if class_label.startswith('.') or file_name.startswith('.'):
                    continue
                
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    
                    file_bytes = f.read()
                    
                    yield TrainingFile(
                        file_path=member.name,
                        file_name=file_name,
                        class_label=class_label,
                        file_bytes=file_bytes
                    )
                except Exception as e:
                    logger.error(f"Error extracting {member.name}: {e}")
                    continue
    
    def download_from_rclone(
        self,
        remote_path: str,
        local_path: str,
        extract: bool = True
    ) -> str:
        """
        Download training data from rclone remote.
        
        Args:
            remote_path: rclone path (e.g., 'gdrive:Music_rights_train/trainData.tgz')
            local_path: Local destination path
            extract: Whether to extract tar archive
        
        Returns:
            Path to extracted directory or downloaded file
        """
        logger.info(f"Downloading from {remote_path} to {local_path}")
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        cmd = ['rclone', 'copy', remote_path, os.path.dirname(local_path), '-P']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"rclone download failed: {result.stderr}")
        
        if extract and local_path.endswith(('.tgz', '.tar.gz', '.tar')):
            extract_dir = local_path.rsplit('.', 2)[0]
            logger.info(f"Extracting to {extract_dir}")
            
            with tarfile.open(local_path, 'r:*') as tar:
                tar.extractall(extract_dir)
            
            return extract_dir
        
        return local_path
    
    def count_files_by_class(self, directory: str) -> dict:
        """Count files per class in a directory."""
        counts = {}
        
        for training_file in self.iterate_from_directory(directory):
            label = training_file.class_label
            counts[label] = counts.get(label, 0) + 1
        
        return counts
