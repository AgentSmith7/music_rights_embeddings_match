#!/usr/bin/env python3
"""
Index documents to Qdrant vector database.

This script:
1. Loads documents from extracted training data
2. Parses each document (CSV, PDF, etc.)
3. Generates embeddings using BGE-large
4. Stores embeddings in Qdrant with metadata
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    OptimizersConfigDiff, HnswConfigDiff
)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Constants
EXTRACT_PATH = "/workspace/training_data/extracted"
STATEMENT_PATH = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
QDRANT_PATH = "/workspace/qdrant_data"
RESULTS_PATH = "/workspace/results"
COLLECTION_NAME = "music_rights_documents"

# Embedding config
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
BATCH_SIZE = 64


@dataclass
class DocumentPayload:
    """Payload stored with each vector in Qdrant."""
    class_label: str
    file_name: str
    file_path: str
    file_type: str
    representation_type: str  # "schema", "content", "summary"
    text_preview: str
    chunk_id: Optional[int] = None


def parse_csv(filepath: str) -> List[Tuple[str, str]]:
    """
    Parse CSV file and return multiple representations.
    Returns list of (representation_type, text) tuples.
    """
    representations = []
    
    try:
        # Try to read the CSV
        df = pd.read_csv(filepath, nrows=100, encoding='utf-8', on_bad_lines='skip')
        
        if df.empty:
            return representations
        
        # 1. Schema representation: column names and types
        schema_parts = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            sample = sample[:50] if len(sample) > 50 else sample
            schema_parts.append(f"{col} ({dtype}): {sample}")
        
        schema_text = "CSV Schema:\n" + "\n".join(schema_parts[:30])  # Limit columns
        representations.append(("schema", schema_text))
        
        # 2. Content representation: sample rows as natural language
        content_parts = []
        for idx, row in df.head(5).iterrows():
            row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            content_parts.append(row_text[:500])  # Limit row length
        
        content_text = "Sample data:\n" + "\n".join(content_parts)
        representations.append(("content", content_text))
        
        # 3. Summary: key statistics
        summary_parts = [
            f"CSV file with {len(df.columns)} columns and approximately {len(df)} rows.",
            f"Columns: {', '.join(df.columns[:15])}",
        ]
        
        # Add numeric column stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        for col in numeric_cols:
            summary_parts.append(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
        
        summary_text = " ".join(summary_parts)
        representations.append(("summary", summary_text))
        
    except Exception as e:
        logger.warning(f"Error parsing CSV {filepath}: {e}")
    
    return representations


def parse_text(filepath: str) -> List[Tuple[str, str]]:
    """Parse text file and return representations."""
    representations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(10000)  # First 10KB
        
        if content.strip():
            representations.append(("content", content[:2000]))
            
            # Summary: first 500 chars
            summary = content[:500].replace('\n', ' ').strip()
            representations.append(("summary", summary))
    
    except Exception as e:
        logger.warning(f"Error parsing text {filepath}: {e}")
    
    return representations


def parse_document(filepath: str) -> List[Tuple[str, str]]:
    """Parse document and return representations based on file type."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.csv']:
        return parse_csv(filepath)
    elif ext in ['.txt', '.text']:
        return parse_text(filepath)
    elif ext in ['.xlsx', '.xls']:
        # Try to read Excel as CSV
        try:
            df = pd.read_excel(filepath, nrows=100)
            # Save as temp CSV and parse
            temp_path = filepath + '.temp.csv'
            df.to_csv(temp_path, index=False)
            result = parse_csv(temp_path)
            os.remove(temp_path)
            return result
        except Exception as e:
            logger.warning(f"Error parsing Excel {filepath}: {e}")
            return []
    else:
        # Unknown type - try as text
        return parse_text(filepath)


def setup_qdrant() -> QdrantClient:
    """Initialize Qdrant client and collection."""
    logger.info(f"Setting up Qdrant at {QDRANT_PATH}")
    
    client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection exists with {info.points_count} points")
        return client
    
    # Create collection
    logger.info(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        )
    )
    
    return client


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    logger.info(f"Model loaded, dimension: {model.get_sentence_embedding_dimension()}")
    return model


def index_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    max_classes: Optional[int] = None,
    max_files_per_class: Optional[int] = None
):
    """Index all documents to Qdrant."""
    
    if not os.path.exists(STATEMENT_PATH):
        raise ValueError(f"Statement path not found: {STATEMENT_PATH}")
    
    class_folders = sorted(os.listdir(STATEMENT_PATH))
    if max_classes:
        class_folders = class_folders[:max_classes]
    
    logger.info(f"Processing {len(class_folders)} classes")
    
    total_indexed = 0
    stats = defaultdict(int)
    
    # Batch for efficiency
    batch_texts = []
    batch_payloads = []
    
    for class_name in tqdm(class_folders, desc="Classes"):
        class_path = os.path.join(STATEMENT_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        
        files = os.listdir(class_path)
        if max_files_per_class:
            files = files[:max_files_per_class]
        
        for filename in files:
            filepath = os.path.join(class_path, filename)
            if not os.path.isfile(filepath):
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            
            # Parse document
            representations = parse_document(filepath)
            
            if not representations:
                stats['parse_failed'] += 1
                continue
            
            for rep_type, text in representations:
                if not text or len(text.strip()) < 10:
                    continue
                
                payload = DocumentPayload(
                    class_label=class_name,
                    file_name=filename,
                    file_path=filepath,
                    file_type=ext,
                    representation_type=rep_type,
                    text_preview=text[:200]
                )
                
                batch_texts.append(text)
                batch_payloads.append(payload)
                
                # Process batch when full
                if len(batch_texts) >= BATCH_SIZE:
                    _index_batch(client, model, batch_texts, batch_payloads)
                    total_indexed += len(batch_texts)
                    stats['indexed'] += len(batch_texts)
                    batch_texts = []
                    batch_payloads = []
        
        stats['classes_processed'] += 1
    
    # Process remaining batch
    if batch_texts:
        _index_batch(client, model, batch_texts, batch_payloads)
        total_indexed += len(batch_texts)
        stats['indexed'] += len(batch_texts)
    
    logger.info(f"Indexing complete: {total_indexed} vectors indexed")
    logger.info(f"Stats: {dict(stats)}")
    
    return stats


def _index_batch(
    client: QdrantClient,
    model: SentenceTransformer,
    texts: List[str],
    payloads: List[DocumentPayload]
):
    """Index a batch of documents."""
    # Generate embeddings
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    
    # Create points
    points = []
    for embedding, payload in zip(embeddings, payloads):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=asdict(payload)
        ))
    
    # Upsert to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def main():
    """Main indexing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-classes', type=int, default=None, help='Limit number of classes')
    parser.add_argument('--max-files', type=int, default=None, help='Limit files per class')
    parser.add_argument('--test', action='store_true', help='Test mode: 5 classes, 10 files each')
    args = parser.parse_args()
    
    if args.test:
        args.max_classes = 5
        args.max_files = 10
    
    logger.info("="*60)
    logger.info("DOCUMENT INDEXING PIPELINE")
    logger.info("="*60)
    
    # Setup
    client = setup_qdrant()
    model = load_embedding_model()
    
    # Index
    start = time.time()
    stats = index_documents(
        client, model,
        max_classes=args.max_classes,
        max_files_per_class=args.max_files
    )
    elapsed = time.time() - start
    
    # Report
    logger.info(f"\nIndexing completed in {elapsed/60:.1f} minutes")
    
    # Save stats
    os.makedirs(RESULTS_PATH, exist_ok=True)
    stats_path = os.path.join(RESULTS_PATH, 'indexing_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dict(stats), f, indent=2)
    logger.info(f"Stats saved to {stats_path}")
    
    # Collection info
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Collection now has {info.points_count} points")


if __name__ == "__main__":
    main()
