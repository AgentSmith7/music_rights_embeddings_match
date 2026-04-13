#!/usr/bin/env python3
"""
3-Stage Bounded Backpressured Pipeline for Music Rights Document Classification

Architecture:
    ENUMERATOR
        ↓
    PATH QUEUE (bounded)
        ↓
    PARSER PROCESS POOL (CPU-bound: pandas, fitz, xlrd)
        ↓
    PARSED DOC QUEUE (bounded, ~256)
        ↓
    SINGLE GPU BATCH EMBEDDER (batch by char budget)
        ↓
    UPSERT QUEUE (bounded, ~64)
        ↓
    BATCH WRITER + CHECKPOINTER

Key design:
- Process-based parsing (CPU-bound work bypasses GIL)
- Bounded queues for backpressure and memory stability
- Batch by char/token budget, not just count
- Checkpoint for resumability
- Compact representations (fingerprints, not raw dumps)
"""

import os
import sys
import json
import hashlib
import argparse
import time
import signal
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import queue
import threading
import random

import numpy as np
from tqdm import tqdm
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Paths
    train_data_path: str = "/workspace/training_data/music_rights_train/Statement"
    qdrant_path: str = "/workspace/qdrant_data"
    results_path: str = "/workspace/pipeline_results.json"
    checkpoint_path: str = "/workspace/checkpoint.jsonl"
    failures_path: str = "/workspace/failures.jsonl"
    
    # Model
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    
    # Batching - by budget, not just count
    batch_size: int = 64           # Max items per GPU batch
    max_batch_chars: int = 120000  # Max total chars per batch (~30k tokens)
    batch_timeout: float = 2.0     # Flush batch after N seconds
    
    # Representation limits (compact fingerprints)
    max_text_length: int = 4096    # Max chars per doc representation
    max_rows_sample: int = 50      # Max rows from tabular
    max_cols_sample: int = 30      # Max columns
    
    # Queue sizes (bounded for backpressure)
    path_queue_size: int = 512
    parsed_queue_size: int = 256
    upsert_queue_size: int = 128
    
    # Workers
    num_parser_workers: int = 8
    upsert_batch_size: int = 500   # Points per Qdrant upsert
    
    # Qdrant
    collection_name: str = "music_rights"
    use_quantization: bool = True
    
    # Split
    train_ratio: float = 0.8
    random_seed: int = 42


# =============================================================================
# INTERNAL RECORD SCHEMA
# =============================================================================

@dataclass
class ParsedDoc:
    """Record passed from parser to embedder."""
    doc_id: str
    class_label: str
    file_type: str
    source_path: str
    representation_text: str
    parser_status: str  # "ok" or "error"
    error_message: str = ""
    file_size: int = 0
    char_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class EmbeddedDoc:
    """Record passed from embedder to writer."""
    doc_id: str
    vector: np.ndarray
    payload: dict


# =============================================================================
# STAGE 1: PARSER (Process Pool - CPU bound)
# =============================================================================

def parse_single_file(args: Tuple[str, str, str, int, int, int]) -> dict:
    """
    Parse a single file into compact representation.
    Runs in separate process to bypass GIL for CPU-bound parsing.
    
    Returns dict (not dataclass) for pickling across processes.
    """
    file_path, class_label, ext, max_chars, max_rows, max_cols = args
    
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]
    
    try:
        file_size = os.path.getsize(file_path)
        
        # Build compact representation based on file type
        if ext in ['.csv', '.tsv', '.tab']:
            text = _parse_csv(file_path, max_rows, max_cols, max_chars)
        elif ext == '.xlsx':
            text = _parse_xlsx(file_path, max_rows, max_cols, max_chars)
        elif ext == '.xls':
            text = _parse_xls(file_path, max_rows, max_cols, max_chars)
        elif ext == '.pdf':
            text = _parse_pdf(file_path, max_chars)
        elif ext in ['.txt', '.text']:
            text = _parse_text(file_path, max_chars)
        else:
            text = _parse_text(file_path, max_chars)
        
        return {
            "doc_id": doc_id,
            "class_label": class_label,
            "file_type": ext,
            "source_path": file_path,
            "representation_text": text,
            "parser_status": "ok",
            "error_message": "",
            "file_size": file_size,
            "char_count": len(text)
        }
        
    except Exception as e:
        return {
            "doc_id": doc_id,
            "class_label": class_label,
            "file_type": ext,
            "source_path": file_path,
            "representation_text": "",
            "parser_status": "error",
            "error_message": str(e)[:200],
            "file_size": 0,
            "char_count": 0
        }


def _parse_csv(path: str, max_rows: int, max_cols: int, max_chars: int) -> str:
    """Parse CSV into compact fingerprint: schema + sample rows."""
    import pandas as pd
    
    # Detect delimiter
    with open(path, 'r', errors='ignore') as f:
        sample = f.read(2048)
    
    delimiters = [',', '\t', '|', ';']
    delimiter = max(delimiters, key=lambda d: sample.count(d))
    
    # Read with limits
    df = pd.read_csv(path, sep=delimiter, nrows=max_rows, 
                     usecols=lambda x: True, on_bad_lines='skip',
                     encoding_errors='ignore', low_memory=False)
    
    # Limit columns
    if len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols]
    
    return _tabular_fingerprint(df, "CSV", delimiter, max_chars)


def _parse_xlsx(path: str, max_rows: int, max_cols: int, max_chars: int) -> str:
    """Parse XLSX into compact fingerprint."""
    import pandas as pd
    
    xl = pd.ExcelFile(path, engine='openpyxl')
    sheet_names = xl.sheet_names[:5]  # Max 5 sheets
    
    parts = [f"XLSX with sheets: {', '.join(sheet_names)}"]
    
    # Parse first sheet
    df = pd.read_excel(xl, sheet_name=0, nrows=max_rows)
    if len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols]
    
    parts.append(_tabular_fingerprint(df, "Sheet", None, max_chars - 100))
    
    return "\n".join(parts)[:max_chars]


def _parse_xls(path: str, max_rows: int, max_cols: int, max_chars: int) -> str:
    """Parse legacy XLS into compact fingerprint."""
    import xlrd
    
    wb = xlrd.open_workbook(path, on_demand=True)
    sheet_names = wb.sheet_names()[:5]
    
    parts = [f"XLS with sheets: {', '.join(sheet_names)}"]
    
    # Read first sheet
    sheet = wb.sheet_by_index(0)
    rows_data = []
    for i in range(min(sheet.nrows, max_rows + 1)):
        row = [str(sheet.cell_value(i, j))[:100] for j in range(min(sheet.ncols, max_cols))]
        rows_data.append(row)
    
    if rows_data:
        headers = rows_data[0] if rows_data else []
        parts.append(f"Columns ({len(headers)}): {', '.join(headers[:20])}")
        
        # Sample rows
        for i, row in enumerate(rows_data[1:6], 1):
            parts.append(f"Row {i}: {' | '.join(row[:10])}")
    
    return "\n".join(parts)[:max_chars]


def _parse_pdf(path: str, max_chars: int) -> str:
    """Parse PDF into compact fingerprint: metadata + key text."""
    import fitz
    
    doc = fitz.open(path)
    parts = [f"PDF: {doc.page_count} pages"]
    
    # Extract text from first few pages
    text_parts = []
    for i in range(min(3, doc.page_count)):
        page = doc[i]
        text = page.get_text()[:max_chars // 3]
        if text.strip():
            text_parts.append(f"[Page {i+1}] {text.strip()}")
    
    doc.close()
    
    parts.extend(text_parts)
    return "\n".join(parts)[:max_chars]


def _parse_text(path: str, max_chars: int) -> str:
    """Parse text file into fingerprint."""
    with open(path, 'r', errors='ignore') as f:
        content = f.read(max_chars)
    
    lines = content.split('\n')[:50]
    return f"TEXT ({len(lines)} lines):\n" + "\n".join(lines)


def _tabular_fingerprint(df, file_type: str, delimiter: str, max_chars: int) -> str:
    """Create compact fingerprint for tabular data."""
    parts = []
    
    # Schema
    cols = list(df.columns)
    parts.append(f"{file_type} | {len(df)} rows x {len(cols)} cols")
    if delimiter:
        parts.append(f"Delimiter: '{delimiter}'")
    parts.append(f"Columns: {', '.join(str(c) for c in cols[:20])}")
    
    # Data types
    dtypes = df.dtypes.value_counts().to_dict()
    dtype_str = ", ".join(f"{k}: {v}" for k, v in list(dtypes.items())[:5])
    parts.append(f"Types: {dtype_str}")
    
    # Sample rows (first 5)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        row_str = " | ".join(str(v)[:50] for v in row.values[:10])
        parts.append(f"Row {i+1}: {row_str}")
    
    return "\n".join(parts)[:max_chars]


# =============================================================================
# STAGE 2: GPU EMBEDDER (Single process/thread)
# =============================================================================

class GPUEmbedder:
    """
    Single GPU embedding worker.
    Batches by char budget, not just count.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.device = None
    
    def load_model(self):
        """Load model to GPU (lazy)."""
        if self.model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        
        from sentence_transformers import SentenceTransformer
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.config.embedding_model, device=self.device)
        
        logger.info(f"Model loaded on {self.device}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts, return float16 vectors."""
        self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings.astype(np.float16)
    
    def create_batches(self, docs: List[dict]) -> List[List[dict]]:
        """
        Create batches by char budget.
        
        Flush when:
        - batch_size >= max items
        - total_chars >= max_batch_chars
        """
        batches = []
        current_batch = []
        current_chars = 0
        
        for doc in docs:
            char_count = doc.get("char_count", len(doc.get("representation_text", "")))
            
            # Check if adding this doc exceeds budget
            if current_batch and (
                len(current_batch) >= self.config.batch_size or
                current_chars + char_count > self.config.max_batch_chars
            ):
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            
            current_batch.append(doc)
            current_chars += char_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches


# =============================================================================
# STAGE 3: QDRANT WRITER (Batched + Checkpointed)
# =============================================================================

class QdrantWriter:
    """
    Batched Qdrant writer with checkpointing.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = None
        self.checkpoint_file = None
        self.failures_file = None
        self.points_written = 0
    
    def initialize(self):
        """Initialize Qdrant and checkpoint files."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            VectorParams, Distance, 
            ScalarQuantization, ScalarQuantizationConfig, ScalarType
        )
        
        logger.info(f"Initializing Qdrant at {self.config.qdrant_path}")
        
        self.client = QdrantClient(path=self.config.qdrant_path)
        
        # Check if collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.config.collection_name in collections:
            logger.info(f"Collection '{self.config.collection_name}' exists")
        else:
            # Create with quantization
            quantization = None
            if self.config.use_quantization:
                quantization = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=True
                    )
                )
            
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE
                ),
                quantization_config=quantization
            )
            logger.info(f"Created collection '{self.config.collection_name}'")
        
        # Open checkpoint files
        self.checkpoint_file = open(self.config.checkpoint_path, 'a')
        self.failures_file = open(self.config.failures_path, 'a')
    
    def write_batch(self, embedded_docs: List[Tuple[str, np.ndarray, dict]]):
        """Write a batch of embedded documents to Qdrant."""
        from qdrant_client.models import PointStruct
        
        points = []
        for doc_id, vector, payload in embedded_docs:
            point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            ))
        
        # Batch upsert with retry
        for attempt in range(3):
            try:
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True
                )
                self.points_written += len(points)
                
                # Checkpoint
                for doc_id, _, payload in embedded_docs:
                    self.checkpoint_file.write(json.dumps({
                        "doc_id": doc_id,
                        "path": payload.get("source_path", ""),
                        "timestamp": datetime.now().isoformat()
                    }) + "\n")
                self.checkpoint_file.flush()
                
                return
                
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    # Log failures
                    for doc_id, _, payload in embedded_docs:
                        self.failures_file.write(json.dumps({
                            "doc_id": doc_id,
                            "path": payload.get("source_path", ""),
                            "error": str(e)[:200]
                        }) + "\n")
                    self.failures_file.flush()
                    logger.warning(f"Failed to write batch after 3 attempts: {e}")
    
    def close(self):
        """Close files."""
        if self.checkpoint_file:
            self.checkpoint_file.close()
        if self.failures_file:
            self.failures_file.close()
    
    def get_completed_ids(self) -> set:
        """Load completed doc IDs from checkpoint for resume."""
        completed = set()
        if os.path.exists(self.config.checkpoint_path):
            with open(self.config.checkpoint_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed.add(rec.get("doc_id", ""))
                    except:
                        pass
        return completed


# =============================================================================
# FILE ENUMERATION + SPLIT
# =============================================================================

def enumerate_files(base_path: str) -> List[Tuple[str, str, str]]:
    """
    Enumerate all files with (path, class_label, extension).
    Class label = parent folder under Statement/.
    """
    files = []
    base = Path(base_path)
    
    for root, _, filenames in os.walk(base):
        for fname in filenames:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            
            # Skip non-data files
            if ext not in ['.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab']:
                continue
            
            # Extract class label from path
            parts = fpath.parts
            try:
                stmt_idx = parts.index('Statement')
                class_label = parts[stmt_idx + 1]
            except (ValueError, IndexError):
                class_label = "unknown"
            
            files.append((str(fpath), class_label, ext))
    
    return files


def stratified_split(files: List[Tuple[str, str, str]], 
                     train_ratio: float = 0.8, 
                     seed: int = 42) -> Tuple[List, List]:
    """Stratified train/val split by class."""
    random.seed(seed)
    
    # Group by class
    by_class = defaultdict(list)
    for f in files:
        by_class[f[1]].append(f)
    
    train_files = []
    val_files = []
    
    for class_label, class_files in by_class.items():
        random.shuffle(class_files)
        split_idx = int(len(class_files) * train_ratio)
        train_files.extend(class_files[:split_idx])
        val_files.extend(class_files[split_idx:])
    
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    return train_files, val_files


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

class Pipeline:
    """
    3-stage bounded backpressured pipeline.
    
    Stage 1: Parser process pool (CPU-bound)
    Stage 2: GPU embedder (single, batches by char budget)
    Stage 3: Qdrant writer (batched, checkpointed)
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embedder = GPUEmbedder(config)
        self.writer = QdrantWriter(config)
        
        # Stats
        self.stats = {
            "total_files": 0,
            "parsed_ok": 0,
            "parsed_error": 0,
            "embedded": 0,
            "written": 0,
            "skipped_checkpoint": 0
        }
    
    def run_indexing(self, files: List[Tuple[str, str, str]], split_name: str = "train"):
        """
        Run the 3-stage pipeline for indexing.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        self.stats["total_files"] = len(files)
        logger.info(f"Starting {split_name} indexing: {len(files)} files")
        
        # Initialize writer (creates collection)
        self.writer.initialize()
        
        # Load completed IDs for resume
        completed_ids = self.writer.get_completed_ids()
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} already completed")
            self.stats["skipped_checkpoint"] = len(completed_ids)
        
        # Pre-compute doc IDs to skip
        skip_ids = set()
        files_to_process = []
        for fpath, class_label, ext in files:
            doc_id = hashlib.md5(fpath.encode()).hexdigest()[:16]
            if doc_id in completed_ids:
                skip_ids.add(doc_id)
            else:
                files_to_process.append((fpath, class_label, ext))
        
        logger.info(f"Files to process: {len(files_to_process)} (skipping {len(skip_ids)})")
        
        # Prepare parser args
        parser_args = [
            (fpath, class_label, ext, 
             self.config.max_text_length, 
             self.config.max_rows_sample,
             self.config.max_cols_sample)
            for fpath, class_label, ext in files_to_process
        ]
        
        # Bounded queues
        parsed_queue = queue.Queue(maxsize=self.config.parsed_queue_size)
        upsert_queue = queue.Queue(maxsize=self.config.upsert_queue_size)
        done_parsing = threading.Event()
        done_embedding = threading.Event()
        
        # STAGE 1: Parser process pool
        def parser_stage():
            with ProcessPoolExecutor(max_workers=self.config.num_parser_workers) as executor:
                futures = {executor.submit(parse_single_file, args): args for args in parser_args}
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        if result["parser_status"] == "ok":
                            self.stats["parsed_ok"] += 1
                            parsed_queue.put(result)  # Blocks if queue full (backpressure)
                        else:
                            self.stats["parsed_error"] += 1
                    except Exception as e:
                        self.stats["parsed_error"] += 1
                        logger.warning(f"Parser future error: {e}")
            
            done_parsing.set()
        
        # STAGE 2: GPU embedder
        def embedder_stage():
            batch = []
            batch_chars = 0
            last_flush = time.time()
            
            while True:
                try:
                    # Non-blocking get with timeout for batch flushing
                    doc = parsed_queue.get(timeout=self.config.batch_timeout)
                    batch.append(doc)
                    batch_chars += doc.get("char_count", 0)
                    
                    # Flush if batch full or char budget exceeded
                    should_flush = (
                        len(batch) >= self.config.batch_size or
                        batch_chars >= self.config.max_batch_chars
                    )
                    
                    if should_flush:
                        self._embed_and_queue(batch, upsert_queue, split_name)
                        batch = []
                        batch_chars = 0
                        last_flush = time.time()
                        
                except queue.Empty:
                    # Timeout - flush if we have items and timeout exceeded
                    if batch and (time.time() - last_flush) >= self.config.batch_timeout:
                        self._embed_and_queue(batch, upsert_queue, split_name)
                        batch = []
                        batch_chars = 0
                        last_flush = time.time()
                    
                    # Check if parsing is done
                    if done_parsing.is_set() and parsed_queue.empty():
                        break
            
            # Final flush
            if batch:
                self._embed_and_queue(batch, upsert_queue, split_name)
            
            done_embedding.set()
        
        # STAGE 3: Writer
        def writer_stage():
            upsert_batch = []
            
            while True:
                try:
                    item = upsert_queue.get(timeout=1.0)
                    upsert_batch.append(item)
                    
                    if len(upsert_batch) >= self.config.upsert_batch_size:
                        self.writer.write_batch(upsert_batch)
                        self.stats["written"] += len(upsert_batch)
                        upsert_batch = []
                        
                except queue.Empty:
                    if done_embedding.is_set() and upsert_queue.empty():
                        break
            
            # Final flush
            if upsert_batch:
                self.writer.write_batch(upsert_batch)
                self.stats["written"] += len(upsert_batch)
        
        # Start stages as threads (parser uses ProcessPoolExecutor internally)
        parser_thread = threading.Thread(target=parser_stage, daemon=True)
        embedder_thread = threading.Thread(target=embedder_stage, daemon=True)
        writer_thread = threading.Thread(target=writer_stage, daemon=True)
        
        # Progress bar
        pbar = tqdm(total=len(files_to_process), desc=f"Indexing {split_name}")
        
        parser_thread.start()
        embedder_thread.start()
        writer_thread.start()
        
        # Monitor progress
        last_written = 0
        while writer_thread.is_alive() or not done_embedding.is_set():
            time.sleep(1.0)
            current = self.stats["written"]
            if current > last_written:
                pbar.update(current - last_written)
                last_written = current
        
        # Wait for completion
        parser_thread.join(timeout=30)
        embedder_thread.join(timeout=30)
        writer_thread.join(timeout=30)
        
        pbar.close()
        self.writer.close()
        
        logger.info(f"Indexing complete: {self.stats}")
        return self.stats
    
    def _embed_and_queue(self, batch: List[dict], upsert_queue: queue.Queue, split_name: str):
        """Embed a batch and put results in upsert queue."""
        texts = [doc["representation_text"] for doc in batch]
        
        try:
            vectors = self.embedder.embed_batch(texts)
            self.stats["embedded"] += len(batch)
            
            for doc, vector in zip(batch, vectors):
                payload = {
                    "class_label": doc["class_label"],
                    "source_path": doc["source_path"],
                    "file_type": doc["file_type"],
                    "split": split_name,
                    "file_size": doc["file_size"]
                }
                upsert_queue.put((doc["doc_id"], vector, payload))
                
        except Exception as e:
            logger.warning(f"Embedding batch failed: {e}")
            # Retry with smaller sub-batches
            if len(batch) > 1:
                mid = len(batch) // 2
                self._embed_and_queue(batch[:mid], upsert_queue, split_name)
                self._embed_and_queue(batch[mid:], upsert_queue, split_name)
    
    def _get_qdrant_client(self):
        """Get or create Qdrant client for inference."""
        if not hasattr(self, '_inference_client') or self._inference_client is None:
            from qdrant_client import QdrantClient
            self._inference_client = QdrantClient(path=self.config.qdrant_path)
        return self._inference_client
    
    def classify(self, file_path: str, top_k: int = 10) -> dict:
        """Classify a single file."""
        ext = Path(file_path).suffix.lower()
        
        # Parse
        result = parse_single_file((
            file_path, "unknown", ext,
            self.config.max_text_length,
            self.config.max_rows_sample,
            self.config.max_cols_sample
        ))
        
        if result["parser_status"] != "ok":
            return {"error": result["error_message"], "predictions": []}
        
        # Embed
        vector = self.embedder.embed_batch([result["representation_text"]])[0]
        
        # Search using shared client
        client = self._get_qdrant_client()
        
        search_result = client.query_points(
            collection_name=self.config.collection_name,
            query=vector.tolist(),
            limit=top_k
        )
        results = search_result.points
        
        # Aggregate by class
        class_scores = defaultdict(list)
        for r in results:
            class_scores[r.payload["class_label"]].append(r.score)
        
        # Rank by mean score
        predictions = [
            {"class": cls, "score": float(np.mean(scores)), "count": len(scores)}
            for cls, scores in class_scores.items()
        ]
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "file": file_path,
            "predictions": predictions[:5],
            "top_class": predictions[0]["class"] if predictions else None
        }
    
    def evaluate(self, val_files: List[Tuple[str, str, str]]) -> dict:
        """Evaluate on validation set."""
        logger.info(f"Evaluating on {len(val_files)} validation files")
        
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for fpath, true_label, ext in tqdm(val_files, desc="Evaluating"):
            result = self.classify(fpath)
            
            if result.get("top_class"):
                total += 1
                class_total[true_label] += 1
                
                if result["top_class"] == true_label:
                    correct += 1
                    class_correct[true_label] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Per-class accuracy
        per_class = {}
        for cls in class_total:
            per_class[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_class_accuracy": per_class,
            "macro_accuracy": np.mean(list(per_class.values())) if per_class else 0
        }
        
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="3-Stage Document Classification Pipeline")
    parser.add_argument("--action", choices=["index", "evaluate", "both"], default="both")
    parser.add_argument("--data-path", default="/workspace/training_data/music_rights_train/Statement")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    config = PipelineConfig(
        train_data_path=args.data_path,
        batch_size=args.batch_size,
        num_parser_workers=args.num_workers
    )
    
    # Enumerate and split
    logger.info("Enumerating files...")
    all_files = enumerate_files(config.train_data_path)
    logger.info(f"Found {len(all_files)} files")
    
    train_files, val_files = stratified_split(all_files, config.train_ratio, config.random_seed)
    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val")
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    if args.action in ["index", "both"]:
        stats = pipeline.run_indexing(train_files, "train")
        
        # Save stats
        with open(config.results_path, 'w') as f:
            json.dump({"indexing_stats": stats}, f, indent=2)
    
    if args.action in ["evaluate", "both"]:
        eval_results = pipeline.evaluate(val_files)
        
        # Update results
        results = {}
        if os.path.exists(config.results_path):
            with open(config.results_path, 'r') as f:
                results = json.load(f)
        
        results["evaluation"] = eval_results
        
        with open(config.results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {config.results_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA in subprocesses
    main()
