# Music Rights Document Classification System
## Implementation Plan

**Version:** 1.1  
**Date:** April 12, 2026  
**Project:** Embedding-based Document Retrieval Classification Pipeline

> **Important:** This is NOT an ML classification system. It uses **embedding-based retrieval** 
> where documents are classified by finding the most similar training documents in vector space 
> and aggregating their class labels. No model training occurs - only embedding generation and indexing.

---

## Executive Summary

This document outlines the design and implementation plan for a production-grade document classification system that uses semantic embeddings and vector search to classify music rights documents (PDFs, CSVs, and other files) into predefined categories.

### Key Constraints
- **Training Data Location:** Google Drive (`gdrive:Music_rights_train/Copy of trainData.tgz`) - 64.5 GB compressed
- **No Local Download:** System must work without fully downloading training data locally
- **Inference:** Process ZIP archives in-memory without extraction

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Strategy](#2-data-strategy)
3. [Module Design](#3-module-design)
4. [Implementation Phases](#4-implementation-phases)
5. [Technical Specifications](#5-technical-specifications)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Deployment Considerations](#7-deployment-considerations)

---

## 1. System Architecture

### 1.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Remote     │    │  Document    │    │  Embedding   │    │  Qdrant   │ │
│  │   Storage    │───▶│  Parser      │───▶│  Generator   │───▶│  Vector   │ │
│  │  (rclone)    │    │              │    │              │    │   Store   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Metadata & Payload Store                          │  │
│  │  • class_label  • file_name  • file_type  • representation_type      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  ZIP Archive │    │  Document    │    │  Embedding   │    │  Qdrant   │ │
│  │  (in-memory) │───▶│  Parser      │───▶│  Generator   │───▶│  Query    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                      │       │
│                                                                      ▼       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Classification Engine                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │
│  │  │ Retrieval  │─▶│  Reranker  │─▶│ Aggregator │─▶│ Class Predictor│  │  │
│  │  │  Stage 1   │  │  Stage 2   │  │            │  │                │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Overview

| Component | Responsibility | Key Technologies |
|-----------|---------------|------------------|
| **Remote Storage Handler** | Stream files from Google Drive via rclone | rclone, streaming I/O |
| **ZIP Reader** | In-memory ZIP traversal and file extraction | Python zipfile, io.BytesIO |
| **Document Parser** | Extract text/structure from PDFs and CSVs | PyMuPDF, pdfplumber, pandas |
| **Embedding Service** | Generate multi-representation embeddings | OpenAI/Sentence-Transformers |
| **Vector Index** | Store and query embeddings | Qdrant |
| **Classifier** | Aggregate evidence and predict classes | Custom scoring logic |

---

## 2. Data Strategy

### 2.1 Training Data Access Strategy

Given the 64.5 GB training archive on Google Drive, we have three options:

#### Option A: Streaming Processing (Recommended)
```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Google Drive ──rclone cat──▶ tar stream ──▶ Process in chunks │
│                                                                  │
│   Benefits:                                                      │
│   • No local storage required                                    │
│   • Process files as they stream                                 │
│   • Resumable processing                                         │
│                                                                  │
│   Challenges:                                                    │
│   • Complex streaming tar extraction                             │
│   • Network dependency                                           │
│   • Cannot random-access files                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Option B: Partial Download with Batching
```
┌─────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Extract file list from archive (metadata only)              │
│   2. Download files in batches (e.g., 100 files at a time)       │
│   3. Process batch → Generate embeddings → Store in Qdrant       │
│   4. Delete local batch → Repeat                                 │
│                                                                  │
│   Benefits:                                                      │
│   • Controlled memory usage                                      │
│   • Resumable                                                    │
│   • Can parallelize                                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Option C: Cloud-Based Processing
```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD PROCESSING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Deploy processing pipeline to cloud (GCP/AWS)                  │
│   • Mount Google Drive directly                                  │
│   • Process with cloud compute                                   │
│   • Store embeddings in managed Qdrant                           │
│                                                                  │
│   Benefits:                                                      │
│   • Fastest processing                                           │
│   • No local resource constraints                                │
└─────────────────────────────────────────────────────────────────┘
```

**Recommendation:** Start with **Option B (Batch Processing)** for development, with architecture supporting **Option C** for production.

### 2.2 Training Data Structure (Confirmed)

The training archive has a **fixed 6-level hierarchy** with class labels at level 6:

```
Fast/TrainData/RYLTY/Organizer/Statement/
├── Believe Digital/
│   ├── 1774531_141417_20191001_20191231.csv
│   ├── 2021 - H1 GLADES Sales Export.csv.csv
│   └── ...
├── The Orchard/
│   ├── statement_2024_Q1.csv
│   └── ...
├── EMPIRE Distribution/
│   └── ...
├── Spotify Settlement/
│   └── ...
└── [55+ distributor/source classes]
```

**Directory Structure Breakdown:**
| Level | Value | Description |
|-------|-------|-------------|
| 1 | `Fast` | Root container |
| 2 | `TrainData` | Training data marker |
| 3 | `RYLTY` | Application/system name |
| 4 | `Organizer` | Module name |
| 5 | `Statement` | Document category |
| 6 | **CLASS LABEL** | Distributor/Source name |
| 7 | Filename | Actual document |

**Label Extraction Rule:** `class_label = path.split('/')[5]` (0-indexed: position 5 = level 6)

**Confirmed Classes (55+ distributors/sources):**
- Believe Digital, The Orchard, EMPIRE Distribution, Spotify Settlement
- 4AD, 88rising, Create Music Group, LabelWorx, Monstercat Records
- SESAC Publisher, STEMRA, PPL Artist, GMR Publisher, KODA Publisher
- And many more...

### 2.3 Multi-Representation Strategy

Each document generates multiple embeddings:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PDF REPRESENTATIONS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                                │
│   │   PDF File  │                                                │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ├──▶ [SUMMARY]     Full document summary embedding      │
│          │                                                       │
│          ├──▶ [CHUNK_1]     Page 1-2 embedding                   │
│          ├──▶ [CHUNK_2]     Page 3-4 embedding                   │
│          ├──▶ [CHUNK_N]     ...                                  │
│          │                                                       │
│          └──▶ [KEYWORDS]    Sparse representation (BM25)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CSV REPRESENTATIONS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                                │
│   │   CSV File  │                                                │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ├──▶ [SCHEMA]      Column names + inferred types        │
│          │                                                       │
│          ├──▶ [CONTENT]     Sample rows as natural language      │
│          │                                                       │
│          ├──▶ [SUMMARY]     LLM-generated table description      │
│          │                                                       │
│          └──▶ [KEYWORDS]    Column names + unique values         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Design

### 3.1 Project Structure

```
music_rights_embeddings_match/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration management
│   │   └── constants.py         # System constants
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── remote_storage.py    # rclone integration
│   │   ├── zip_reader.py        # In-memory ZIP handling
│   │   ├── tar_reader.py        # Streaming tar extraction
│   │   └── file_iterator.py     # Unified file iteration
│   │
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base_parser.py       # Abstract parser interface
│   │   ├── pdf_parser.py        # PDF text extraction
│   │   ├── csv_parser.py        # CSV semantic conversion
│   │   ├── text_parser.py       # Plain text handling
│   │   └── parser_factory.py    # Parser selection logic
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── chunking.py          # Text chunking strategies
│   │   ├── representations.py   # Multi-representation logic
│   │   └── sparse_encoder.py    # BM25/sparse embeddings
│   │
│   ├── vectordb/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py     # Qdrant operations
│   │   ├── collection_manager.py # Collection setup
│   │   └── payload_schema.py    # Payload definitions
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Stage 1: ANN retrieval
│   │   ├── reranker.py          # Stage 2: Reranking
│   │   ├── aggregator.py        # Class score aggregation
│   │   ├── classifier.py        # Final classification
│   │   └── thresholds.py        # Unknown/abstain logic
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py # End-to-end training
│   │   └── inference_pipeline.py # End-to-end inference
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py    # Logging setup
│       ├── metrics.py           # Evaluation metrics
│       └── helpers.py           # Utility functions
│
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── infer.py                 # Inference entry point
│   ├── evaluate.py              # Evaluation script
│   └── explore_data.py          # Data exploration
│
├── tests/
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_embeddings.py
│   ├── test_classification.py
│   └── fixtures/
│       └── sample_files/
│
├── configs/
│   ├── default.yaml             # Default configuration
│   ├── production.yaml          # Production settings
│   └── development.yaml         # Development settings
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_classification_tuning.ipynb
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── IMPLEMENTATION_PLAN.md       # This document
```

### 3.2 Core Interfaces

#### 3.2.1 Document Parser Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class RepresentationType(Enum):
    SUMMARY = "summary"
    CHUNK = "chunk"
    SCHEMA = "schema"
    CONTENT = "content"
    KEYWORDS = "keywords"

@dataclass
class DocumentRepresentation:
    text: str
    representation_type: RepresentationType
    metadata: dict  # chunk_id, page_range, etc.

@dataclass
class ParsedDocument:
    file_name: str
    file_type: str
    representations: List[DocumentRepresentation]
    raw_text: Optional[str] = None

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, file_name: str) -> ParsedDocument:
        """Parse file bytes into semantic representations."""
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if parser supports this file type."""
        pass
```

#### 3.2.2 Embedding Service Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class EmbeddingResult:
    vector: np.ndarray
    model_name: str
    dimension: int

class EmbeddingService(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass
```

#### 3.2.3 Vector Store Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class VectorPayload:
    class_label: str
    file_name: str
    file_type: str
    representation_type: str
    chunk_id: Optional[int] = None
    text_preview: Optional[str] = None

@dataclass
class SearchResult:
    id: str
    score: float
    payload: VectorPayload

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, vectors: List[np.ndarray], payloads: List[VectorPayload]) -> None:
        """Insert or update vectors with payloads."""
        pass
    
    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Dict[int, float],
        top_k: int,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """Hybrid dense + sparse search."""
        pass
```

#### 3.2.4 Classifier Interface

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    class_scores: Dict[str, float]
    supporting_evidence: List[SearchResult]
    needs_review: bool = False

class Classifier(ABC):
    @abstractmethod
    def classify(
        self, 
        document: ParsedDocument,
        embeddings: List[EmbeddingResult]
    ) -> ClassificationResult:
        """Classify a document based on its embeddings."""
        pass
```

### 3.3 Qdrant Payload Schema

```python
PAYLOAD_SCHEMA = {
    "class_label": str,           # Label from training folder
    "file_name": str,             # Original file name
    "file_path": str,             # Full path (for ZIP: internal path)
    "file_type": str,             # "pdf", "csv", "txt", etc.
    "representation_type": str,   # "summary", "chunk", "schema", "content"
    "chunk_id": int,              # Chunk index (null for non-chunks)
    "chunk_metadata": {           # Additional chunk info
        "page_start": int,
        "page_end": int,
        "char_start": int,
        "char_end": int
    },
    "text_preview": str,          # First 500 chars for debugging
    "source": str,                # "training" or "inference"
    "created_at": str,            # ISO timestamp
}
```

---

## 4. Implementation Phases

### Phase 1: Foundation (MVP)
**Goal:** End-to-end pipeline with basic functionality

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1 SCOPE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ✓ Remote file access via rclone                                │
│   ✓ Basic PDF text extraction                                    │
│   ✓ Basic CSV to text conversion                                 │
│   ✓ Single embedding per document (summary only)                 │
│   ✓ Qdrant setup and basic operations                            │
│   ✓ Simple top-1 classification                                  │
│   ✓ ZIP file in-memory reading                                   │
│                                                                  │
│   Deliverables:                                                  │
│   • Working training pipeline                                    │
│   • Working inference pipeline                                   │
│   • Basic accuracy metrics                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
1. [ ] Set up project structure and dependencies
2. [ ] Implement rclone streaming wrapper
3. [ ] Implement basic PDF parser (PyMuPDF)
4. [ ] Implement basic CSV parser (pandas → text)
5. [ ] Set up embedding service (OpenAI text-embedding-3-small)
6. [ ] Set up Qdrant (local Docker instance)
7. [ ] Implement training pipeline
8. [ ] Implement ZIP reader
9. [ ] Implement inference pipeline
10. [ ] Basic evaluation script

### Phase 2: Multi-Representation
**Goal:** Robust document representations

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 2 SCOPE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ✓ PDF chunking (semantic + page-based)                         │
│   ✓ Multiple embeddings per document                             │
│   ✓ CSV schema extraction                                        │
│   ✓ CSV content sampling                                         │
│   ✓ Representation-aware storage                                 │
│                                                                  │
│   Deliverables:                                                  │
│   • Chunking strategies implemented                              │
│   • Multi-representation embeddings                              │
│   • Updated Qdrant schema                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
1. [ ] Implement semantic chunking for PDFs
2. [ ] Implement page-based chunking
3. [ ] Implement token-window chunking
4. [ ] Implement CSV schema extractor
5. [ ] Implement CSV content sampler
6. [ ] Update embedding service for batching
7. [ ] Update Qdrant upsert for multi-vectors

### Phase 3: Advanced Retrieval
**Goal:** Hybrid search and reranking

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 3 SCOPE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ✓ Sparse embeddings (BM25)                                     │
│   ✓ Hybrid search (dense + sparse)                               │
│   ✓ Two-stage retrieval                                          │
│   ✓ Reranking module                                             │
│   ✓ Metadata filtering                                           │
│                                                                  │
│   Deliverables:                                                  │
│   • Hybrid search implementation                                 │
│   • Reranker with multiple signals                               │
│   • Filtered retrieval                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
1. [ ] Implement BM25 sparse encoder
2. [ ] Configure Qdrant for hybrid search
3. [ ] Implement RRF fusion
4. [ ] Implement reranker module
5. [ ] Add representation-type filtering
6. [ ] Add file-type filtering

### Phase 4: Classification Refinement
**Goal:** Robust classification with aggregation

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 4 SCOPE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ✓ Class score aggregation                                      │
│   ✓ Late fusion scoring                                          │
│   ✓ Unknown/abstain handling                                     │
│   ✓ Class prototypes                                             │
│   ✓ Confidence calibration                                       │
│                                                                  │
│   Deliverables:                                                  │
│   • Aggregation strategies                                       │
│   • Threshold tuning                                             │
│   • Calibrated confidence scores                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
1. [ ] Implement class aggregation (max, avg, count)
2. [ ] Implement late fusion scoring
3. [ ] Implement threshold-based abstention
4. [ ] Compute and store class prototypes
5. [ ] Implement confidence calibration
6. [ ] Add tie-breaker logic

### Phase 5: Evaluation & Optimization
**Goal:** Production-ready quality

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 5 SCOPE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ✓ Comprehensive evaluation framework                           │
│   ✓ Hyperparameter tuning                                        │
│   ✓ Performance optimization                                     │
│   ✓ Error analysis                                               │
│                                                                  │
│   Deliverables:                                                  │
│   • Evaluation dashboard                                         │
│   • Tuned parameters                                             │
│   • Performance benchmarks                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**
1. [ ] Implement Recall@k, MRR metrics
2. [ ] Implement accuracy, F1, confusion matrix
3. [ ] Create evaluation script
4. [ ] Tune aggregation weights
5. [ ] Tune similarity thresholds
6. [ ] Optimize batch sizes
7. [ ] Profile and optimize bottlenecks

---

## 5. Technical Specifications

### 5.1 Embedding Models

| Model | Dimension | Use Case | Cost |
|-------|-----------|----------|------|
| **`Alibaba-NLP/gte-large-en-v1.5`** | 1024 | **Primary (GPU, local)** | Free |
| `nvidia/NV-Embed-v2` | 4096 | High-quality alternative | Free (compatibility issues) |
| `text-embedding-3-small` | 1536 | API fallback | $0.02/1M tokens |
| BM25 | Sparse | Keyword matching | Free |

**Current Setup:** Using `Alibaba-NLP/gte-large-en-v1.5` on RunPod GPU (RTX 4090).
- 8192 token context window
- 1024-dimensional embeddings
- Excellent performance on retrieval benchmarks
- No API costs or rate limits

### 5.2 Chunking Parameters

```python
CHUNKING_CONFIG = {
    "pdf": {
        "strategy": "semantic",  # or "page", "token_window"
        "max_chunk_tokens": 512,
        "overlap_tokens": 50,
        "min_chunk_tokens": 100,
    },
    "csv": {
        "max_sample_rows": 10,
        "max_columns_in_schema": 50,
    }
}
```

### 5.3 Classification Parameters

```python
CLASSIFICATION_CONFIG = {
    "retrieval": {
        "top_k_stage1": 50,      # ANN retrieval
        "top_k_stage2": 20,      # After reranking
        "top_k_aggregation": 10, # For class scoring
    },
    "aggregation": {
        "weight_max_sim": 0.5,
        "weight_avg_sim": 0.3,
        "weight_count": 0.2,
    },
    "thresholds": {
        "min_similarity": 0.65,
        "min_margin": 0.1,       # Between top 2 classes
    },
    "late_fusion": {
        "weight_summary": 0.4,
        "weight_chunk": 0.4,
        "weight_sparse": 0.2,
    }
}
```

### 5.4 Qdrant Configuration

```python
QDRANT_CONFIG = {
    "collection_name": "music_rights_documents",
    "vector_config": {
        "dense": {
            "size": 1536,
            "distance": "Cosine",
        },
        "sparse": {
            "index": {
                "on_disk": False,
            }
        }
    },
    "optimizers_config": {
        "indexing_threshold": 20000,
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
    }
}
```

---

## 6. Evaluation Framework

### 6.1 Metrics

#### Retrieval Metrics
- **Recall@k:** Proportion of relevant documents in top-k
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant result
- **NDCG@k:** Normalized Discounted Cumulative Gain

#### Classification Metrics
- **Accuracy:** Overall correct predictions
- **Macro F1:** Average F1 across all classes
- **Weighted F1:** F1 weighted by class frequency
- **Confusion Matrix:** Per-class error analysis

### 6.2 Evaluation Protocol

```python
def evaluate_pipeline(test_data, pipeline):
    """
    Evaluation protocol:
    1. Hold out 20% of training data for validation
    2. Run inference on held-out set
    3. Compute all metrics
    4. Generate error analysis report
    """
    results = {
        "retrieval": {
            "recall@5": compute_recall(predictions, k=5),
            "recall@10": compute_recall(predictions, k=10),
            "mrr": compute_mrr(predictions),
        },
        "classification": {
            "accuracy": compute_accuracy(predictions),
            "macro_f1": compute_f1(predictions, average="macro"),
            "weighted_f1": compute_f1(predictions, average="weighted"),
            "confusion_matrix": compute_confusion_matrix(predictions),
        },
        "abstention": {
            "abstention_rate": compute_abstention_rate(predictions),
            "accuracy_on_predictions": compute_accuracy_excluding_abstentions(predictions),
        }
    }
    return results
```

---

## 7. Deployment Considerations

### 7.1 Local Development Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 4. Configure environment
cp .env.example .env
# Edit .env with API keys

# 5. Run training
python scripts/train.py --config configs/development.yaml

# 6. Run inference
python scripts/infer.py --input path/to/test.zip --output results.json
```

### 7.2 Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   API       │     │  Worker     │     │  Qdrant     │       │
│   │   Gateway   │────▶│  Service    │────▶│  Cluster    │       │
│   │   (FastAPI) │     │  (Celery)   │     │  (Managed)  │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│          │                   │                                   │
│          ▼                   ▼                                   │
│   ┌─────────────┐     ┌─────────────┐                           │
│   │   Redis     │     │   S3/GCS    │                           │
│   │   (Queue)   │     │   (Storage) │                           │
│   └─────────────┘     └─────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Resource Requirements

| Component | Development | Production |
|-----------|-------------|------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| Storage | 50 GB | 500+ GB |
| GPU | Optional | Recommended |
| Qdrant | Local Docker | Managed cluster |

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large training data (64GB) | Slow processing | Batch processing, streaming |
| Network failures | Interrupted training | Checkpointing, resume logic |
| Embedding API rate limits | Slow training | Batching, caching, local fallback |
| Class imbalance | Poor minority class recall | Stratified sampling, class weights |
| Ambiguous documents | Low confidence | Abstention mechanism |
| PDF parsing failures | Missing data | Fallback parsers, error logging |

---

## 9. Next Steps

1. **Immediate:** Explore training data structure (extract sample from archive)
2. **Day 1:** Set up project skeleton and dependencies
3. **Day 2-3:** Implement Phase 1 (MVP)
4. **Day 4-5:** Test with sample data, iterate
5. **Week 2:** Implement Phases 2-3
6. **Week 3:** Implement Phases 4-5
7. **Week 4:** Production hardening and documentation

---

## Appendix A: Dependencies

```
# requirements.txt
# Core
python>=3.10
pydantic>=2.0
pyyaml>=6.0

# Data Processing
pandas>=2.0
numpy>=1.24

# PDF Processing
pymupdf>=1.23  # aka fitz
pdfplumber>=0.10

# Embeddings
openai>=1.0
sentence-transformers>=2.2

# Vector Database
qdrant-client>=1.7

# Sparse Embeddings
rank-bm25>=0.2

# Utilities
python-dotenv>=1.0
tqdm>=4.65
loguru>=0.7

# Testing
pytest>=7.0
pytest-asyncio>=0.21

# Development
black>=23.0
ruff>=0.1
mypy>=1.0
```

---

## Appendix B: Sample Output Format

```json
{
  "file_path": "contracts/2024/artist_agreement_001.pdf",
  "predicted_class": "artist_contract",
  "confidence": 0.87,
  "needs_review": false,
  "class_scores": {
    "artist_contract": 0.87,
    "licensing_agreement": 0.45,
    "royalty_statement": 0.23
  },
  "supporting_evidence": [
    {
      "file_name": "sample_artist_contract.pdf",
      "class_label": "artist_contract",
      "similarity": 0.92,
      "representation_type": "summary"
    },
    {
      "file_name": "artist_deal_memo.pdf",
      "class_label": "artist_contract", 
      "similarity": 0.88,
      "representation_type": "chunk"
    }
  ],
  "processing_time_ms": 234
}
```

---

*Document Version: 1.0 | Last Updated: April 12, 2026*
