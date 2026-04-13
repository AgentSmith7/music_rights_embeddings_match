# System Architecture

## Overview

The Music Rights Document Classification system uses embedding-based retrieval to classify documents into predefined categories. The architecture is designed for:

- **Scalability**: Handle 100k+ documents efficiently
- **Reliability**: Checkpointing and error recovery
- **Performance**: GPU acceleration and batched processing
- **Accuracy**: Multiple aggregation strategies with confidence thresholds

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Files   │───▶│   Parser     │───▶│  Embedder    │───▶│   Qdrant     │  │
│  │ (on disk)│    │   Pool       │    │  (GPU)       │    │   Writer     │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│       │                │                    │                    │          │
│       │           Bounded Queue        Bounded Queue        Checkpoint      │
│       │           (256 items)          (64 items)           + Retry         │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ Progress │                                                               │
│  │  Logger  │                                                               │
│  └──────────┘                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Input   │───▶│   Parser     │───▶│  Embedder    │───▶│   Qdrant     │  │
│  │  File    │    │              │    │  (GPU)       │    │   Search     │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                 │           │
│                                                                 ▼           │
│                                              ┌──────────────────────────┐   │
│                                              │     Aggregation          │   │
│                                              │  ┌────────────────────┐  │   │
│                                              │  │ Primary: margin    │  │   │
│                                              │  │ threshold (0.10)   │  │   │
│                                              │  └────────────────────┘  │   │
│                                              │           │              │   │
│                                              │     confident?          │   │
│                                              │      /      \           │   │
│                                              │    yes       no         │   │
│                                              │     │         │         │   │
│                                              │     ▼         ▼         │   │
│                                              │  Predict   Fallback     │   │
│                                              │            (weighted)   │   │
│                                              └──────────────────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│                                              ┌──────────────────────────┐   │
│                                              │   Output (CSV/JSON)      │   │
│                                              │   - predicted_class      │   │
│                                              │   - confidence           │   │
│                                              │   - is_uncertain         │   │
│                                              └──────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Parser

Converts raw files into semantic text representations.

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PARSER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐                                               │
│  │  CSV    │──▶ Schema (columns) + Sample rows (10)        │
│  └─────────┘                                               │
│                                                             │
│  ┌─────────┐                                               │
│  │  XLSX   │──▶ Sheet names + Schema + Sample rows         │
│  └─────────┘                                               │
│                                                             │
│  ┌─────────┐                                               │
│  │  PDF    │──▶ First 3 pages text (PyMuPDF)               │
│  └─────────┘                                               │
│                                                             │
│  ┌─────────┐                                               │
│  │  TXT    │──▶ First 4096 characters                      │
│  └─────────┘                                               │
│                                                             │
│  Output: Compact semantic fingerprint (< 5KB text)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Design Rationale**: Compact fingerprints rather than full content because:
- Reduces embedding cost
- More stable embeddings (less noise)
- Faster processing
- Schema/structure is often more discriminative than content

### 2. Embedding Model

Uses BAAI/bge-large-en-v1.5 for semantic embeddings.

```
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING MODEL                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model: BAAI/bge-large-en-v1.5                             │
│  Dimension: 1024                                            │
│  Device: CUDA (GPU)                                         │
│  Precision: float16 (for storage)                          │
│                                                             │
│  Batching Strategy:                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  max_batch_items: 32                                │   │
│  │  max_batch_chars: 120,000                           │   │
│  │  batch_timeout: 1 second                            │   │
│  │                                                     │   │
│  │  Flush when ANY condition met:                      │   │
│  │  - items >= 32                                      │   │
│  │  - chars >= 120,000                                 │   │
│  │  - timeout reached                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Vector Database (Qdrant)

Stores embeddings with metadata for similarity search.

```
┌─────────────────────────────────────────────────────────────┐
│                    QDRANT VECTOR DB                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Collection: music_rights                                   │
│  Vectors: 64,912 (training set)                            │
│  Dimension: 1024                                            │
│  Distance: Cosine                                           │
│  Quantization: Scalar (INT8)                               │
│                                                             │
│  Payload Schema:                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  {                                                  │   │
│  │    "doc_id": "unique_identifier",                   │   │
│  │    "class_label": "ASCAP_ROYALTY",                  │   │
│  │    "file_name": "statement_2024.csv",               │   │
│  │    "file_type": ".csv",                             │   │
│  │    "source_path": "/path/to/file"                   │   │
│  │  }                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Index: HNSW (default)                                     │
│  Storage: Local disk (qdrant_data/)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Classification Aggregation

Multiple strategies for combining retrieval results.

```
┌─────────────────────────────────────────────────────────────┐
│                 AGGREGATION STRATEGIES                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Given: top-k retrieved documents with similarity scores    │
│                                                             │
│  1. MEAN (baseline)                                         │
│     class_score = mean(similarities for class)              │
│                                                             │
│  2. MAX                                                     │
│     class_score = max(similarities for class)               │
│                                                             │
│  3. WEIGHTED (best for coverage)                            │
│     class_score = 0.5 * max + 0.3 * mean + 0.2 * count     │
│                                                             │
│  4. MARGIN THRESHOLD (best for accuracy)                    │
│     margin = top_score - second_score                       │
│     if margin < 0.10: ABSTAIN                              │
│     else: predict top class                                 │
│                                                             │
│  Production Strategy:                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Try margin threshold (primary)                  │   │
│  │     - If confident (margin >= 0.10): predict        │   │
│  │     - If uncertain: use fallback                    │   │
│  │                                                     │   │
│  │  2. Fallback: weighted aggregation                  │   │
│  │     - Always produces a prediction                  │   │
│  │     - Flag as uncertain if score < 0.70             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Flow

```
1. Enumerate files from training directory
   └─▶ Extract class_label from folder structure
   
2. Parse each file into semantic fingerprint
   └─▶ CSV: schema + sample rows
   └─▶ PDF: first pages text
   └─▶ Excel: sheet info + schema + samples
   
3. Batch documents for GPU embedding
   └─▶ Character-budget batching (120k chars)
   └─▶ Normalize embeddings
   └─▶ Convert to float16
   
4. Upsert to Qdrant in batches
   └─▶ 100 points per batch
   └─▶ Checkpoint after each batch
   └─▶ Retry on failure
```

### Inference Flow

```
1. Parse input document
   └─▶ Same parsing as training
   
2. Generate embedding
   └─▶ Single document, GPU
   
3. Query Qdrant for top-k (k=10)
   └─▶ Returns similar docs with scores
   
4. Aggregate by class
   └─▶ Group results by class_label
   └─▶ Compute class scores
   
5. Apply decision logic
   └─▶ Primary: margin threshold
   └─▶ Fallback: weighted aggregation
   
6. Output prediction
   └─▶ class, confidence, is_uncertain
```

## Performance Characteristics

### Training Pipeline

| Metric | Value |
|--------|-------|
| Throughput | ~15 files/second |
| GPU utilization | 80-95% |
| Memory usage | ~4GB GPU, ~8GB RAM |
| Checkpoint frequency | Every 100 documents |

### Inference

| Metric | Value |
|--------|-------|
| Single file latency | ~100ms |
| Batch throughput | ~50 files/second |
| Qdrant search time | ~5ms |
| Embedding time | ~50ms |

## Error Handling

```
┌─────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Parser Errors:                                             │
│  - Log to failures.jsonl                                    │
│  - Continue processing other files                          │
│  - Mark as PARSE_ERROR in output                           │
│                                                             │
│  Embedding Errors:                                          │
│  - Retry batch once                                         │
│  - Split into smaller sub-batches                          │
│  - Log failures                                             │
│                                                             │
│  Qdrant Write Errors:                                       │
│  - Exponential backoff retry (3 attempts)                  │
│  - Dead-letter queue for persistent failures               │
│                                                             │
│  Checkpointing:                                             │
│  - checkpoint.jsonl tracks completed doc_ids               │
│  - Resume skips already-processed documents                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Options

### Local Development

```bash
# Start Qdrant (local mode)
# No separate server needed - uses local storage

python demo/demo_classify.py --file doc.csv --qdrant-path ./qdrant_data
```

### Cloud Deployment (RunPod)

```bash
# GPU instance with CUDA
# Upload qdrant_data/ and scripts/

python scripts/streaming_pipeline.py \
    --data-path /workspace/training_data \
    --qdrant-path /workspace/qdrant_data \
    --mode train
```

### Production Considerations

1. **Qdrant Server Mode**: For multi-client access, run Qdrant as a service
2. **Model Caching**: Pre-load embedding model to reduce cold start
3. **Batch Processing**: Use batch inference for throughput
4. **Monitoring**: Track confidence distributions, abstention rates

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | BAAI/bge-large-en-v1.5 (SentenceTransformers) |
| Vector Database | Qdrant (local mode) |
| PDF Parsing | PyMuPDF (fitz) |
| Tabular Parsing | Pandas |
| GPU Framework | PyTorch + CUDA |
| Progress Tracking | tqdm, loguru |
