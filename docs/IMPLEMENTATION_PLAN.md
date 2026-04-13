# Music Rights Document Classification System
## Implementation Plan

**Version:** 2.0  
**Date:** April 13, 2026  
**Status:** IMPLEMENTED  
**Project:** Embedding-based Document Retrieval Classification Pipeline

> **System Overview:** This is an **embedding-based retrieval classification system** that classifies 
> music rights documents by finding the most similar training documents in vector space and 
> aggregating their class labels. No model training occurs - only embedding generation and indexing.

---

## Executive Summary

This document describes the **implemented** production-grade document classification system for music rights documents. The system achieved **99.87% accuracy** on confident predictions (with margin threshold) and **89.93% baseline accuracy** on the validation set.

### Key Results
- **Training Set:** 65,970 documents indexed across 58 classes
- **Validation Set:** 16,493 documents (80/20 stratified split)
- **Best Configuration:** Margin threshold (0.10) achieving 99.87% accuracy on 796/1000 confident predictions
- **Holdout Inference:** DURECO (598 files) and PubStrengholtSociety (1,852 files) processed

### System Highlights
- 3-stage bounded, backpressured pipeline architecture
- Local GPU embedding with BAAI/bge-large-en-v1.5 (1024 dimensions)
- Qdrant vector database with INT8 scalar quantization
- Primary/fallback classification strategy with confidence thresholds

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Pipeline](#2-data-pipeline)
3. [Classification Strategy](#3-classification-strategy)
4. [Experiment Results](#4-experiment-results)
5. [Implementation Details](#5-implementation-details)
6. [Deployment](#6-deployment)

---

## 1. System Architecture

### 1.1 Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3-STAGE BOUNDED PIPELINE (IMPLEMENTED)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: PARSER PROCESS POOL (CPU-bound, multiprocessing)                 │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  File Iterator → Extension Detection → Parse to Compact Fingerprint  │  │
│   │  • PDF: First 3 pages text (PyMuPDF)                                 │  │
│   │  • CSV/Excel: Schema + 10 sample rows (pandas)                       │  │
│   │  • TXT: First 4096 chars                                             │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼ Bounded Queue (maxsize=256)                   │
│                                                                              │
│   STAGE 2: SINGLE GPU BATCH EMBEDDER                                        │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Collect parsed items → Batch by char budget → Embed → Float16       │  │
│   │  • Model: BAAI/bge-large-en-v1.5 (1024 dim)                          │  │
│   │  • Batch size: 32 items or 120K chars                                │  │
│   │  • Timeout flush: 1 second                                           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼ Bounded Queue (maxsize=64)                    │
│                                                                              │
│   STAGE 3: BATCH WRITER + CHECKPOINTER                                      │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Batch upserts to Qdrant → Checkpoint progress → Log failures        │  │
│   │  • Upsert batch: 100 points                                          │  │
│   │  • Checkpoint: JSONL progress log                                    │  │
│   │  • Failures: Separate failure log                                    │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Inference Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  ZIP/tar.gz  │    │  Document    │    │  Embedding   │                  │
│   │  (in-memory) │───▶│  Parser      │───▶│  Generator   │                  │
│   │  + nested    │    │              │    │  (GPU)       │                  │
│   └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                   │                          │
│                                                   ▼                          │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      CLASSIFICATION ENGINE                            │  │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │  │
│   │  │ Qdrant Search  │─▶│ Score          │─▶│ Primary/Fallback       │  │  │
│   │  │ (top-k=10)     │  │ Aggregation    │  │ Strategy Selection     │  │  │
│   │  └────────────────┘  └────────────────┘  └────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                   │                          │
│                                                   ▼                          │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  OUTPUT: CSV + JSON with predictions, confidence, uncertainty flags  │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Component Summary

| Component | Implementation | Technology |
|-----------|---------------|------------|
| **File Parsing** | Extension-specific parsers | PyMuPDF (PDF), pandas (CSV/Excel), xlrd/openpyxl |
| **Embedding** | Local GPU model | BAAI/bge-large-en-v1.5 via sentence-transformers |
| **Vector Store** | Local Qdrant | INT8 scalar quantization, 64,912 points |
| **Classification** | Retrieval + aggregation | Mean/weighted aggregation with thresholds |

---

## 2. Data Pipeline

### 2.1 Training Data Structure

**Source:** Extracted from `trainData.tgz` (351 GB extracted)

```
/workspace/extracted/Fast/TrainData/RYLTY/Organizer/Statement/
├── Believe Digital/           # Class label
│   ├── file1.csv
│   └── file2.pdf
├── Sony Music Entertainment/  # Class label
│   └── ...
├── STIM/                      # Class label
│   └── ...
└── [58 total classes]
```

**Label Extraction:** `class_label = path_parts[5]` (6th directory level)

### 2.2 Dataset Statistics (from EDA)

| Metric | Value |
|--------|-------|
| Total files | 82,463 |
| Training set (80%) | 65,970 |
| Validation set (20%) | 16,493 |
| Unique classes | 58 |
| File types | CSV (67.5%), PDF (19.8%), Excel (12.4%), TXT (0.3%) |

**Top 10 Classes by File Count:**
1. PRS Writer: 11,006 (13.3%)
2. Sony Music Entertainment: 8,295 (10.1%)
3. BMI Writer: 6,553 (7.9%)
4. STIM: 5,313 (6.4%)
5. ASCAP Publisher: 4,783 (5.8%)
6. SESAC Publisher: 4,426 (5.4%)
7. SACEM Writer: 3,198 (3.9%)
8. BUMA Writer: 2,891 (3.5%)
9. STIM Writer: 2,654 (3.2%)
10. Warner Music Group: 2,341 (2.8%)

### 2.3 Document Representation Strategy

**Implemented: Full Content Summary (Option C)**

Each document is converted to a single semantic fingerprint:

```python
# PDF Representation
def parse_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    text_parts = []
    for page_num in range(min(3, len(doc))):  # First 3 pages
        text_parts.append(doc[page_num].get_text()[:2000])
    return "\n".join(text_parts)[:4096]

# CSV/Excel Representation  
def parse_tabular(content: bytes) -> str:
    df = pd.read_csv/excel(content, nrows=50)
    cols = list(df.columns[:30])
    sample = df.head(10).to_string(index=False, max_colwidth=50)
    return f"Schema: {', '.join(cols)}\nSample Data:\n{sample}"
```

---

## 3. Classification Strategy

### 3.1 Two-Stage Strategy (Implemented)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Query Document                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PRIMARY STRATEGY                                        │   │
│   │  • Aggregation: Mean                                     │   │
│   │  • Top-k: 10                                             │   │
│   │  • Min margin: 0.10 (between top-2 classes)              │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                         │
│        ├── Margin >= 0.10? ──▶ HIGH CONFIDENCE PREDICTION        │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  FALLBACK STRATEGY                                       │   │
│   │  • Aggregation: Weighted (0.5*max + 0.3*avg + 0.2*count) │   │
│   │  • Top-k: 10                                             │   │
│   │  • Uncertainty threshold: 0.70                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│        │                                                         │
│        ▼                                                         │
│   PREDICTION (with uncertainty flag if confidence < 0.70)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Aggregation Methods Tested

| Method | Formula | Accuracy |
|--------|---------|----------|
| **Mean** | `avg(similarities)` per class | 89.93% (baseline) |
| **Max** | `max(similarity)` per class | 92.10% |
| **Weighted** | `0.5*max + 0.3*avg + 0.2*norm_count` | 92.39% |
| **Count** | `count(matches)` per class | 90.72% |
| **Sum** | `sum(similarities)` per class | 90.27% |

### 3.3 Confidence Thresholds

| Threshold Type | Value | Effect |
|---------------|-------|--------|
| Min similarity | 0.65-0.70 | Abstain if top score below |
| Min margin | 0.10 | Abstain if gap between top-2 classes below |
| Combined | Both | Most selective, highest accuracy |

---

## 4. Experiment Results

### 4.1 Leaderboard (1000-sample validation set)

| Rank | Experiment | Accuracy | Predictions | Notes |
|------|------------|----------|-------------|-------|
| 1 | **thresh_margin_010** | **99.87%** | 796/1000 | Min margin 0.10 |
| 2 | thresh_combined | 99.51% | 814/1000 | Similarity 0.65 + margin 0.10 |
| 3 | best_weighted_k20_thresh | 98.91% | 828/1000 | Weighted + thresholds |
| 4 | agg_weighted | 92.39% | 985/1000 | Weighted aggregation |
| 5 | topk_20 | 92.26% | 982/1000 | Top-20 retrieval |
| 6 | agg_max | 92.10% | 975/1000 | Max similarity |
| 7 | agg_count | 90.72% | 981/1000 | Vote count |
| 8 | topk_5 | 90.52% | 981/1000 | Top-5 retrieval |
| 9 | thresh_sim_070 | 90.40% | 979/1000 | Min similarity 0.70 |
| 10 | **baseline** | **89.93%** | 983/1000 | Mean, top-10, no thresholds |

### 4.2 Key Findings

1. **Margin threshold is most effective** - Requiring 0.10 gap between top-2 classes achieves near-perfect accuracy on confident predictions

2. **Trade-off: Accuracy vs Coverage** - Higher thresholds improve accuracy but reduce prediction coverage (abstain more)

3. **Weighted aggregation helps** - Combining max, avg, and count signals outperforms single metrics

4. **Top-k tuning matters** - Top-10 to Top-20 is optimal; too few or too many hurts performance

### 4.3 Production Recommendation

```python
PRODUCTION_CONFIG = {
    "primary": {
        "aggregation": "mean",
        "top_k": 10,
        "min_margin": 0.10,  # High-confidence gate
    },
    "fallback": {
        "aggregation": "weighted",
        "weights": {"max": 0.5, "avg": 0.3, "count": 0.2},
        "top_k": 10,
    },
    "uncertainty_threshold": 0.70,
}
```

---

## 5. Implementation Details

### 5.1 Project Structure (Actual)

```
music_rights_embeddings_match/
├── scripts/
│   ├── streaming_pipeline.py    # Main training pipeline (941 lines)
│   ├── experiment_runner.py     # Experiment framework (669 lines)
│   ├── efficient_eda.py         # EDA script (169 lines)
│   ├── dureco_inference.py      # DURECO holdout inference
│   ├── pubstrengholt_inference.py # PubStrengholt inference
│   └── spot_check.py            # Manual verification
│
├── demo/
│   ├── README.md                # Demo documentation
│   ├── demo_classify.py         # Single file classification
│   ├── demo_batch_inference.py  # Batch inference
│   └── demo_explorer.py         # Qdrant explorer
│
├── docs/
│   ├── IMPLEMENTATION_PLAN.md   # This document
│   ├── ARCHITECTURE.md          # Detailed architecture
│   ├── EDA_RESULTS.md           # EDA findings
│   ├── EXPERIMENT_RESULTS.md    # Full experiment results
│   └── EXPERIMENT_CONFIGURATIONS.md
│
├── dureco_predictions.csv       # DURECO results (598 files)
├── pubstrengholt_predictions.csv # PubStrengholt results (1852 files)
├── eda_results.json             # EDA data
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
```

### 5.2 Key Scripts

#### streaming_pipeline.py
- 3-stage producer-consumer architecture
- Multiprocessing for CPU-bound parsing
- Bounded queues for backpressure
- Checkpointing and failure logging
- Character-budget batching for GPU

#### experiment_runner.py
- Systematic experiment framework
- 15 configuration variants tested
- Stratified validation sampling
- Detailed metrics collection

### 5.3 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Python | 3.11 | - |
| Embedding Model | BAAI/bge-large-en-v1.5 | sentence-transformers |
| Vector Database | Qdrant | Local mode, INT8 quantization |
| PDF Parsing | PyMuPDF (fitz) | 1.23+ |
| Tabular Parsing | pandas, openpyxl, xlrd | Latest |
| GPU | NVIDIA RTX 4090 | RunPod |
| Logging | loguru | 0.7+ |

### 5.4 Qdrant Configuration

```python
QDRANT_CONFIG = {
    "collection_name": "music_rights",
    "vectors": {
        "size": 1024,  # bge-large-en-v1.5 dimension
        "distance": "Cosine",
    },
    "quantization": {
        "scalar": {
            "type": "int8",
            "always_ram": True,
        }
    },
    "payload_schema": {
        "class_label": str,
        "file_name": str,
        "file_type": str,
        "source_path": str,
    }
}
```

---

## 6. Deployment

### 6.1 Training Environment (RunPod)

```bash
# GPU Instance: RTX 4090, 24GB VRAM
# Storage: 500GB pod volume

# Setup
pip install sentence-transformers qdrant-client pandas pymupdf openpyxl xlrd loguru tqdm

# Run training pipeline
python streaming_pipeline.py \
    --data-path /workspace/extracted/Fast/TrainData/RYLTY/Organizer/Statement \
    --qdrant-path /workspace/qdrant_data \
    --split-ratio 0.8 \
    --workers 4

# Run experiments
python experiment_runner.py \
    --qdrant-path /workspace/qdrant_data \
    --sample-size 1000
```

### 6.2 Inference Usage

```python
from scripts.dureco_inference import DURECOInference

# Initialize
inference = DURECOInference(
    qdrant_path="/workspace/qdrant_data",
    embedding_model="BAAI/bge-large-en-v1.5"
)

# Run on ZIP file
results = inference.run_inference_on_zip("holdout.zip")

# Save results
inference.save_results_csv("predictions.csv")
inference.save_results_json("predictions.json")
inference.print_summary()
```

### 6.3 Demo Scripts

```bash
# Single file classification
python demo/demo_classify.py document.pdf

# Batch inference
python demo/demo_batch_inference.py --input folder/ --output results.csv

# Explore vector database
python demo/demo_explorer.py --stats
python demo/demo_explorer.py --search "royalty statement"
```

### 6.4 Backup Strategy

```bash
# Backup to Google Drive via rclone
rclone copy /workspace/qdrant_data gdrive:music_rights_project/qdrant_data
rclone copy /workspace/dureco_predictions.csv gdrive:music_rights_project/
rclone copy /workspace/experiment_results_1k.json gdrive:music_rights_project/
```

---

## 7. Spot Check Results

Manual verification of 15 predictions across holdout datasets:

### DURECO Dataset
| File | Predicted | Actual Source | Verdict |
|------|-----------|---------------|---------|
| 95504028.csv | Sena Artist | SENA folder | **CORRECT** |
| 95677064.pdf | Sony Music | SENA folder | INCORRECT (PDF bias) |
| 95623850.csv | PRS Writer | SENA folder | INCORRECT |
| 95752066.csv | Sena Artist | SENA folder | **CORRECT** |

### PubStrengholt Dataset
| File | Predicted | Actual Source | Verdict |
|------|-----------|---------------|---------|
| 27649154.csv | STIM | STIM folder | **CORRECT** |
| 27649151.pdf | Sony Music | STIM folder | INCORRECT (PDF bias) |
| NCB files | ASCAP/SoundExchange | NCB folder | EXPECTED (out-of-distribution) |

### Observations
1. **CSV files**: Generally classified correctly (~80-90% accuracy)
2. **PDF files**: Significant "Sony Music Entertainment" bias (~40-50% accuracy)
3. **Unknown societies**: Misclassified to nearest neighbor (expected)

### Recommendations
1. Add more diverse PDF training examples
2. Consider file-type-specific strategies
3. Add NCB as a class if common in production

---

## 8. Future Improvements

1. **Multi-representation embeddings** - Add chunk-level and schema-level embeddings
2. **Hybrid search** - Combine dense + sparse (BM25) retrieval
3. **Reranking stage** - Add cross-encoder reranker for top candidates
4. **PDF-specific model** - Train or fine-tune for PDF document understanding
5. **Active learning** - Flag uncertain predictions for human review and retraining

---

*Document Version: 2.0 | Last Updated: April 13, 2026 | Status: IMPLEMENTED*
