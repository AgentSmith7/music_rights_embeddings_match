# Music Rights Document Classification

A production-grade document classification system for music rights documents using embedding-based retrieval and vector search. Achieves **99.87% accuracy** on confident predictions.

## Overview

This system classifies music rights documents (royalty statements, contracts, metadata files) into their respective categories using:
- **Semantic embeddings** from BAAI/bge-large-en-v1.5
- **Vector similarity search** with Qdrant
- **Intelligent aggregation** strategies for robust classification
- **Confidence thresholds** to flag uncertain predictions

## Performance

| Metric | Value |
|--------|-------|
| Accuracy (confident predictions) | 99.87% |
| Accuracy (all predictions) | 92.39% |
| Training documents | 64,912 |
| Validation documents | 16,716 |
| Supported file types | CSV, XLSX, XLS, PDF, TXT |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Classify a single file
python demo/demo_classify.py --file document.csv --qdrant-path ./qdrant_data

# Batch process a directory
python demo/demo_batch_inference.py --input ./documents/ --output-csv results.csv

# Interactive exploration
python demo/demo_explorer.py --qdrant-path ./qdrant_data
```

## Project Structure

```
├── demo/                    # Runnable demo scripts
│   ├── demo_classify.py     # Single file classification
│   ├── demo_batch_inference.py  # Batch processing
│   └── demo_explorer.py     # Interactive exploration
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── EDA_RESULTS.md       # Exploratory data analysis
│   ├── EXPERIMENT_RESULTS.md    # Experiment leaderboard
│   ├── EXPERIMENT_CONFIGURATIONS.md  # Experiment matrix
│   └── IMPLEMENTATION_PLAN.md   # Detailed implementation plan
├── scripts/                 # Core pipeline scripts
│   ├── streaming_pipeline.py    # 3-stage training pipeline
│   ├── experiment_runner.py     # Experiment framework
│   ├── dureco_inference.py      # Holdout inference
│   └── efficient_eda.py         # EDA utilities
├── holdout_datasets/        # Test datasets (not for training)
└── requirements.txt
```

## Architecture

The system uses a **3-stage bounded, backpressured pipeline**:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Parser Pool    │───▶│  GPU Embedder    │───▶│  Qdrant Writer  │
│  (CPU, multi-   │    │  (single GPU,    │    │  (batched       │
│   process)      │    │   char-budget    │    │   upserts)      │
│                 │    │   batching)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        ▲                      │                       │
        │                      ▼                       ▼
   Bounded Queue          Bounded Queue          Checkpoint
   (backpressure)         (backpressure)         + Retry
```

### Key Design Decisions

1. **Compact Fingerprints**: Documents are parsed into semantic summaries (schema + sample rows for tabular data, first pages for PDFs) rather than full content
2. **Bounded Queues**: Prevents memory overflow with large datasets
3. **Checkpointing**: Resume from failures without reprocessing
4. **Confidence Thresholds**: Abstain on uncertain predictions for higher accuracy

## Classification Strategy

### Inference Pipeline

1. **Parse** document into semantic representation
2. **Embed** using BGE-large model (GPU accelerated)
3. **Search** Qdrant for top-k similar training documents
4. **Aggregate** class scores using weighted formula
5. **Apply thresholds** for confidence filtering

### Best Configuration (from experiments)

```python
# Primary strategy: High-confidence predictions
aggregation = "mean"
top_k = 10
min_margin = 0.10  # Abstain if margin < 0.10

# Fallback strategy: When primary abstains
aggregation = "weighted"  # 0.5*max + 0.3*avg + 0.2*count
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and diagrams
- [EDA Results](docs/EDA_RESULTS.md) - Training data analysis
- [Experiment Results](docs/EXPERIMENT_RESULTS.md) - Performance leaderboard
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Detailed technical plan

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM
- ~10GB disk space for Qdrant index

## License

MIT
