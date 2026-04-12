# Music Rights Embeddings Match

Embedding-based document classification system for music rights documents using semantic similarity and vector search.

## Features

- **Multi-representation embeddings**: Documents are represented by multiple embeddings (summary, chunks, schema, keywords)
- **GPU-accelerated**: Uses local embedding models on GPU (Alibaba-NLP/gte-large-en-v1.5)
- **Vector database**: Qdrant for efficient similarity search
- **Hybrid retrieval**: Dense + sparse embeddings for robust matching
- **Aggregation-based classification**: Uses top-k matches with weighted scoring

## Project Structure

```
├── src/
│   ├── config/          # Configuration settings
│   ├── data/            # Data loaders (ZIP, tar, directory)
│   ├── parsers/         # PDF, CSV, text parsers
│   ├── embeddings/      # GPU embedding service
│   ├── vectordb/        # Qdrant integration
│   ├── classification/  # Classifier with aggregation
│   └── pipelines/       # Training & inference pipelines
├── scripts/
│   ├── train.py         # Training entry point
│   ├── infer.py         # Inference entry point
│   └── test_setup.py    # Setup verification
├── holdout_datasets/    # Final test datasets (not for training)
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python scripts/test_setup.py

# Train on labeled data
python scripts/train.py --source /path/to/training_data

# Run inference
python scripts/infer.py --input /path/to/test.zip --output results.json
```

## Classification Approach

1. Parse documents into semantic representations
2. Generate embeddings using GTE-large model
3. Store in Qdrant vector database
4. For inference: retrieve top-k similar documents
5. Aggregate class scores using: `0.5 * max_sim + 0.3 * avg_sim + 0.2 * count_ratio`
6. Apply confidence thresholds for abstention

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
