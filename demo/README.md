# Demo Scripts

This folder contains runnable demo scripts for the Music Rights Document Classification system.

## Prerequisites

1. **Python 3.9+** with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU** (recommended): CUDA-enabled GPU for fast embedding generation

3. **Qdrant Vector Database**: Either running locally or with data path configured

## Available Demos

### 1. Quick Classification Demo (`demo_classify.py`)
Classify a single file or directory of files.

```bash
# Classify a single file
python demo/demo_classify.py --file /path/to/document.csv

# Classify all files in a directory
python demo/demo_classify.py --directory /path/to/documents/

# Use custom Qdrant path
python demo/demo_classify.py --file doc.pdf --qdrant-path /path/to/qdrant_data
```

### 2. Batch Inference Demo (`demo_batch_inference.py`)
Run inference on a ZIP archive or large directory with progress tracking.

```bash
# Process a ZIP file
python demo/demo_batch_inference.py --input /path/to/archive.zip --output results.json

# Process with CSV output
python demo/demo_batch_inference.py --input /path/to/docs/ --output-csv results.csv
```

### 3. Interactive Explorer (`demo_explorer.py`)
Interactively explore the vector database and find similar documents.

```bash
python demo/demo_explorer.py --qdrant-path /path/to/qdrant_data
```

### 4. Embedding Visualization (`demo_visualize.py`)
Generate t-SNE/UMAP visualizations of document embeddings.

```bash
python demo/demo_visualize.py --qdrant-path /path/to/qdrant_data --output embeddings.html
```

## Configuration

All demos support these common arguments:
- `--qdrant-path`: Path to Qdrant data directory (default: `./qdrant_data`)
- `--model`: Embedding model name (default: `BAAI/bge-large-en-v1.5`)
- `--device`: Device for inference (`cuda` or `cpu`, default: auto-detect)

## Example Workflow

```bash
# 1. Classify a single document
python demo/demo_classify.py --file sample.csv
# Output: Predicted class: ASCAP_ROYALTY with confidence 0.89

# 2. Batch process a folder
python demo/demo_batch_inference.py --input ./test_docs/ --output-csv predictions.csv

# 3. Explore similar documents
python demo/demo_explorer.py
# > Enter query: royalty statement BMI
# > Found 10 similar documents...
```

## Output Formats

### CSV Output
```csv
file_path,file_name,predicted_class,confidence,margin,is_uncertain,strategy
doc1.csv,doc1.csv,ASCAP_ROYALTY,0.8934,0.1523,False,primary
doc2.pdf,doc2.pdf,BMI_STATEMENT,0.7821,0.0891,True,fallback
```

### JSON Output
```json
{
  "file_path": "doc1.csv",
  "predicted_class": "ASCAP_ROYALTY",
  "confidence": 0.8934,
  "margin": 0.1523,
  "is_uncertain": false,
  "top_classes": [
    {"class": "ASCAP_ROYALTY", "score": 0.8934},
    {"class": "BMI_STATEMENT", "score": 0.7411}
  ]
}
```
