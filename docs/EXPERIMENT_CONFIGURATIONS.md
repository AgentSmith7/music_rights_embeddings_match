# Experiment Configuration Matrix
## Music Rights Document Classification System

**Version:** 1.0  
**Date:** April 12, 2026  
**Purpose:** Comprehensive experiment configurations for embedding-based retrieval classification

---

## Overview

This document defines all experiment configurations to systematically explore the design space for our embedding-based document classification system. Configurations are organized by dimension, with recommended execution order for efficient exploration.

**Total Experiment Space:** ~100,000+ combinations  
**Recommended Staged Approach:** ~50-80 experiments

---

## Dimension 1: Embedding Models

### 1.1 Dense Embedding Models (Local GPU)

| ID | Model | Dimensions | Context | Speed | Quality | Notes |
|----|-------|------------|---------|-------|---------|-------|
| **M1** | `BAAI/bge-small-en-v1.5` | 384 | 512 | ⚡⚡⚡ | ★★☆ | Fast baseline |
| **M2** | `BAAI/bge-base-en-v1.5` | 768 | 512 | ⚡⚡ | ★★★ | Good balance |
| **M3** | `BAAI/bge-large-en-v1.5` | 1024 | 512 | ⚡ | ★★★★ | Current default |
| **M4** | `Alibaba-NLP/gte-base-en-v1.5` | 768 | 8192 | ⚡⚡ | ★★★ | Long context |
| **M5** | `Alibaba-NLP/gte-large-en-v1.5` | 1024 | 8192 | ⚡ | ★★★★ | Plan primary |
| **M6** | `intfloat/e5-small-v2` | 384 | 512 | ⚡⚡⚡ | ★★☆ | E5 family |
| **M7** | `intfloat/e5-base-v2` | 768 | 512 | ⚡⚡ | ★★★ | E5 mid-tier |
| **M8** | `intfloat/e5-large-v2` | 1024 | 512 | ⚡ | ★★★★ | E5 large |
| **M9** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | ⚡⚡⚡ | ★★☆ | Classic lightweight |
| **M10** | `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | ⚡⚡ | ★★★ | Strong general |
| **M11** | `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | ⚡⚡ | ★★★★ | Long + Matryoshka |
| **M12** | `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 512 | ⚡ | ★★★★★ | MTEB leader |
| **M13** | `thenlper/gte-large` | 1024 | 512 | ⚡ | ★★★★ | Original GTE |
| **M14** | `intfloat/multilingual-e5-large` | 1024 | 512 | ⚡ | ★★★★ | If multilingual |

### 1.2 API-Based Models (Fallback/Comparison)

| ID | Model | Dimensions | Context | Cost | Notes |
|----|-------|------------|---------|------|-------|
| **M15** | `text-embedding-3-small` | 1536 | 8191 | $0.02/1M | OpenAI small |
| **M16** | `text-embedding-3-large` | 3072 | 8191 | $0.13/1M | OpenAI large |
| **M17** | `text-embedding-ada-002` | 1536 | 8191 | $0.10/1M | OpenAI legacy |

### 1.3 Sparse Embedding Models

| ID | Model | Type | Notes |
|----|-------|------|-------|
| **S1** | BM25 (rank-bm25) | Sparse | Keyword matching baseline |
| **S2** | SPLADE | Learned sparse | Better than BM25 |
| **S3** | TF-IDF | Sparse | Simple baseline |

---

## Dimension 2: Multi-Representation Strategy

### 2.1 Representation Types (from Implementation Plan Section 2.3)

| ID | Type | Description | Applicable To |
|----|------|-------------|---------------|
| **SUMMARY** | Document summary | Full document semantic summary | PDF, CSV, TXT |
| **CHUNK** | Text chunks | Segmented text windows | PDF, TXT |
| **SCHEMA** | Column structure | Column names + inferred types | CSV, Excel |
| **CONTENT** | Sample content | Sample rows as natural language | CSV, Excel |
| **KEYWORDS** | Keyword extraction | Column names + unique values | CSV, PDF |
| **PAGE** | Per-page embedding | Individual page content | PDF |

### 2.2 Representation Combinations

| ID | Representations | Description | Use Case |
|----|-----------------|-------------|----------|
| **R1** | SUMMARY only | Single embedding per doc | Baseline |
| **R2** | SUMMARY + SCHEMA | Add CSV structure | CSV-heavy data |
| **R3** | SUMMARY + CHUNK | Add text segments | Long PDFs |
| **R4** | SUMMARY + CONTENT | Add sample rows | CSV content |
| **R5** | SCHEMA + CONTENT | CSV-focused (no summary) | Pure tabular |
| **R6** | SUMMARY + SCHEMA + CONTENT | Full CSV representation | Comprehensive CSV |
| **R7** | SUMMARY + CHUNK + KEYWORDS | PDF + sparse | Hybrid PDF |
| **R8** | ALL (SUMMARY + CHUNK + SCHEMA + CONTENT + KEYWORDS) | Maximum coverage | Full exploration |
| **R9** | CHUNK only | Pure chunk-based | No summarization |
| **R10** | PAGE only | Per-page embeddings | Page-level matching |

---

## Dimension 3: Chunking Strategy

### 3.1 Chunking Methods (from Implementation Plan Section 5.2)

| ID | Strategy | Description | Parameters |
|----|----------|-------------|------------|
| **CH1** | Token window | Fixed token windows | size, overlap |
| **CH2** | Semantic | Sentence/paragraph boundaries | min_size, max_size |
| **CH3** | Page-based | PDF page boundaries | pages_per_chunk |
| **CH4** | Recursive | Hierarchical splitting | separators, sizes |

### 3.2 Chunking Parameters

| ID | Strategy | Chunk Size | Overlap | Min Size | Notes |
|----|----------|------------|---------|----------|-------|
| **C1** | Token window | 128 | 12 | 50 | Very small |
| **C2** | Token window | 256 | 25 | 100 | Small |
| **C3** | Token window | 512 | 50 | 100 | Default (plan) |
| **C4** | Token window | 1024 | 100 | 200 | Large |
| **C5** | Token window | 2048 | 200 | 400 | Very large |
| **C6** | Semantic | Variable | N/A | 100 | Sentence-based |
| **C7** | Page-based | 1 page | 0 | N/A | Single page |
| **C8** | Page-based | 2 pages | 1 | N/A | Page pairs |
| **C9** | Recursive | 1000/500/250 | 50 | 100 | Multi-level |

### 3.3 CSV Sampling Parameters

| ID | Max Rows | Max Columns | Include Headers | Notes |
|----|----------|-------------|-----------------|-------|
| **CSV1** | 5 | 20 | Yes | Minimal |
| **CSV2** | 10 | 50 | Yes | Default (plan) |
| **CSV3** | 20 | 100 | Yes | Comprehensive |
| **CSV4** | 50 | All | Yes | Full sample |

---

## Dimension 4: Search Strategy

### 4.1 Search Types

| ID | Type | Description | Notes |
|----|------|-------------|-------|
| **DENSE** | Dense vector search | Cosine similarity on embeddings | Primary |
| **SPARSE** | Sparse vector search | BM25/SPLADE | Keyword matching |
| **HYBRID** | Dense + Sparse fusion | Combined scoring | Best of both |

### 4.2 Hybrid Search Fusion Methods

| ID | Method | Description | Parameters |
|----|--------|-------------|------------|
| **F1** | Linear combination | α * dense + (1-α) * sparse | α ∈ [0,1] |
| **F2** | RRF (Reciprocal Rank Fusion) | 1/(k + rank) fusion | k = 60 |
| **F3** | Normalized score fusion | Normalize then combine | weights |

### 4.3 Hybrid Search Alpha Values (Dense/Sparse Balance)

| ID | Alpha (Dense Weight) | Sparse Weight | Notes |
|----|---------------------|---------------|-------|
| **H1** | 1.0 | 0.0 | Dense only |
| **H2** | 0.9 | 0.1 | Heavy dense |
| **H3** | 0.8 | 0.2 | Default (plan) |
| **H4** | 0.7 | 0.3 | Balanced-dense |
| **H5** | 0.5 | 0.5 | Equal |
| **H6** | 0.3 | 0.7 | Balanced-sparse |
| **H7** | 0.0 | 1.0 | Sparse only |

---

## Dimension 5: Retrieval Configuration

### 5.1 Two-Stage Retrieval (from Implementation Plan Section 5.3)

| ID | Stage 1 (ANN) | Stage 2 (Rerank) | Aggregation | Notes |
|----|---------------|------------------|-------------|-------|
| **K1** | 10 | 5 | 3 | Minimal |
| **K2** | 20 | 10 | 5 | Small |
| **K3** | 50 | 20 | 10 | Default (plan) |
| **K4** | 100 | 50 | 20 | Large |
| **K5** | 200 | 100 | 50 | Very large |
| **K6** | 500 | 200 | 100 | Maximum |

### 5.2 Reranking Strategies

| ID | Method | Description | Notes |
|----|--------|-------------|-------|
| **RR1** | None | No reranking | Baseline |
| **RR2** | Cross-encoder | BERT cross-encoder | Slow but accurate |
| **RR3** | ColBERT | Late interaction | Fast reranking |
| **RR4** | Score normalization | Normalize + re-sort | Simple |

### 5.3 Metadata Filtering (from Implementation Plan Phase 3)

| ID | Filter | Description | Notes |
|----|--------|-------------|-------|
| **MF1** | None | No filtering | Baseline |
| **MF2** | File type | Match query file type | PDF→PDF |
| **MF3** | Representation type | Match rep type | Summary→Summary |
| **MF4** | File type + Rep type | Combined | Strict matching |

---

## Dimension 6: Aggregation Strategy

### 6.1 Class Score Aggregation (from Implementation Plan Section 5.3)

| ID | Method | Formula | Notes |
|----|--------|---------|-------|
| **A1** | Top-1 | Class of nearest neighbor | Simplest |
| **A2** | Max similarity | max(sim) per class | Best match |
| **A3** | Average similarity | mean(sim) per class | Smooth |
| **A4** | Vote count | count per class | Democratic |
| **A5** | Sum similarity | sum(sim) per class | Cumulative |
| **A6** | Weighted combination | w1*max + w2*avg + w3*count | Flexible |

### 6.2 Aggregation Weight Configurations

| ID | Max Weight | Avg Weight | Count Weight | Notes |
|----|------------|------------|--------------|-------|
| **W1** | 1.0 | 0.0 | 0.0 | Max only |
| **W2** | 0.0 | 1.0 | 0.0 | Avg only |
| **W3** | 0.0 | 0.0 | 1.0 | Count only |
| **W4** | 0.5 | 0.3 | 0.2 | Default (plan) |
| **W5** | 0.7 | 0.2 | 0.1 | Heavy max |
| **W6** | 0.4 | 0.4 | 0.2 | Balanced max/avg |
| **W7** | 0.3 | 0.3 | 0.4 | Heavy count |
| **W8** | 0.33 | 0.33 | 0.34 | Equal |

---

## Dimension 7: Late Fusion Strategy

### 7.1 Late Fusion Across Representation Types (from Implementation Plan Section 5.3)

| ID | Summary Weight | Chunk Weight | Sparse Weight | Notes |
|----|----------------|--------------|---------------|-------|
| **LF1** | 1.0 | 0.0 | 0.0 | Summary only |
| **LF2** | 0.0 | 1.0 | 0.0 | Chunks only |
| **LF3** | 0.4 | 0.4 | 0.2 | Default (plan) |
| **LF4** | 0.5 | 0.3 | 0.2 | Heavy summary |
| **LF5** | 0.3 | 0.5 | 0.2 | Heavy chunks |
| **LF6** | 0.3 | 0.3 | 0.4 | Heavy sparse |
| **LF7** | 0.33 | 0.33 | 0.34 | Equal |

### 7.2 Late Fusion Methods

| ID | Method | Description |
|----|--------|-------------|
| **LFM1** | Score averaging | Average class scores across reps |
| **LFM2** | Score max | Max class score across reps |
| **LFM3** | Rank fusion | RRF across representation rankings |
| **LFM4** | Learned weights | Train weights on validation set |

---

## Dimension 8: Confidence & Thresholds

### 8.1 Similarity Thresholds (from Implementation Plan Section 5.3)

| ID | Min Similarity | Description | Expected Abstention |
|----|----------------|-------------|---------------------|
| **T1** | 0.40 | Very permissive | <5% |
| **T2** | 0.50 | Permissive | 5-10% |
| **T3** | 0.60 | Moderate | 10-20% |
| **T4** | 0.65 | Default (plan) | 15-25% |
| **T5** | 0.70 | Conservative | 20-35% |
| **T6** | 0.75 | Strict | 30-45% |
| **T7** | 0.80 | Very strict | 40-60% |

### 8.2 Margin Thresholds (Gap Between Top 2 Classes)

| ID | Min Margin | Description |
|----|------------|-------------|
| **MG1** | 0.00 | No margin required |
| **MG2** | 0.05 | Small margin |
| **MG3** | 0.10 | Default (plan) |
| **MG4** | 0.15 | Moderate margin |
| **MG5** | 0.20 | Large margin |
| **MG6** | 0.30 | Very large margin |

### 8.3 Confidence Calibration Methods

| ID | Method | Description |
|----|--------|-------------|
| **CC1** | None | Raw scores |
| **CC2** | Temperature scaling | Softmax with temperature |
| **CC3** | Platt scaling | Logistic calibration |
| **CC4** | Isotonic regression | Non-parametric |

---

## Dimension 9: Class Prototype Strategy

### 9.1 Class Prototype Methods (from Implementation Plan Phase 4)

| ID | Method | Description |
|----|--------|-------------|
| **CP1** | None | No prototypes |
| **CP2** | Centroid | Mean of class embeddings |
| **CP3** | Medoid | Most central embedding |
| **CP4** | Multiple centroids | K-means within class |

### 9.2 Prototype Usage

| ID | Usage | Description |
|----|-------|-------------|
| **PU1** | Retrieval only | Use prototypes in search |
| **PU2** | Tiebreaker | Use when classes tied |
| **PU3** | Ensemble | Combine with retrieval |
| **PU4** | Fallback | Use when low confidence |

---

## Dimension 10: Storage & Quantization

### 10.1 Vector Quantization

| ID | Method | Compression | Quality Loss |
|----|--------|-------------|--------------|
| **Q1** | None (float32) | 1x | None |
| **Q2** | Float16 | 2x | Minimal |
| **Q3** | Scalar (INT8) | 4x | Small |
| **Q4** | Binary | 32x | Moderate |
| **Q5** | Product Quantization | 8-32x | Moderate |

### 10.2 HNSW Index Parameters

| ID | M | ef_construct | ef_search | Notes |
|----|---|--------------|-----------|-------|
| **IX1** | 8 | 50 | 50 | Fast, lower quality |
| **IX2** | 16 | 100 | 100 | Default (plan) |
| **IX3** | 32 | 200 | 200 | High quality |
| **IX4** | 64 | 400 | 400 | Maximum quality |

---

## Recommended Experiment Sequence

### Stage 1: Model Selection (8-14 experiments)
**Goal:** Find best embedding model  
**Fixed:** R1 (summary), A1 (top-1), K3 (50/20/10), T4 (0.65)

```
Run: M1, M2, M3, M4, M5, M7, M8, M10, M11, M12
Pick: Best 2-3 models for further testing
```

### Stage 2: Representation Strategy (6-10 experiments)
**Goal:** Find best representation combination  
**Fixed:** Best model, A4 (weighted), K3

```
Run: R1, R2, R3, R6, R7, R8
Pick: Best representation strategy
```

### Stage 3: Chunking Optimization (5-9 experiments)
**Goal:** Optimize chunk parameters (if chunks help)  
**Fixed:** Best model + representation

```
Run: C1, C2, C3, C4, C6, C7
Pick: Best chunking config
```

### Stage 4: Search Strategy (5-7 experiments)
**Goal:** Test hybrid search  
**Fixed:** Best model + rep + chunking

```
Run: H1, H3, H4, H5, H7 (with S1 sparse)
Pick: Best hybrid config
```

### Stage 5: Retrieval Depth (5-6 experiments)
**Goal:** Optimize top-k  
**Fixed:** Best config so far

```
Run: K1, K2, K3, K4, K5
Pick: Best retrieval depth
```

### Stage 6: Aggregation Strategy (6-8 experiments)
**Goal:** Optimize class scoring  
**Fixed:** Best config so far

```
Run: A1, A2, A3, A4, A6 with W4, W5, W6
Pick: Best aggregation
```

### Stage 7: Late Fusion (5-7 experiments)
**Goal:** Optimize representation fusion  
**Fixed:** Best config so far

```
Run: LF1, LF3, LF4, LF5, LF7
Pick: Best fusion weights
```

### Stage 8: Threshold Tuning (6-8 experiments)
**Goal:** Optimize confidence thresholds  
**Fixed:** Best config so far

```
Run: T2, T3, T4, T5, T6 with MG2, MG3, MG4
Pick: Best threshold config
```

### Stage 9: Advanced Features (4-6 experiments)
**Goal:** Test advanced strategies  
**Fixed:** Best config so far

```
Run: Class prototypes (CP2, CP3), Reranking (RR2), Quantization (Q2, Q3)
Pick: Features that improve without hurting speed
```

### Stage 10: Final Validation
**Goal:** Validate on holdout  
**Run:** Best config on DURECO.zip holdout set

---

## Evaluation Metrics (from Implementation Plan Section 6)

### Retrieval Metrics
- **Recall@k** (k=5, 10, 20): Proportion of correct class in top-k
- **MRR**: Mean Reciprocal Rank
- **NDCG@k**: Normalized Discounted Cumulative Gain

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Macro F1**: Average F1 across all classes
- **Weighted F1**: F1 weighted by class frequency
- **Confusion Matrix**: Per-class analysis

### Abstention Metrics
- **Abstention Rate**: % of predictions abstained
- **Accuracy on Predictions**: Accuracy excluding abstentions
- **Coverage**: 1 - abstention rate

### Efficiency Metrics
- **Indexing Time**: Time to index training data
- **Query Latency**: Time per inference (p50, p95, p99)
- **Storage Size**: Vector DB size on disk
- **Memory Usage**: Peak RAM during inference

---

## Leaderboard Template

| Rank | Config ID | Model | Representations | Aggregation | Top-K | Accuracy | Macro F1 | MRR | Latency (ms) |
|------|-----------|-------|-----------------|-------------|-------|----------|----------|-----|--------------|
| 1 | EXP-042 | M12 | R6 | W4 | K3 | 0.923 | 0.891 | 0.945 | 45 |
| 2 | EXP-038 | M5 | R6 | W5 | K3 | 0.918 | 0.885 | 0.941 | 52 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## Configuration File Format

```yaml
experiment:
  id: "EXP-001"
  name: "Baseline BGE-large summary-only"
  
model:
  name: "BAAI/bge-large-en-v1.5"
  device: "cuda"
  batch_size: 32
  
representations:
  - type: "summary"
    enabled: true
  - type: "chunk"
    enabled: false
  - type: "schema"
    enabled: false
    
chunking:
  strategy: "token_window"
  chunk_size: 512
  overlap: 50
  min_size: 100
  
search:
  type: "dense"  # dense, sparse, hybrid
  hybrid_alpha: 0.8
  
retrieval:
  top_k_stage1: 50
  top_k_stage2: 20
  top_k_aggregation: 10
  reranker: null
  
aggregation:
  method: "weighted"
  weights:
    max_sim: 0.5
    avg_sim: 0.3
    count: 0.2
    
late_fusion:
  weights:
    summary: 0.4
    chunk: 0.4
    sparse: 0.2
    
thresholds:
  min_similarity: 0.65
  min_margin: 0.10
  
quantization:
  embeddings: "float16"
  qdrant: "scalar_int8"
```

---

*Document Version: 1.0 | Last Updated: April 12, 2026*
