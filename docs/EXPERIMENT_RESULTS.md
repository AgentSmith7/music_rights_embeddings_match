# Experiment Results - Music Rights Document Classification

## Leaderboard (1000-sample validation set)

| Rank | Experiment | Accuracy | Macro Acc | Predictions | Notes |
|------|------------|----------|-----------|-------------|-------|
| 1 | **thresh_margin_010** | **99.87%** | 99.50% | 796/1000 | Min margin 0.10 between top-2 classes |
| 2 | **thresh_combined** | **99.51%** | 97.61% | 814/1000 | Min similarity 0.65 AND min margin 0.10 |
| 3 | **best_weighted_k20_thresh** | **98.91%** | 96.73% | 828/1000 | Weighted + top-20 + thresholds |
| 4 | agg_weighted | 92.39% | 82.72% | 985/1000 | Weighted: 0.5*max + 0.3*avg + 0.2*count |
| 5 | topk_20 | 92.26% | 89.36% | 982/1000 | Top-20 retrieval with mean aggregation |
| 6 | agg_max | 92.10% | 87.61% | 975/1000 | Max similarity per class |
| 7 | agg_count | 90.72% | 80.53% | 981/1000 | Vote count per class |
| 8 | topk_5 | 90.52% | 86.49% | 981/1000 | Top-5 retrieval with mean aggregation |
| 9 | agg_weighted_heavy_max | 90.50% | 85.22% | 979/1000 | Weighted: 0.7*max + 0.2*avg + 0.1*count |
| 10 | thresh_sim_070 | 90.40% | 87.58% | 979/1000 | Min similarity 0.70, abstain if below |
| 11 | agg_sum | 90.27% | 81.91% | 987/1000 | Sum of similarities per class |
| 12 | topk_50 | 90.08% | 84.84% | 988/1000 | Top-50 retrieval with mean aggregation |
| 13 | **baseline** | **89.93%** | 86.36% | 983/1000 | Mean aggregation, top-10, no thresholds |
| 14 | thresh_sim_065 | 89.86% | 87.94% | 976/1000 | Min similarity 0.65, abstain if below |
| 15 | best_weighted_k20 | 88.67% | 76.31% | 980/1000 | Weighted aggregation with top-20 |

## Key Findings

### 1. Confidence Thresholds Dramatically Improve Accuracy
- **thresh_margin_010** achieves 99.87% accuracy by only predicting when confident
- Trade-off: ~20% abstention rate (only 796/1000 predictions)
- For high-stakes applications, this is ideal

### 2. Aggregation Methods Matter
- **agg_weighted** (92.39%) beats baseline (89.93%) by +2.46%
- **agg_max** (92.10%) is nearly as good and simpler
- **agg_sum** and **agg_count** underperform

### 3. Top-K Sweet Spot
- **topk_20** (92.26%) performs best among top-k variations
- topk_5 is too restrictive, topk_50 adds noise
- Default top-10 is reasonable but not optimal

### 4. Best Configurations for Different Use Cases

#### High Accuracy (with abstention allowed)
- Use **thresh_margin_010** for 99.87% accuracy
- Flag uncertain predictions for human review

#### Balanced (no abstention)
- Use **agg_weighted** for 92.39% accuracy on all samples
- Or **topk_20** for 92.26% with better macro accuracy

#### Production Recommendation
- Use **thresh_combined** (99.51% accuracy, 81.4% coverage)
- For remaining 18.6%, use **agg_weighted** as fallback

## Experiment Details

### Model Configuration
- **Embedding Model**: BAAI/bge-large-en-v1.5
- **Vector DB**: Qdrant with INT8 scalar quantization
- **Representation**: Full content summary (compact fingerprints)
- **Training Set**: 64,912 documents
- **Validation Set**: 16,716 documents (1,000 sampled for experiments)

### Aggregation Methods Tested
1. **mean**: Average similarity scores per class
2. **max**: Maximum similarity score per class
3. **sum**: Sum of similarity scores per class
4. **count**: Number of neighbors per class
5. **weighted**: 0.5*max + 0.3*avg + 0.2*count

### Threshold Configurations
- **thresh_sim_065**: Abstain if top score < 0.65
- **thresh_sim_070**: Abstain if top score < 0.70
- **thresh_margin_010**: Abstain if margin < 0.10
- **thresh_combined**: Both similarity and margin thresholds

## Next Steps
1. Apply best configuration to DURECO holdout set
2. Generate predictions with confidence scores
3. Flag low-confidence predictions for review
