#!/usr/bin/env python3
"""
Experiment Runner for Music Rights Document Classification

Tests different aggregation strategies, retrieval configs, and thresholds
on the validation set to find optimal configuration before DURECO inference.

Builds a leaderboard comparing all experiments.
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime
import random

import numpy as np
from tqdm import tqdm
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    
    # Retrieval
    top_k: int = 10
    
    # Aggregation method: "mean", "max", "sum", "count", "weighted"
    aggregation: str = "mean"
    
    # Weights for weighted aggregation (max, avg, count)
    weight_max: float = 0.5
    weight_avg: float = 0.3
    weight_count: float = 0.2
    
    # Confidence thresholds
    min_similarity: float = 0.0  # 0 = no threshold
    min_margin: float = 0.0      # 0 = no margin requirement
    
    # Filtering
    filter_file_type: bool = False  # Match query file type
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config_name: str
    accuracy: float
    macro_accuracy: float
    correct: int
    total: int
    abstained: int
    abstention_rate: float
    accuracy_on_predictions: float  # Accuracy excluding abstentions
    
    # Per-class metrics
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    duration_seconds: float = 0.0
    
    # Confidence stats
    avg_confidence: float = 0.0
    avg_margin: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# PREDEFINED EXPERIMENTS
# =============================================================================

EXPERIMENTS = {
    # Baseline (current)
    "baseline": ExperimentConfig(
        name="baseline",
        description="Current baseline: mean aggregation, top-10, no thresholds",
        top_k=10,
        aggregation="mean",
    ),
    
    # Aggregation methods
    "agg_max": ExperimentConfig(
        name="agg_max",
        description="Max similarity per class",
        top_k=10,
        aggregation="max",
    ),
    "agg_sum": ExperimentConfig(
        name="agg_sum",
        description="Sum of similarities per class",
        top_k=10,
        aggregation="sum",
    ),
    "agg_count": ExperimentConfig(
        name="agg_count",
        description="Vote count per class",
        top_k=10,
        aggregation="count",
    ),
    "agg_weighted": ExperimentConfig(
        name="agg_weighted",
        description="Weighted: 0.5*max + 0.3*avg + 0.2*count",
        top_k=10,
        aggregation="weighted",
        weight_max=0.5,
        weight_avg=0.3,
        weight_count=0.2,
    ),
    "agg_weighted_heavy_max": ExperimentConfig(
        name="agg_weighted_heavy_max",
        description="Weighted: 0.7*max + 0.2*avg + 0.1*count",
        top_k=10,
        aggregation="weighted",
        weight_max=0.7,
        weight_avg=0.2,
        weight_count=0.1,
    ),
    
    # Top-k variations
    "topk_5": ExperimentConfig(
        name="topk_5",
        description="Top-5 retrieval with mean aggregation",
        top_k=5,
        aggregation="mean",
    ),
    "topk_20": ExperimentConfig(
        name="topk_20",
        description="Top-20 retrieval with mean aggregation",
        top_k=20,
        aggregation="mean",
    ),
    "topk_50": ExperimentConfig(
        name="topk_50",
        description="Top-50 retrieval with mean aggregation",
        top_k=50,
        aggregation="mean",
    ),
    
    # Confidence thresholds
    "thresh_sim_065": ExperimentConfig(
        name="thresh_sim_065",
        description="Min similarity 0.65, abstain if below",
        top_k=10,
        aggregation="mean",
        min_similarity=0.65,
    ),
    "thresh_sim_070": ExperimentConfig(
        name="thresh_sim_070",
        description="Min similarity 0.70, abstain if below",
        top_k=10,
        aggregation="mean",
        min_similarity=0.70,
    ),
    "thresh_margin_010": ExperimentConfig(
        name="thresh_margin_010",
        description="Min margin 0.10 between top-2 classes",
        top_k=10,
        aggregation="mean",
        min_margin=0.10,
    ),
    "thresh_combined": ExperimentConfig(
        name="thresh_combined",
        description="Min similarity 0.65 AND min margin 0.10",
        top_k=10,
        aggregation="mean",
        min_similarity=0.65,
        min_margin=0.10,
    ),
    
    # Combined best practices
    "best_weighted_k20": ExperimentConfig(
        name="best_weighted_k20",
        description="Weighted aggregation with top-20",
        top_k=20,
        aggregation="weighted",
        weight_max=0.5,
        weight_avg=0.3,
        weight_count=0.2,
    ),
    "best_weighted_k20_thresh": ExperimentConfig(
        name="best_weighted_k20_thresh",
        description="Weighted + top-20 + thresholds",
        top_k=20,
        aggregation="weighted",
        weight_max=0.5,
        weight_avg=0.3,
        weight_count=0.2,
        min_similarity=0.65,
        min_margin=0.10,
    ),
}


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_scores(
    results: List[dict],
    config: ExperimentConfig
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Aggregate retrieval results into class scores.
    
    Returns:
        class_scores: Dict mapping class -> score
        details: Dict with aggregation details for analysis
    """
    # Group by class
    class_results = defaultdict(list)
    for r in results:
        class_label = r["payload"]["class_label"]
        class_results[class_label].append(r["score"])
    
    class_scores = {}
    details = {"per_class": {}}
    
    for cls, scores in class_results.items():
        max_score = max(scores)
        avg_score = np.mean(scores)
        count = len(scores)
        sum_score = sum(scores)
        
        if config.aggregation == "mean":
            class_scores[cls] = avg_score
        elif config.aggregation == "max":
            class_scores[cls] = max_score
        elif config.aggregation == "sum":
            class_scores[cls] = sum_score
        elif config.aggregation == "count":
            class_scores[cls] = count
        elif config.aggregation == "weighted":
            # Normalize count to [0, 1] range based on top_k
            norm_count = count / config.top_k
            class_scores[cls] = (
                config.weight_max * max_score +
                config.weight_avg * avg_score +
                config.weight_count * norm_count
            )
        else:
            class_scores[cls] = avg_score  # Default to mean
        
        details["per_class"][cls] = {
            "max": max_score,
            "avg": avg_score,
            "count": count,
            "final_score": class_scores[cls]
        }
    
    return class_scores, details


def apply_thresholds(
    class_scores: Dict[str, float],
    config: ExperimentConfig
) -> Tuple[Optional[str], float, float, bool]:
    """
    Apply confidence thresholds and return prediction.
    
    Returns:
        predicted_class: Class label or None if abstained
        confidence: Top class score
        margin: Gap between top-2 classes
        abstained: Whether prediction was abstained
    """
    if not class_scores:
        return None, 0.0, 0.0, True
    
    # Sort by score
    sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_class, top_score = sorted_classes[0]
    second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
    margin = top_score - second_score
    
    # Check thresholds
    abstained = False
    
    if config.min_similarity > 0 and top_score < config.min_similarity:
        abstained = True
    
    if config.min_margin > 0 and margin < config.min_margin:
        abstained = True
    
    if abstained:
        return None, top_score, margin, True
    
    return top_class, top_score, margin, False


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Runs experiments on validation set and builds leaderboard."""
    
    def __init__(self, qdrant_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.qdrant_path = qdrant_path
        self.embedding_model = embedding_model
        self.embedder = None
        self.qdrant_client = None
        self.collection_name = "music_rights"
        self.results: List[ExperimentResult] = []
    
    def _load_embedder(self):
        """Lazy load embedding model."""
        if self.embedder is not None:
            return
        
        logger.info(f"Loading embedding model: {self.embedding_model}")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model, device="cuda")
        logger.info("Model loaded on GPU")
    
    def _load_qdrant(self):
        """Lazy load Qdrant client."""
        if self.qdrant_client is not None:
            return
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_path}")
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        logger.info("Qdrant connected")
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        self._load_embedder()
        embedding = self.embedder.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float16)
    
    def _search(self, vector: np.ndarray, top_k: int, file_type_filter: Optional[str] = None) -> List[dict]:
        """Search Qdrant for similar documents."""
        self._load_qdrant()
        
        # Build filter if needed
        query_filter = None
        if file_type_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="file_type", match=MatchValue(value=file_type_filter))]
            )
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]
    
    def _parse_file(self, file_path: str) -> str:
        """Parse file into semantic representation."""
        # Import parser from streaming_pipeline
        from streaming_pipeline import parse_single_file
        
        ext = Path(file_path).suffix.lower()
        result = parse_single_file((
            file_path, "unknown", ext,
            4096,  # max_text_length
            50,    # max_rows_sample
            30     # max_cols_sample
        ))
        
        if result["parser_status"] != "ok":
            return ""
        
        return result["representation_text"]
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        val_files: List[Tuple[str, str, str]],
        max_samples: Optional[int] = None
    ) -> ExperimentResult:
        """
        Run a single experiment on validation files.
        
        Args:
            config: Experiment configuration
            val_files: List of (file_path, true_label, extension)
            max_samples: Limit samples for quick testing
        """
        logger.info(f"Running experiment: {config.name}")
        logger.info(f"  Description: {config.description}")
        
        start_time = time.time()
        
        # Sample if needed
        files_to_test = val_files
        if max_samples and max_samples < len(val_files):
            files_to_test = random.sample(val_files, max_samples)
        
        correct = 0
        total = 0
        abstained = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        confidences = []
        margins = []
        
        for fpath, true_label, ext in tqdm(files_to_test, desc=f"Exp: {config.name}"):
            # Parse
            text = self._parse_file(fpath)
            if not text:
                continue
            
            # Embed
            vector = self._embed(text)
            
            # Search
            file_type_filter = ext if config.filter_file_type else None
            results = self._search(vector, config.top_k, file_type_filter)
            
            if not results:
                abstained += 1
                continue
            
            # Aggregate
            class_scores, _ = aggregate_scores(results, config)
            
            # Apply thresholds
            predicted, confidence, margin, did_abstain = apply_thresholds(class_scores, config)
            
            confidences.append(confidence)
            margins.append(margin)
            
            if did_abstain:
                abstained += 1
                continue
            
            # Evaluate
            total += 1
            class_total[true_label] += 1
            
            if predicted == true_label:
                correct += 1
                class_correct[true_label] += 1
        
        duration = time.time() - start_time
        
        # Compute metrics
        accuracy = correct / total if total > 0 else 0
        abstention_rate = abstained / len(files_to_test) if files_to_test else 0
        accuracy_on_predictions = correct / total if total > 0 else 0
        
        per_class_acc = {}
        for cls in class_total:
            per_class_acc[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        
        macro_accuracy = np.mean(list(per_class_acc.values())) if per_class_acc else 0
        
        result = ExperimentResult(
            config_name=config.name,
            accuracy=accuracy,
            macro_accuracy=macro_accuracy,
            correct=correct,
            total=total,
            abstained=abstained,
            abstention_rate=abstention_rate,
            accuracy_on_predictions=accuracy_on_predictions,
            per_class_accuracy=per_class_acc,
            duration_seconds=duration,
            avg_confidence=np.mean(confidences) if confidences else 0,
            avg_margin=np.mean(margins) if margins else 0,
        )
        
        self.results.append(result)
        
        logger.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        logger.info(f"  Macro Accuracy: {macro_accuracy:.4f}")
        logger.info(f"  Abstention Rate: {abstention_rate:.2%}")
        logger.info(f"  Duration: {duration:.1f}s")
        
        return result
    
    def run_all_experiments(
        self,
        val_files: List[Tuple[str, str, str]],
        experiment_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ):
        """Run multiple experiments."""
        experiments_to_run = experiment_names or list(EXPERIMENTS.keys())
        
        for exp_name in experiments_to_run:
            if exp_name not in EXPERIMENTS:
                logger.warning(f"Unknown experiment: {exp_name}")
                continue
            
            config = EXPERIMENTS[exp_name]
            self.run_experiment(config, val_files, max_samples)
    
    def get_leaderboard(self) -> str:
        """Generate leaderboard as formatted string."""
        if not self.results:
            return "No results yet."
        
        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x.accuracy, reverse=True)
        
        lines = []
        lines.append("=" * 100)
        lines.append("EXPERIMENT LEADERBOARD")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Rank':<5} {'Experiment':<30} {'Accuracy':<10} {'Macro':<10} {'Abstain%':<10} {'Acc(pred)':<10} {'Conf':<8} {'Margin':<8}")
        lines.append("-" * 100)
        
        for i, r in enumerate(sorted_results, 1):
            lines.append(
                f"{i:<5} {r.config_name:<30} {r.accuracy:.4f}    {r.macro_accuracy:.4f}    "
                f"{r.abstention_rate:.2%}     {r.accuracy_on_predictions:.4f}     "
                f"{r.avg_confidence:.3f}   {r.avg_margin:.3f}"
            )
        
        lines.append("-" * 100)
        lines.append("")
        lines.append("Legend:")
        lines.append("  Accuracy   = Correct / Total predictions (excluding abstentions)")
        lines.append("  Macro      = Average per-class accuracy")
        lines.append("  Abstain%   = Percentage of samples where model abstained")
        lines.append("  Acc(pred)  = Accuracy on non-abstained predictions")
        lines.append("  Conf       = Average confidence score of top prediction")
        lines.append("  Margin     = Average gap between top-2 class scores")
        lines.append("")
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "experiments": [r.to_dict() for r in self.results],
            "leaderboard": self.get_leaderboard()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


# =============================================================================
# FILE ENUMERATION
# =============================================================================

def enumerate_files(base_path: str) -> List[Tuple[str, str, str]]:
    """Enumerate all files with (path, class_label, extension)."""
    files = []
    base = Path(base_path)
    
    for root, _, filenames in os.walk(base):
        for fname in filenames:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            
            if ext not in ['.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab']:
                continue
            
            parts = fpath.parts
            try:
                stmt_idx = parts.index('Statement')
                class_label = parts[stmt_idx + 1]
            except (ValueError, IndexError):
                class_label = "unknown"
            
            files.append((str(fpath), class_label, ext))
    
    return files


def stratified_split(files: List[Tuple[str, str, str]], train_ratio: float = 0.8, seed: int = 42):
    """Stratified train/val split."""
    random.seed(seed)
    
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
    
    return train_files, val_files


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run experiments on validation set")
    parser.add_argument("--data-path", default="/workspace/training_data/extracted/Fast/TrainData/RYLTY/Organizer/Statement")
    parser.add_argument("--qdrant-path", default="/workspace/qdrant_data")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    parser.add_argument("--max-samples", type=int, help="Limit samples for quick testing")
    parser.add_argument("--output", default="/workspace/experiment_results.json")
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments")
    args = parser.parse_args()
    
    if args.list_experiments:
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<25} {config.description}")
        print()
        return
    
    # Enumerate and split files
    logger.info("Enumerating files...")
    all_files = enumerate_files(args.data_path)
    logger.info(f"Found {len(all_files)} files")
    
    _, val_files = stratified_split(all_files, train_ratio=0.8, seed=42)
    logger.info(f"Validation set: {len(val_files)} files")
    
    # Run experiments
    runner = ExperimentRunner(args.qdrant_path)
    runner.run_all_experiments(
        val_files,
        experiment_names=args.experiments,
        max_samples=args.max_samples
    )
    
    # Print leaderboard
    print("\n" + runner.get_leaderboard())
    
    # Save results
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
