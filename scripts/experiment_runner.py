#!/usr/bin/env python3
"""
Experiment runner for comparing different embedding models, chunking strategies, and configurations.

Creates a train/validation split and maintains a leaderboard of results.
"""

import os
import sys
import json
import time
import uuid
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Constants
EXTRACT_PATH = "/workspace/training_data/extracted"
STATEMENT_PATH = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
QDRANT_BASE_PATH = "/workspace/qdrant_experiments"
RESULTS_PATH = "/workspace/results"
LEADERBOARD_PATH = os.path.join(RESULTS_PATH, "leaderboard.json")

# Embedding models to test
EMBEDDING_MODELS = [
    {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "trust_remote": False},
    {"name": "BAAI/bge-base-en-v1.5", "dim": 768, "trust_remote": False},
    {"name": "sentence-transformers/all-mpnet-base-v2", "dim": 768, "trust_remote": False},
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "trust_remote": False},
    {"name": "intfloat/e5-large-v2", "dim": 1024, "trust_remote": False},
    {"name": "intfloat/e5-base-v2", "dim": 768, "trust_remote": False},
]

# Chunking strategies
CHUNKING_STRATEGIES = [
    {"name": "schema_content", "use_schema": True, "use_content": True, "use_summary": False},
    {"name": "schema_only", "use_schema": True, "use_content": False, "use_summary": False},
    {"name": "content_only", "use_schema": False, "use_content": True, "use_summary": False},
    {"name": "all_representations", "use_schema": True, "use_content": True, "use_summary": True},
]

# Aggregation strategies
AGGREGATION_CONFIGS = [
    {"name": "balanced", "weight_max": 0.4, "weight_avg": 0.3, "weight_count": 0.3},
    {"name": "max_focused", "weight_max": 0.6, "weight_avg": 0.2, "weight_count": 0.2},
    {"name": "count_focused", "weight_max": 0.3, "weight_avg": 0.2, "weight_count": 0.5},
]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    model_name: str
    model_dim: int
    chunking_strategy: str
    use_schema: bool
    use_content: bool
    use_summary: bool
    aggregation_name: str
    weight_max: float
    weight_avg: float
    weight_count: float
    top_k: int = 20
    validation_split: float = 0.2


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: str
    config: Dict
    accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    macro_f1: float
    avg_confidence: float
    abstention_rate: float
    indexing_time_sec: float
    inference_time_sec: float
    total_train_docs: int
    total_val_docs: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def create_train_val_split(validation_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create train/validation split stratified by class.
    Returns dict mapping class_name -> list of file paths.
    """
    random.seed(seed)
    
    if not os.path.exists(STATEMENT_PATH):
        raise ValueError(f"Statement path not found: {STATEMENT_PATH}")
    
    train_split = {}
    val_split = {}
    
    class_folders = os.listdir(STATEMENT_PATH)
    logger.info(f"Creating train/val split for {len(class_folders)} classes")
    
    for class_name in class_folders:
        class_path = os.path.join(STATEMENT_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(files)
        
        val_size = max(1, int(len(files) * validation_ratio))
        
        val_files = files[:val_size]
        train_files = files[val_size:]
        
        train_split[class_name] = [os.path.join(class_path, f) for f in train_files]
        val_split[class_name] = [os.path.join(class_path, f) for f in val_files]
    
    total_train = sum(len(v) for v in train_split.values())
    total_val = sum(len(v) for v in val_split.values())
    logger.info(f"Split: {total_train} train, {total_val} validation")
    
    return train_split, val_split


def parse_csv_file(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse CSV and return representations based on config."""
    representations = []
    
    try:
        df = pd.read_csv(filepath, nrows=100, encoding='utf-8', on_bad_lines='skip')
        if df.empty:
            return representations
        
        # Schema representation
        if config.use_schema:
            schema_parts = []
            for col in df.columns[:30]:
                dtype = str(df[col].dtype)
                sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
                schema_parts.append(f"{col} ({dtype}): {sample[:50]}")
            schema_text = "CSV Schema:\n" + "\n".join(schema_parts)
            representations.append(("schema", schema_text))
        
        # Content representation
        if config.use_content:
            content_parts = []
            for idx, row in df.head(5).iterrows():
                row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                content_parts.append(row_text[:500])
            content_text = "Sample data:\n" + "\n".join(content_parts)
            representations.append(("content", content_text))
        
        # Summary representation
        if config.use_summary:
            summary = f"CSV with {len(df.columns)} columns: {', '.join(df.columns[:10])}"
            representations.append(("summary", summary))
            
    except Exception as e:
        pass
    
    return representations


def parse_document(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse document based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.csv']:
        return parse_csv_file(filepath, config)
    elif ext in ['.txt', '.text']:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)
            if content.strip():
                reps = []
                if config.use_content:
                    reps.append(("content", content))
                if config.use_summary:
                    reps.append(("summary", content[:500]))
                return reps
        except:
            pass
    return []


def run_experiment(config: ExperimentConfig, train_split: Dict, val_split: Dict) -> ExperimentResult:
    """Run a single experiment with given configuration."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {config.experiment_id}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Chunking: {config.chunking_strategy}")
    logger.info(f"Aggregation: {config.aggregation_name}")
    logger.info(f"{'='*60}")
    
    # Setup Qdrant for this experiment
    qdrant_path = os.path.join(QDRANT_BASE_PATH, config.experiment_id)
    os.makedirs(qdrant_path, exist_ok=True)
    
    client = QdrantClient(path=qdrant_path)
    collection_name = "experiment"
    
    # Create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=config.model_dim, distance=Distance.COSINE)
    )
    
    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = SentenceTransformer(config.model_name, device='cuda')
    
    # Index training documents
    logger.info("Indexing training documents...")
    start_index = time.time()
    
    batch_texts = []
    batch_payloads = []
    total_indexed = 0
    
    for class_name, files in tqdm(train_split.items(), desc="Indexing"):
        for filepath in files:
            representations = parse_document(filepath, config)
            
            for rep_type, text in representations:
                if text and len(text.strip()) >= 10:
                    batch_texts.append(text)
                    batch_payloads.append({
                        "class_label": class_name,
                        "file_path": filepath,
                        "rep_type": rep_type
                    })
                    
                    if len(batch_texts) >= 64:
                        embeddings = model.encode(batch_texts, show_progress_bar=False)
                        points = [
                            PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload=pay)
                            for emb, pay in zip(embeddings, batch_payloads)
                        ]
                        client.upsert(collection_name=collection_name, points=points)
                        total_indexed += len(batch_texts)
                        batch_texts = []
                        batch_payloads = []
    
    # Final batch
    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload=pay)
            for emb, pay in zip(embeddings, batch_payloads)
        ]
        client.upsert(collection_name=collection_name, points=points)
        total_indexed += len(batch_texts)
    
    indexing_time = time.time() - start_index
    logger.info(f"Indexed {total_indexed} vectors in {indexing_time:.1f}s")
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    start_eval = time.time()
    
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    confidences = []
    abstentions = 0
    
    class_predictions = defaultdict(list)
    
    for class_name, files in tqdm(val_split.items(), desc="Evaluating"):
        for filepath in files:
            representations = parse_document(filepath, config)
            
            if not representations:
                abstentions += 1
                total += 1
                continue
            
            # Get embeddings for all representations
            texts = [text for _, text in representations]
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Query and aggregate
            all_matches = []
            for emb in embeddings:
                results = client.query_points(
                    collection_name=collection_name,
                    query=emb.tolist(),
                    limit=config.top_k
                ).points
                for r in results:
                    all_matches.append({"score": r.score, "class": r.payload["class_label"]})
            
            if not all_matches:
                abstentions += 1
                total += 1
                continue
            
            # Aggregate scores
            class_scores = aggregate_scores(all_matches, config)
            sorted_classes = sorted(class_scores.items(), key=lambda x: -x[1])
            
            predicted = sorted_classes[0][0]
            confidence = sorted_classes[0][1]
            confidences.append(confidence)
            
            # Check accuracy
            total += 1
            if predicted == class_name:
                correct += 1
            
            top3_classes = [c for c, _ in sorted_classes[:3]]
            if class_name in top3_classes:
                correct_top3 += 1
            
            top5_classes = [c for c, _ in sorted_classes[:5]]
            if class_name in top5_classes:
                correct_top5 += 1
            
            class_predictions[class_name].append(predicted)
    
    inference_time = time.time() - start_eval
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    top3_accuracy = correct_top3 / total if total > 0 else 0
    top5_accuracy = correct_top5 / total if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    abstention_rate = abstentions / total if total > 0 else 0
    
    # Calculate macro F1
    macro_f1 = calculate_macro_f1(class_predictions, val_split.keys())
    
    result = ExperimentResult(
        experiment_id=config.experiment_id,
        config=asdict(config),
        accuracy=accuracy,
        top3_accuracy=top3_accuracy,
        top5_accuracy=top5_accuracy,
        macro_f1=macro_f1,
        avg_confidence=avg_confidence,
        abstention_rate=abstention_rate,
        indexing_time_sec=indexing_time,
        inference_time_sec=inference_time,
        total_train_docs=sum(len(v) for v in train_split.values()),
        total_val_docs=total
    )
    
    logger.info(f"\nResults:")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Top-3 Accuracy: {top3_accuracy*100:.2f}%")
    logger.info(f"  Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Avg Confidence: {avg_confidence:.4f}")
    logger.info(f"  Abstention Rate: {abstention_rate*100:.2f}%")
    
    # Cleanup
    del model
    import gc
    gc.collect()
    
    return result


def aggregate_scores(matches: List[Dict], config: ExperimentConfig) -> Dict[str, float]:
    """Aggregate match scores by class."""
    class_sims = defaultdict(list)
    for m in matches:
        class_sims[m["class"]].append(m["score"])
    
    max_count = max(len(v) for v in class_sims.values()) if class_sims else 1
    
    scores = {}
    for cls, sims in class_sims.items():
        max_sim = max(sims)
        avg_sim = np.mean(sims)
        count_norm = len(sims) / max_count
        
        scores[cls] = (
            config.weight_max * max_sim +
            config.weight_avg * avg_sim +
            config.weight_count * count_norm
        )
    
    return scores


def calculate_macro_f1(predictions: Dict[str, List[str]], classes: List[str]) -> float:
    """Calculate macro F1 score."""
    f1_scores = []
    
    for cls in classes:
        if cls not in predictions:
            continue
        
        preds = predictions[cls]
        tp = sum(1 for p in preds if p == cls)
        fp = sum(1 for other_cls, other_preds in predictions.items() 
                 if other_cls != cls for p in other_preds if p == cls)
        fn = sum(1 for p in preds if p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0


def update_leaderboard(result: ExperimentResult):
    """Update the leaderboard with new result."""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []
    
    leaderboard.append(asdict(result))
    
    # Sort by accuracy
    leaderboard.sort(key=lambda x: -x['accuracy'])
    
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    logger.info(f"Leaderboard updated: {LEADERBOARD_PATH}")


def print_leaderboard():
    """Print current leaderboard."""
    if not os.path.exists(LEADERBOARD_PATH):
        logger.info("No leaderboard yet")
        return
    
    with open(LEADERBOARD_PATH, 'r') as f:
        leaderboard = json.load(f)
    
    print("\n" + "="*100)
    print("LEADERBOARD")
    print("="*100)
    print(f"{'Rank':<5} {'Model':<35} {'Chunking':<20} {'Accuracy':<10} {'Top3':<10} {'F1':<10}")
    print("-"*100)
    
    for i, entry in enumerate(leaderboard[:20], 1):
        model = entry['config']['model_name'].split('/')[-1][:30]
        chunking = entry['config']['chunking_strategy'][:18]
        print(f"{i:<5} {model:<35} {chunking:<20} {entry['accuracy']*100:>7.2f}%  {entry['top3_accuracy']*100:>7.2f}%  {entry['macro_f1']:>7.4f}")
    
    print("="*100)


def run_all_experiments(max_experiments: Optional[int] = None):
    """Run all experiment combinations."""
    logger.info("Creating train/validation split...")
    train_split, val_split = create_train_val_split(validation_ratio=0.2)
    
    experiments = []
    
    for model_cfg in EMBEDDING_MODELS:
        for chunk_cfg in CHUNKING_STRATEGIES:
            for agg_cfg in AGGREGATION_CONFIGS:
                exp_id = f"{model_cfg['name'].split('/')[-1]}_{chunk_cfg['name']}_{agg_cfg['name']}"
                
                config = ExperimentConfig(
                    experiment_id=exp_id,
                    model_name=model_cfg['name'],
                    model_dim=model_cfg['dim'],
                    chunking_strategy=chunk_cfg['name'],
                    use_schema=chunk_cfg['use_schema'],
                    use_content=chunk_cfg['use_content'],
                    use_summary=chunk_cfg['use_summary'],
                    aggregation_name=agg_cfg['name'],
                    weight_max=agg_cfg['weight_max'],
                    weight_avg=agg_cfg['weight_avg'],
                    weight_count=agg_cfg['weight_count'],
                )
                experiments.append(config)
    
    logger.info(f"Total experiments to run: {len(experiments)}")
    
    if max_experiments:
        experiments = experiments[:max_experiments]
        logger.info(f"Limited to {max_experiments} experiments")
    
    for i, config in enumerate(experiments, 1):
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"EXPERIMENT {i}/{len(experiments)}")
        logger.info(f"{'#'*60}")
        
        try:
            result = run_experiment(config, train_split, val_split)
            update_leaderboard(result)
            print_leaderboard()
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-experiments', type=int, default=None)
    parser.add_argument('--show-leaderboard', action='store_true')
    parser.add_argument('--single-model', type=str, default=None, help='Run only this model')
    args = parser.parse_args()
    
    if args.show_leaderboard:
        print_leaderboard()
        return
    
    if args.single_model:
        # Filter to single model
        global EMBEDDING_MODELS
        EMBEDDING_MODELS = [m for m in EMBEDDING_MODELS if args.single_model in m['name']]
        if not EMBEDDING_MODELS:
            logger.error(f"Model not found: {args.single_model}")
            return
    
    run_all_experiments(max_experiments=args.max_experiments)


if __name__ == "__main__":
    main()
