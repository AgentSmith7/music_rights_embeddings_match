#!/usr/bin/env python3
"""
Experiment runner v2 with optimized storage:
- Float16 embeddings (50% storage reduction)
- Qdrant scalar quantization option
- Efficient batch processing
- Single experiment mode for initial testing
"""

import os
import sys
import json
import time
import uuid
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    OptimizersConfigDiff, HnswConfigDiff
)

# Constants
EXTRACT_PATH = "/workspace/training_data/extracted"
STATEMENT_PATH = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
QDRANT_BASE_PATH = "/workspace/qdrant_experiments"
RESULTS_PATH = "/workspace/results"
LEADERBOARD_PATH = os.path.join(RESULTS_PATH, "leaderboard.json")
EMBEDDINGS_CACHE_PATH = "/workspace/embeddings_cache"

# Default experiment config
DEFAULT_MODEL = {"name": "BAAI/bge-large-en-v1.5", "dim": 1024}
DEFAULT_CHUNKING = {"name": "schema_content", "use_schema": True, "use_content": True, "use_summary": False}
DEFAULT_AGGREGATION = {"name": "balanced", "weight_max": 0.4, "weight_avg": 0.3, "weight_count": 0.3}


@dataclass
class ExperimentConfig:
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
    use_quantization: bool = True
    use_float16: bool = True


@dataclass
class ExperimentResult:
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
    total_vectors: int
    storage_mb: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def create_train_val_split(validation_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Create stratified train/validation split."""
    random.seed(seed)
    
    if not os.path.exists(STATEMENT_PATH):
        raise ValueError(f"Statement path not found: {STATEMENT_PATH}")
    
    train_split = {}
    val_split = {}
    
    class_folders = [d for d in os.listdir(STATEMENT_PATH) 
                     if os.path.isdir(os.path.join(STATEMENT_PATH, d))]
    log(f"Found {len(class_folders)} classes")
    
    for class_name in class_folders:
        class_path = os.path.join(STATEMENT_PATH, class_name)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        if not files:
            continue
            
        random.shuffle(files)
        val_size = max(1, int(len(files) * validation_ratio))
        
        val_files = files[:val_size]
        train_files = files[val_size:]
        
        train_split[class_name] = [os.path.join(class_path, f) for f in train_files]
        val_split[class_name] = [os.path.join(class_path, f) for f in val_files]
    
    total_train = sum(len(v) for v in train_split.values())
    total_val = sum(len(v) for v in val_split.values())
    log(f"Split: {total_train} train, {total_val} validation across {len(train_split)} classes")
    
    return train_split, val_split


def parse_csv_file(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse CSV and return representations."""
    representations = []
    
    try:
        df = pd.read_csv(filepath, nrows=100, encoding='utf-8', on_bad_lines='skip')
        if df.empty:
            return representations
        
        if config.use_schema:
            schema_parts = []
            for col in df.columns[:30]:
                dtype = str(df[col].dtype)
                sample = str(df[col].dropna().iloc[0])[:50] if not df[col].dropna().empty else ""
                schema_parts.append(f"{col} ({dtype}): {sample}")
            schema_text = "CSV Schema:\n" + "\n".join(schema_parts)
            representations.append(("schema", schema_text))
        
        if config.use_content:
            content_parts = []
            for idx, row in df.head(5).iterrows():
                row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                content_parts.append(row_text[:500])
            content_text = "Sample data:\n" + "\n".join(content_parts)
            representations.append(("content", content_text))
        
        if config.use_summary:
            summary = f"CSV with {len(df.columns)} columns: {', '.join(df.columns[:10])}"
            representations.append(("summary", summary))
            
    except Exception:
        pass
    
    return representations


def parse_excel_file(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse Excel file."""
    representations = []
    
    try:
        df = pd.read_excel(filepath, nrows=100)
        if df.empty:
            return representations
        
        if config.use_schema:
            schema_parts = []
            for col in df.columns[:30]:
                dtype = str(df[col].dtype)
                sample = str(df[col].dropna().iloc[0])[:50] if not df[col].dropna().empty else ""
                schema_parts.append(f"{col} ({dtype}): {sample}")
            schema_text = "Excel Schema:\n" + "\n".join(schema_parts)
            representations.append(("schema", schema_text))
        
        if config.use_content:
            content_parts = []
            for idx, row in df.head(5).iterrows():
                row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                content_parts.append(row_text[:500])
            content_text = "Sample data:\n" + "\n".join(content_parts)
            representations.append(("content", content_text))
            
    except Exception:
        pass
    
    return representations


def parse_text_file(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse text file."""
    representations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4000)
        
        if content.strip() and len(content.strip()) >= 20:
            if config.use_content:
                representations.append(("content", content[:2000]))
            if config.use_summary:
                representations.append(("summary", content[:500]))
    except Exception:
        pass
    
    return representations


def parse_document(filepath: str, config: ExperimentConfig) -> List[Tuple[str, str]]:
    """Parse document based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        return parse_csv_file(filepath, config)
    elif ext in ['.xlsx', '.xls']:
        return parse_excel_file(filepath, config)
    elif ext in ['.txt', '.text', '.log']:
        return parse_text_file(filepath, config)
    
    return []


def run_single_experiment(
    model_name: str = DEFAULT_MODEL["name"],
    model_dim: int = DEFAULT_MODEL["dim"],
    chunking: Dict = None,
    aggregation: Dict = None,
    use_quantization: bool = True,
    use_float16: bool = True
) -> ExperimentResult:
    """Run a single experiment with specified configuration."""
    
    chunking = chunking or DEFAULT_CHUNKING
    aggregation = aggregation or DEFAULT_AGGREGATION
    
    exp_id = f"{model_name.split('/')[-1]}_{chunking['name']}_{aggregation['name']}"
    
    config = ExperimentConfig(
        experiment_id=exp_id,
        model_name=model_name,
        model_dim=model_dim,
        chunking_strategy=chunking['name'],
        use_schema=chunking['use_schema'],
        use_content=chunking['use_content'],
        use_summary=chunking['use_summary'],
        aggregation_name=aggregation['name'],
        weight_max=aggregation['weight_max'],
        weight_avg=aggregation['weight_avg'],
        weight_count=aggregation['weight_count'],
        use_quantization=use_quantization,
        use_float16=use_float16
    )
    
    log(f"\n{'='*70}")
    log(f"EXPERIMENT: {config.experiment_id}")
    log(f"Model: {config.model_name}")
    log(f"Chunking: {config.chunking_strategy}")
    log(f"Aggregation: {config.aggregation_name}")
    log(f"Quantization: {config.use_quantization}, Float16: {config.use_float16}")
    log(f"{'='*70}")
    
    # Create splits
    log("Creating train/validation split...")
    train_split, val_split = create_train_val_split(validation_ratio=config.validation_split)
    
    # Setup Qdrant
    qdrant_path = os.path.join(QDRANT_BASE_PATH, config.experiment_id)
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)
    os.makedirs(qdrant_path, exist_ok=True)
    
    client = QdrantClient(path=qdrant_path)
    collection_name = "documents"
    
    # Create collection with optional quantization
    vector_config = VectorParams(size=config.model_dim, distance=Distance.COSINE)
    
    if config.use_quantization:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config
        )
    
    # Load model
    log(f"Loading model: {config.model_name}")
    model = SentenceTransformer(config.model_name, device='cuda')
    
    # Index training documents
    log("Indexing training documents...")
    start_index = time.time()
    
    batch_texts = []
    batch_payloads = []
    total_indexed = 0
    batch_size = 128
    
    all_train_files = [(cls, fp) for cls, files in train_split.items() for fp in files]
    
    for class_name, filepath in tqdm(all_train_files, desc="Indexing"):
        representations = parse_document(filepath, config)
        
        for rep_type, text in representations:
            if text and len(text.strip()) >= 10:
                batch_texts.append(text)
                batch_payloads.append({
                    "class_label": class_name,
                    "file_path": filepath,
                    "rep_type": rep_type
                })
                
                if len(batch_texts) >= batch_size:
                    embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
                    
                    if config.use_float16:
                        embeddings = embeddings.astype(np.float16)
                    
                    points = [
                        PointStruct(
                            id=str(uuid.uuid4()), 
                            vector=emb.astype(np.float32).tolist(),
                            payload=pay
                        )
                        for emb, pay in zip(embeddings, batch_payloads)
                    ]
                    client.upsert(collection_name=collection_name, points=points)
                    total_indexed += len(batch_texts)
                    batch_texts = []
                    batch_payloads = []
    
    # Final batch
    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        if config.use_float16:
            embeddings = embeddings.astype(np.float16)
        points = [
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=emb.astype(np.float32).tolist(),
                payload=pay
            )
            for emb, pay in zip(embeddings, batch_payloads)
        ]
        client.upsert(collection_name=collection_name, points=points)
        total_indexed += len(batch_texts)
    
    indexing_time = time.time() - start_index
    log(f"Indexed {total_indexed} vectors in {indexing_time:.1f}s")
    
    # Get storage size
    storage_mb = sum(
        os.path.getsize(os.path.join(dp, f)) 
        for dp, dn, filenames in os.walk(qdrant_path) 
        for f in filenames
    ) / (1024 * 1024)
    log(f"Qdrant storage: {storage_mb:.1f} MB")
    
    # Evaluate on validation set
    log("Evaluating on validation set...")
    start_eval = time.time()
    
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    confidences = []
    abstentions = 0
    class_predictions = defaultdict(list)
    
    all_val_files = [(cls, fp) for cls, files in val_split.items() for fp in files]
    
    for class_name, filepath in tqdm(all_val_files, desc="Evaluating"):
        representations = parse_document(filepath, config)
        
        if not representations:
            abstentions += 1
            total += 1
            continue
        
        texts = [text for _, text in representations]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
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
        class_sims = defaultdict(list)
        for m in all_matches:
            class_sims[m["class"]].append(m["score"])
        
        max_count = max(len(v) for v in class_sims.values())
        
        class_scores = {}
        for cls, sims in class_sims.items():
            max_sim = max(sims)
            avg_sim = np.mean(sims)
            count_norm = len(sims) / max_count
            class_scores[cls] = (
                config.weight_max * max_sim +
                config.weight_avg * avg_sim +
                config.weight_count * count_norm
            )
        
        sorted_classes = sorted(class_scores.items(), key=lambda x: -x[1])
        predicted = sorted_classes[0][0]
        confidence = sorted_classes[0][1]
        confidences.append(confidence)
        
        total += 1
        if predicted == class_name:
            correct += 1
        
        if class_name in [c for c, _ in sorted_classes[:3]]:
            correct_top3 += 1
        
        if class_name in [c for c, _ in sorted_classes[:5]]:
            correct_top5 += 1
        
        class_predictions[class_name].append(predicted)
    
    inference_time = time.time() - start_eval
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    top3_accuracy = correct_top3 / total if total > 0 else 0
    top5_accuracy = correct_top5 / total if total > 0 else 0
    avg_confidence = float(np.mean(confidences)) if confidences else 0
    abstention_rate = abstentions / total if total > 0 else 0
    
    # Calculate macro F1
    f1_scores = []
    for cls in val_split.keys():
        if cls not in class_predictions:
            continue
        preds = class_predictions[cls]
        tp = sum(1 for p in preds if p == cls)
        fp = sum(1 for other_cls, other_preds in class_predictions.items() 
                 if other_cls != cls for p in other_preds if p == cls)
        fn = sum(1 for p in preds if p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0
    
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
        total_val_docs=total,
        total_vectors=total_indexed,
        storage_mb=storage_mb
    )
    
    log(f"\n{'='*50}")
    log(f"RESULTS")
    log(f"{'='*50}")
    log(f"Accuracy:        {accuracy*100:.2f}%")
    log(f"Top-3 Accuracy:  {top3_accuracy*100:.2f}%")
    log(f"Top-5 Accuracy:  {top5_accuracy*100:.2f}%")
    log(f"Macro F1:        {macro_f1:.4f}")
    log(f"Avg Confidence:  {avg_confidence:.4f}")
    log(f"Abstention Rate: {abstention_rate*100:.2f}%")
    log(f"Indexing Time:   {indexing_time:.1f}s")
    log(f"Inference Time:  {inference_time:.1f}s")
    log(f"Total Vectors:   {total_indexed}")
    log(f"Storage:         {storage_mb:.1f} MB")
    log(f"{'='*50}")
    
    # Save result
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []
    
    leaderboard.append(asdict(result))
    leaderboard.sort(key=lambda x: -x['accuracy'])
    
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    log(f"Results saved to {LEADERBOARD_PATH}")
    
    # Cleanup
    del model
    import gc
    gc.collect()
    
    if hasattr(sys.modules.get('torch', None), 'cuda'):
        import torch
        torch.cuda.empty_cache()
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding experiment")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL["name"],
                        help='Model name (default: BAAI/bge-large-en-v1.5)')
    parser.add_argument('--no-quantization', action='store_true',
                        help='Disable Qdrant quantization')
    parser.add_argument('--no-float16', action='store_true',
                        help='Disable float16 embeddings')
    args = parser.parse_args()
    
    # Determine model dimension
    model_dims = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
    }
    
    model_dim = model_dims.get(args.model, 1024)
    
    result = run_single_experiment(
        model_name=args.model,
        model_dim=model_dim,
        use_quantization=not args.no_quantization,
        use_float16=not args.no_float16
    )
    
    return result


if __name__ == "__main__":
    main()
