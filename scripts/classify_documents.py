#!/usr/bin/env python3
"""
Classify documents using the indexed Qdrant database.

This script:
1. Loads a ZIP file or directory of documents
2. Parses each document
3. Queries Qdrant for similar documents
4. Aggregates results to predict class
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Constants
QDRANT_PATH = "/workspace/qdrant_data"
RESULTS_PATH = "/workspace/results"
COLLECTION_NAME = "music_rights_documents"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Classification config
TOP_K = 20
MIN_SIMILARITY = 0.5
WEIGHT_MAX_SIM = 0.4
WEIGHT_AVG_SIM = 0.3
WEIGHT_COUNT = 0.3


@dataclass
class ClassificationResult:
    """Result of classifying a single document."""
    file_name: str
    predicted_class: str
    confidence: float
    class_scores: Dict[str, float]
    top_matches: List[Dict]
    abstained: bool = False


def parse_csv_bytes(content: bytes, filename: str) -> List[Tuple[str, str]]:
    """Parse CSV from bytes."""
    representations = []
    
    try:
        df = pd.read_csv(BytesIO(content), nrows=100, encoding='utf-8', on_bad_lines='skip')
        
        if df.empty:
            return representations
        
        # Schema
        schema_parts = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            sample = sample[:50]
            schema_parts.append(f"{col} ({dtype}): {sample}")
        
        schema_text = "CSV Schema:\n" + "\n".join(schema_parts[:30])
        representations.append(("schema", schema_text))
        
        # Content
        content_parts = []
        for idx, row in df.head(5).iterrows():
            row_text = ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            content_parts.append(row_text[:500])
        
        content_text = "Sample data:\n" + "\n".join(content_parts)
        representations.append(("content", content_text))
        
    except Exception as e:
        logger.warning(f"Error parsing CSV {filename}: {e}")
    
    return representations


def parse_document_bytes(content: bytes, filename: str) -> List[Tuple[str, str]]:
    """Parse document from bytes based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.csv']:
        return parse_csv_bytes(content, filename)
    elif ext in ['.txt', '.text']:
        try:
            text = content.decode('utf-8', errors='ignore')[:2000]
            return [("content", text)]
        except:
            return []
    else:
        # Try as text
        try:
            text = content.decode('utf-8', errors='ignore')[:2000]
            if text.strip():
                return [("content", text)]
        except:
            pass
        return []


def aggregate_class_scores(matches: List[Dict]) -> Dict[str, float]:
    """
    Aggregate search results into class scores.
    
    Uses weighted combination of:
    - Max similarity per class
    - Average similarity per class
    - Count of matches per class
    """
    class_sims = defaultdict(list)
    
    for match in matches:
        class_label = match['payload']['class_label']
        class_sims[class_label].append(match['score'])
    
    if not class_sims:
        return {}
    
    # Normalize counts
    max_count = max(len(sims) for sims in class_sims.values())
    
    class_scores = {}
    for class_label, sims in class_sims.items():
        max_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
        count_norm = len(sims) / max_count
        
        score = (
            WEIGHT_MAX_SIM * max_sim +
            WEIGHT_AVG_SIM * avg_sim +
            WEIGHT_COUNT * count_norm
        )
        class_scores[class_label] = score
    
    return class_scores


def classify_document(
    client: QdrantClient,
    model: SentenceTransformer,
    content: bytes,
    filename: str
) -> ClassificationResult:
    """Classify a single document."""
    
    # Parse document
    representations = parse_document_bytes(content, filename)
    
    if not representations:
        return ClassificationResult(
            file_name=filename,
            predicted_class="UNKNOWN",
            confidence=0.0,
            class_scores={},
            top_matches=[],
            abstained=True
        )
    
    # Get embeddings for all representations
    texts = [text for _, text in representations]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Query Qdrant for each representation and combine results
    all_matches = []
    
    for embedding in embeddings:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=TOP_K
        )
        
        for result in results:
            all_matches.append({
                'score': result.score,
                'payload': result.payload
            })
    
    if not all_matches:
        return ClassificationResult(
            file_name=filename,
            predicted_class="UNKNOWN",
            confidence=0.0,
            class_scores={},
            top_matches=[],
            abstained=True
        )
    
    # Aggregate scores
    class_scores = aggregate_class_scores(all_matches)
    
    # Sort and get prediction
    sorted_classes = sorted(class_scores.items(), key=lambda x: -x[1])
    
    predicted_class = sorted_classes[0][0]
    confidence = sorted_classes[0][1]
    
    # Check if should abstain
    abstained = False
    if confidence < MIN_SIMILARITY:
        abstained = True
    elif len(sorted_classes) > 1:
        margin = sorted_classes[0][1] - sorted_classes[1][1]
        if margin < 0.05:  # Too close
            abstained = True
    
    # Get top matches for evidence
    top_matches = sorted(all_matches, key=lambda x: -x['score'])[:5]
    
    return ClassificationResult(
        file_name=filename,
        predicted_class=predicted_class if not abstained else "NEEDS_REVIEW",
        confidence=confidence,
        class_scores=dict(sorted_classes[:10]),
        top_matches=top_matches,
        abstained=abstained
    )


def classify_zip(
    client: QdrantClient,
    model: SentenceTransformer,
    zip_path: str
) -> List[ClassificationResult]:
    """Classify all documents in a ZIP file."""
    
    results = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = [f for f in zf.namelist() if not f.endswith('/')]
        logger.info(f"Found {len(file_list)} files in ZIP")
        
        for filename in tqdm(file_list, desc="Classifying"):
            try:
                content = zf.read(filename)
                result = classify_document(client, model, content, filename)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
                results.append(ClassificationResult(
                    file_name=filename,
                    predicted_class="ERROR",
                    confidence=0.0,
                    class_scores={},
                    top_matches=[],
                    abstained=True
                ))
    
    return results


def classify_directory(
    client: QdrantClient,
    model: SentenceTransformer,
    dir_path: str
) -> List[ClassificationResult]:
    """Classify all documents in a directory."""
    
    results = []
    
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append((filename, filepath))
    
    logger.info(f"Found {len(files)} files in directory")
    
    for filename, filepath in tqdm(files, desc="Classifying"):
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            result = classify_document(client, model, content, filename)
            results.append(result)
        except Exception as e:
            logger.warning(f"Error processing {filename}: {e}")
    
    return results


def evaluate_results(results: List[ClassificationResult], ground_truth: Optional[Dict[str, str]] = None):
    """Evaluate classification results."""
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*60)
    
    total = len(results)
    abstained = sum(1 for r in results if r.abstained)
    classified = total - abstained
    
    print(f"\nTotal documents: {total}")
    print(f"Classified: {classified} ({classified/total*100:.1f}%)")
    print(f"Abstained/Needs Review: {abstained} ({abstained/total*100:.1f}%)")
    
    # Class distribution
    class_counts = defaultdict(int)
    for r in results:
        if not r.abstained:
            class_counts[r.predicted_class] += 1
    
    print(f"\nPredicted class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cls}: {count} ({count/classified*100:.1f}%)")
    
    # Confidence distribution
    confidences = [r.confidence for r in results if not r.abstained]
    if confidences:
        print(f"\nConfidence statistics:")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
        print(f"  Mean: {sum(confidences)/len(confidences):.3f}")
    
    # If ground truth provided, compute accuracy
    if ground_truth:
        correct = 0
        total_with_truth = 0
        
        for r in results:
            if r.file_name in ground_truth and not r.abstained:
                total_with_truth += 1
                if r.predicted_class == ground_truth[r.file_name]:
                    correct += 1
        
        if total_with_truth > 0:
            accuracy = correct / total_with_truth
            print(f"\nAccuracy (on {total_with_truth} files with ground truth): {accuracy*100:.1f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to ZIP file or directory')
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--ground-truth', default=None, help='Ground truth JSON for evaluation')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("DOCUMENT CLASSIFICATION PIPELINE")
    logger.info("="*60)
    
    # Setup
    logger.info("Loading Qdrant client...")
    client = QdrantClient(path=QDRANT_PATH)
    
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Collection has {info.points_count} indexed vectors")
    
    logger.info("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    
    # Classify
    if args.input.endswith('.zip'):
        results = classify_zip(client, model, args.input)
    else:
        results = classify_directory(client, model, args.input)
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)
    
    # Evaluate
    evaluate_results(results, ground_truth)
    
    # Save results
    output_path = args.output or os.path.join(RESULTS_PATH, 'classification_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
