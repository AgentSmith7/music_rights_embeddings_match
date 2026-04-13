#!/usr/bin/env python3
"""
PubStrengholtSociety Inference Script

Runs inference on PubStrengholtSociety holdout dataset using the best configuration.
Handles tar.gz archives with nested ZIP files.
"""

import os
import sys
import json
import argparse
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import io

import numpy as np
from tqdm import tqdm
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


@dataclass
class PredictionResult:
    file_path: str
    file_name: str
    file_type: str
    predicted_class: str
    confidence: float
    margin: float
    is_uncertain: bool
    top_classes: List[Dict[str, float]]
    strategy_used: str


@dataclass
class InferenceConfig:
    primary_aggregation: str = "mean"
    primary_top_k: int = 10
    primary_min_margin: float = 0.10
    fallback_aggregation: str = "weighted"
    fallback_top_k: int = 10
    fallback_weight_max: float = 0.5
    fallback_weight_avg: float = 0.3
    fallback_weight_count: float = 0.2
    uncertainty_threshold: float = 0.70


class PubStrengholtInference:
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab'}
    
    def __init__(self, qdrant_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.qdrant_path = qdrant_path
        self.embedding_model = embedding_model
        self.embedder = None
        self.qdrant_client = None
        self.collection_name = "music_rights"
        self.config = InferenceConfig()
        self.results: List[PredictionResult] = []
    
    def _load_embedder(self):
        if self.embedder is not None:
            return
        logger.info(f"Loading embedding model: {self.embedding_model}")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(self.embedding_model, device="cuda")
        logger.info("Model loaded on GPU")
    
    def _load_qdrant(self):
        if self.qdrant_client is not None:
            return
        logger.info(f"Connecting to Qdrant at {self.qdrant_path}")
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        logger.info("Qdrant connected")
    
    def _embed(self, text: str) -> np.ndarray:
        self._load_embedder()
        embedding = self.embedder.encode(
            text, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.astype(np.float16)
    
    def _search(self, vector: np.ndarray, top_k: int) -> List[dict]:
        self._load_qdrant()
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name, query=vector.tolist(), limit=top_k
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]
    
    def _parse_file_content(self, file_path: str, content: bytes) -> str:
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.csv':
                return self._parse_csv(content)
            elif ext in ['.xlsx', '.xls']:
                return self._parse_excel(content, ext)
            elif ext == '.pdf':
                return self._parse_pdf(content)
            elif ext in ['.txt', '.tsv', '.tab']:
                return self._parse_text(content)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
        return ""
    
    def _parse_csv(self, content: bytes) -> str:
        import pandas as pd
        try:
            df = pd.read_csv(io.BytesIO(content), nrows=50, encoding='utf-8', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(io.BytesIO(content), nrows=50, encoding='latin-1', on_bad_lines='skip')
            except:
                return ""
        cols = list(df.columns[:30])
        sample_rows = df.head(10).to_string(index=False, max_colwidth=50)
        return f"CSV Schema: {', '.join(str(c) for c in cols)}\nSample Data:\n{sample_rows}"
    
    def _parse_excel(self, content: bytes, ext: str) -> str:
        import pandas as pd
        try:
            engine = 'xlrd' if ext == '.xls' else 'openpyxl'
            df = pd.read_excel(io.BytesIO(content), nrows=50, engine=engine)
        except:
            return ""
        cols = list(df.columns[:30])
        sample_rows = df.head(10).to_string(index=False, max_colwidth=50)
        return f"Excel Schema: {', '.join(str(c) for c in cols)}\nSample Data:\n{sample_rows}"
    
    def _parse_pdf(self, content: bytes) -> str:
        import fitz
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text_parts = []
            for page_num in range(min(3, len(doc))):
                text_parts.append(doc[page_num].get_text()[:2000])
            doc.close()
            return "\n".join(text_parts)[:4096]
        except:
            return ""
    
    def _parse_text(self, content: bytes) -> str:
        try:
            return content.decode('utf-8')[:4096]
        except:
            try:
                return content.decode('latin-1')[:4096]
            except:
                return ""
    
    def _aggregate_scores(self, results: List[dict], method: str, top_k: int) -> Dict[str, float]:
        class_results = defaultdict(list)
        for r in results:
            class_results[r["payload"]["class_label"]].append(r["score"])
        
        class_scores = {}
        for cls, scores in class_results.items():
            max_score = max(scores)
            avg_score = np.mean(scores)
            count = len(scores)
            if method == "mean":
                class_scores[cls] = avg_score
            elif method == "max":
                class_scores[cls] = max_score
            elif method == "weighted":
                norm_count = count / top_k
                class_scores[cls] = (
                    self.config.fallback_weight_max * max_score +
                    self.config.fallback_weight_avg * avg_score +
                    self.config.fallback_weight_count * norm_count
                )
            else:
                class_scores[cls] = avg_score
        return class_scores
    
    def _predict_single(self, file_path: str, text: str) -> PredictionResult:
        ext = Path(file_path).suffix.lower()
        vector = self._embed(text)
        results = self._search(vector, self.config.primary_top_k)
        
        if not results:
            return PredictionResult(
                file_path=file_path, file_name=Path(file_path).name, file_type=ext,
                predicted_class="UNKNOWN", confidence=0.0, margin=0.0,
                is_uncertain=True, top_classes=[], strategy_used="none"
            )
        
        class_scores = self._aggregate_scores(results, self.config.primary_aggregation, self.config.primary_top_k)
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        if margin >= self.config.primary_min_margin:
            return PredictionResult(
                file_path=file_path, file_name=Path(file_path).name, file_type=ext,
                predicted_class=top_class, confidence=float(top_score), margin=float(margin),
                is_uncertain=top_score < self.config.uncertainty_threshold,
                top_classes=[{"class": c, "score": float(s)} for c, s in sorted_classes[:5]],
                strategy_used="primary"
            )
        
        class_scores = self._aggregate_scores(results, self.config.fallback_aggregation, self.config.fallback_top_k)
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        return PredictionResult(
            file_path=file_path, file_name=Path(file_path).name, file_type=ext,
            predicted_class=top_class, confidence=float(top_score), margin=float(margin),
            is_uncertain=top_score < self.config.uncertainty_threshold or margin < 0.05,
            top_classes=[{"class": c, "score": float(s)} for c, s in sorted_classes[:5]],
            strategy_used="fallback"
        )
    
    def _extract_files_from_zip(self, zip_content: bytes, parent_path: str) -> List[Tuple[str, bytes]]:
        """Extract files from a ZIP, including nested ZIPs."""
        files = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_content))
            for name in zf.namelist():
                if name.startswith('__MACOSX') or name.startswith('._'):
                    continue
                if name.endswith('/'):
                    continue
                
                full_path = f"{parent_path}/{name}"
                ext = Path(name).suffix.lower()
                
                if ext == '.zip':
                    # Nested ZIP
                    try:
                        nested_content = zf.read(name)
                        nested_files = self._extract_files_from_zip(nested_content, full_path.rsplit('.zip', 1)[0])
                        files.extend(nested_files)
                    except Exception as e:
                        logger.warning(f"Failed to process nested ZIP {name}: {e}")
                elif ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        content = zf.read(name)
                        files.append((full_path, content))
                    except Exception as e:
                        logger.warning(f"Failed to read {name}: {e}")
            zf.close()
        except Exception as e:
            logger.warning(f"Failed to open ZIP at {parent_path}: {e}")
        return files
    
    def run_inference_on_targz(self, targz_path: str) -> List[PredictionResult]:
        logger.info(f"Opening tar.gz: {targz_path}")
        results = []
        all_files = []
        
        # First pass: extract all files including from nested ZIPs
        logger.info("Scanning for files (including nested ZIPs)...")
        with tarfile.open(targz_path, 'r:gz') as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if member.name.startswith('._') or '/__MACOSX/' in member.name:
                    continue
                
                ext = Path(member.name).suffix.lower()
                
                if ext == '.zip':
                    # Extract ZIP and process its contents
                    try:
                        f = tf.extractfile(member)
                        if f:
                            zip_content = f.read()
                            zip_files = self._extract_files_from_zip(zip_content, member.name.rsplit('.zip', 1)[0])
                            all_files.extend(zip_files)
                    except Exception as e:
                        logger.warning(f"Failed to process ZIP {member.name}: {e}")
                elif ext in self.SUPPORTED_EXTENSIONS:
                    try:
                        f = tf.extractfile(member)
                        if f:
                            content = f.read()
                            all_files.append((member.name, content))
                    except Exception as e:
                        logger.warning(f"Failed to read {member.name}: {e}")
        
        logger.info(f"Found {len(all_files)} files to process")
        
        # Process all files
        for file_path, content in tqdm(all_files, desc="Processing PubStrengholt"):
            try:
                text = self._parse_file_content(file_path, content)
                
                if not text:
                    results.append(PredictionResult(
                        file_path=file_path, file_name=Path(file_path).name,
                        file_type=Path(file_path).suffix.lower(),
                        predicted_class="PARSE_ERROR", confidence=0.0, margin=0.0,
                        is_uncertain=True, top_classes=[], strategy_used="none"
                    ))
                    continue
                
                result = self._predict_single(file_path, text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                results.append(PredictionResult(
                    file_path=file_path, file_name=Path(file_path).name,
                    file_type=Path(file_path).suffix.lower(),
                    predicted_class="ERROR", confidence=0.0, margin=0.0,
                    is_uncertain=True, top_classes=[], strategy_used="none"
                ))
        
        self.results = results
        return results
    
    def save_results_csv(self, output_path: str):
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'file_name', 'predicted_class', 'confidence', 'margin', 'is_uncertain', 'strategy'])
            for r in self.results:
                writer.writerow([
                    r.file_path, r.file_name, r.predicted_class,
                    f"{r.confidence:.4f}", f"{r.margin:.4f}",
                    str(r.is_uncertain), r.strategy_used
                ])
        logger.info(f"CSV saved to {output_path}")
    
    def save_results_json(self, output_path: str):
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "primary_aggregation": self.config.primary_aggregation,
                "primary_top_k": self.config.primary_top_k,
                "primary_min_margin": self.config.primary_min_margin,
                "fallback_aggregation": self.config.fallback_aggregation,
                "uncertainty_threshold": self.config.uncertainty_threshold,
            },
            "summary": {
                "total_files": len(self.results),
                "primary_strategy_used": sum(1 for r in self.results if r.strategy_used == "primary"),
                "fallback_strategy_used": sum(1 for r in self.results if r.strategy_used == "fallback"),
                "uncertain_predictions": sum(1 for r in self.results if r.is_uncertain),
                "parse_errors": sum(1 for r in self.results if r.predicted_class in ["PARSE_ERROR", "ERROR"]),
            },
            "predictions": [
                {
                    "file_path": r.file_path,
                    "file_name": r.file_name,
                    "file_type": r.file_type,
                    "predicted_class": r.predicted_class,
                    "confidence": r.confidence,
                    "margin": r.margin,
                    "is_uncertain": str(r.is_uncertain),
                    "strategy_used": r.strategy_used,
                    "top_classes": r.top_classes
                }
                for r in self.results
            ]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved to {output_path}")
    
    def print_summary(self):
        total = len(self.results)
        if total == 0:
            print("No results to summarize")
            return
        
        primary = sum(1 for r in self.results if r.strategy_used == "primary")
        fallback = sum(1 for r in self.results if r.strategy_used == "fallback")
        uncertain = sum(1 for r in self.results if r.is_uncertain)
        errors = sum(1 for r in self.results if r.predicted_class in ["PARSE_ERROR", "ERROR", "UNKNOWN"])
        
        class_counts = defaultdict(int)
        for r in self.results:
            class_counts[r.predicted_class] += 1
        
        print("\n" + "=" * 60)
        print("PUBSTRENGHOLT INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {total}")
        print(f"Primary strategy (high confidence): {primary} ({100*primary/total:.1f}%)")
        print(f"Fallback strategy: {fallback} ({100*fallback/total:.1f}%)")
        print(f"Uncertain predictions: {uncertain} ({100*uncertain/total:.1f}%)")
        print(f"Parse errors: {errors} ({100*errors/total:.1f}%)")
        print()
        print("Top 10 predicted classes:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {count} ({100*count/total:.1f}%)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run PubStrengholt inference")
    parser.add_argument("--input", required=True, help="Path to PubStrengholtSociety.tar.gz")
    parser.add_argument("--qdrant-path", default="/workspace/qdrant_data")
    parser.add_argument("--output-csv", default="/workspace/pubstrengholt_predictions.csv")
    parser.add_argument("--output-json", default="/workspace/pubstrengholt_predictions.json")
    args = parser.parse_args()
    
    inference = PubStrengholtInference(args.qdrant_path)
    inference.run_inference_on_targz(args.input)
    inference.save_results_csv(args.output_csv)
    inference.save_results_json(args.output_json)
    inference.print_summary()


if __name__ == "__main__":
    main()
