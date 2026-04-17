#!/usr/bin/env python3
"""
Weighted Aggregation Inference Script

Uses weighted aggregation (0.5*max + 0.3*avg + 0.2*count) without thresholds.
Provides higher coverage alternative to primary/fallback strategy.
"""

import os
import sys
import json
import argparse
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
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
    top_classes: List[Dict[str, float]]


class WeightedInference:
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab'}
    
    def __init__(self, qdrant_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.qdrant_path = qdrant_path
        self.embedding_model = embedding_model
        self.embedder = None
        self.qdrant_client = None
        self.collection_name = "music_rights"
        self.top_k = 10
        self.weight_max = 0.5
        self.weight_avg = 0.3
        self.weight_count = 0.2
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
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(io.BytesIO(content), nrows=50, encoding=encoding, 
                                    on_bad_lines='skip', sep=sep)
                    if len(df.columns) > 1:
                        cols = list(df.columns[:30])
                        sample_rows = df.head(10).to_string(index=False, max_colwidth=50)
                        return f"CSV Schema: {', '.join(str(c) for c in cols)}\nSample Data:\n{sample_rows}"
                except:
                    continue
        return ""
    
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
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(encoding)[:4096]
            except:
                continue
        return ""
    
    def _weighted_aggregation(self, results: List[dict]) -> Dict[str, float]:
        class_results = defaultdict(list)
        for r in results:
            class_results[r["payload"]["class_label"]].append(r["score"])
        
        class_scores = {}
        for cls, scores in class_results.items():
            max_score = max(scores)
            avg_score = np.mean(scores)
            norm_count = len(scores) / self.top_k
            class_scores[cls] = (
                self.weight_max * max_score +
                self.weight_avg * avg_score +
                self.weight_count * norm_count
            )
        return class_scores
    
    def _predict_single(self, file_path: str, text: str) -> PredictionResult:
        ext = Path(file_path).suffix.lower()
        vector = self._embed(text)
        results = self._search(vector, self.top_k)
        
        if not results:
            return PredictionResult(
                file_path=file_path, file_name=Path(file_path).name, file_type=ext,
                predicted_class="UNKNOWN", confidence=0.0, margin=0.0, top_classes=[]
            )
        
        class_scores = self._weighted_aggregation(results)
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        return PredictionResult(
            file_path=file_path, file_name=Path(file_path).name, file_type=ext,
            predicted_class=top_class, confidence=float(top_score), margin=float(margin),
            top_classes=[{"class": c, "score": float(s)} for c, s in sorted_classes[:5]]
        )
    
    def _extract_files_from_zip(self, zip_content: bytes, parent_path: str) -> List[Tuple[str, bytes]]:
        files = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_content))
            for name in zf.namelist():
                if name.startswith('__MACOSX') or name.startswith('._') or '/.DS_Store' in name:
                    continue
                if name.endswith('/'):
                    continue
                
                full_path = f"{parent_path}/{name}" if parent_path else name
                ext = Path(name).suffix.lower()
                
                if ext == '.zip':
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
    
    def run_inference_on_zip(self, zip_path: str) -> List[PredictionResult]:
        logger.info(f"Opening ZIP: {zip_path}")
        results = []
        
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        logger.info("Scanning for files (including nested ZIPs)...")
        all_files = self._extract_files_from_zip(zip_content, Path(zip_path).stem)
        logger.info(f"Found {len(all_files)} files to process")
        
        for file_path, content in tqdm(all_files, desc=f"Processing {Path(zip_path).stem}"):
            try:
                text = self._parse_file_content(file_path, content)
                
                if not text:
                    results.append(PredictionResult(
                        file_path=file_path, file_name=Path(file_path).name,
                        file_type=Path(file_path).suffix.lower(),
                        predicted_class="PARSE_ERROR", confidence=0.0, margin=0.0, top_classes=[]
                    ))
                    continue
                
                result = self._predict_single(file_path, text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                results.append(PredictionResult(
                    file_path=file_path, file_name=Path(file_path).name,
                    file_type=Path(file_path).suffix.lower(),
                    predicted_class="ERROR", confidence=0.0, margin=0.0, top_classes=[]
                ))
        
        self.results = results
        return results
    
    def save_results_csv(self, output_path: str):
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'file_name', 'predicted_class', 'confidence', 'margin'])
            for r in self.results:
                writer.writerow([
                    r.file_path, r.file_name, r.predicted_class,
                    f"{r.confidence:.4f}", f"{r.margin:.4f}"
                ])
        logger.info(f"CSV saved to {output_path}")
    
    def print_summary(self):
        total = len(self.results)
        if total == 0:
            print("No results to summarize")
            return
        
        errors = sum(1 for r in self.results if r.predicted_class in ["PARSE_ERROR", "ERROR", "UNKNOWN"])
        
        class_counts = defaultdict(int)
        for r in self.results:
            class_counts[r.predicted_class] += 1
        
        print("\n" + "=" * 60)
        print("WEIGHTED AGGREGATION INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total files processed: {total}")
        print(f"Parse errors: {errors} ({100*errors/total:.1f}%)")
        print()
        print("Top 10 predicted classes:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {count} ({100*count/total:.1f}%)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run weighted aggregation inference")
    parser.add_argument("--input", required=True, help="Path to ZIP file")
    parser.add_argument("--qdrant-path", default="/workspace/qdrant_data")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    
    if args.output_prefix is None:
        args.output_prefix = Path(args.input).stem
    
    output_csv = f"/workspace/{args.output_prefix}_weighted_predictions.csv"
    
    inference = WeightedInference(args.qdrant_path)
    inference.run_inference_on_zip(args.input)
    inference.save_results_csv(output_csv)
    inference.print_summary()


if __name__ == "__main__":
    main()
