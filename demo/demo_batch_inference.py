#!/usr/bin/env python3
"""
Batch Inference Demo

Run inference on a ZIP archive or large directory with progress tracking.
Supports both CSV and JSON output formats.

Usage:
    python demo_batch_inference.py --input /path/to/archive.zip --output results.json
    python demo_batch_inference.py --input /path/to/docs/ --output-csv results.csv
"""

import os
import sys
import json
import csv
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
import io

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install sentence-transformers qdrant-client tqdm")
    sys.exit(1)


@dataclass
class PredictionResult:
    file_path: str
    file_name: str
    file_type: str
    predicted_class: str
    confidence: float
    margin: float
    is_uncertain: bool
    strategy_used: str
    top_classes: List[Dict[str, float]]


class BatchInference:
    """Batch inference engine with progress tracking."""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab'}
    
    def __init__(self, qdrant_path: str, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.qdrant_path = qdrant_path
        self.model_name = model_name
        self.model = None
        self.client = None
        self.collection_name = "music_rights"
        self.results: List[PredictionResult] = []
        
        # Inference config
        self.top_k = 10
        self.min_margin = 0.10
        self.uncertainty_threshold = 0.70
    
    def _load_resources(self):
        if self.model is None:
            print("Loading embedding model...")
            self.model = SentenceTransformer(self.model_name, device="cuda")
            print("Model loaded on GPU")
        
        if self.client is None:
            print(f"Connecting to Qdrant...")
            self.client = QdrantClient(path=self.qdrant_path)
            print("Connected")
    
    def _parse_content(self, file_path: str, content: bytes) -> str:
        """Parse file content into text."""
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
            pass
        return ""
    
    def _parse_csv(self, content: bytes) -> str:
        import pandas as pd
        try:
            df = pd.read_csv(io.BytesIO(content), nrows=50, on_bad_lines='skip')
        except:
            df = pd.read_csv(io.BytesIO(content), nrows=50, encoding='latin-1', on_bad_lines='skip')
        cols = list(df.columns[:30])
        sample = df.head(10).to_string(index=False, max_colwidth=50)
        return f"CSV Schema: {', '.join(cols)}\nSample Data:\n{sample}"
    
    def _parse_excel(self, content: bytes, ext: str) -> str:
        import pandas as pd
        engine = 'xlrd' if ext == '.xls' else 'openpyxl'
        df = pd.read_excel(io.BytesIO(content), nrows=50, engine=engine)
        cols = list(df.columns[:30])
        sample = df.head(10).to_string(index=False, max_colwidth=50)
        return f"Excel Schema: {', '.join(str(c) for c in cols)}\nSample Data:\n{sample}"
    
    def _parse_pdf(self, content: bytes) -> str:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []
        for i in range(min(3, len(doc))):
            text_parts.append(doc[i].get_text()[:2000])
        doc.close()
        return "\n".join(text_parts)[:4096]
    
    def _parse_text(self, content: bytes) -> str:
        try:
            return content.decode('utf-8')[:4096]
        except:
            return content.decode('latin-1')[:4096]
    
    def _predict(self, file_path: str, text: str) -> PredictionResult:
        """Make prediction for a single file."""
        ext = Path(file_path).suffix.lower()
        
        # Embed
        embedding = self.model.encode(text, normalize_embeddings=True)
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=self.top_k
        )
        
        if not results.points:
            return PredictionResult(
                file_path=file_path,
                file_name=Path(file_path).name,
                file_type=ext,
                predicted_class="UNKNOWN",
                confidence=0.0,
                margin=0.0,
                is_uncertain=True,
                strategy_used="none",
                top_classes=[]
            )
        
        # Aggregate (mean for primary)
        class_results = defaultdict(list)
        for r in results.points:
            class_results[r.payload["class_label"]].append(r.score)
        
        class_scores = {cls: np.mean(scores) for cls, scores in class_results.items()}
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        # Primary strategy: margin threshold
        if margin >= self.min_margin:
            return PredictionResult(
                file_path=file_path,
                file_name=Path(file_path).name,
                file_type=ext,
                predicted_class=top_class,
                confidence=top_score,
                margin=margin,
                is_uncertain=top_score < self.uncertainty_threshold,
                strategy_used="primary",
                top_classes=[{"class": c, "score": s} for c, s in sorted_classes[:5]]
            )
        
        # Fallback: weighted aggregation
        class_scores = {}
        for cls, scores in class_results.items():
            max_s = max(scores)
            avg_s = np.mean(scores)
            cnt = len(scores) / self.top_k
            class_scores[cls] = 0.5 * max_s + 0.3 * avg_s + 0.2 * cnt
        
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        return PredictionResult(
            file_path=file_path,
            file_name=Path(file_path).name,
            file_type=ext,
            predicted_class=top_class,
            confidence=top_score,
            margin=margin,
            is_uncertain=top_score < self.uncertainty_threshold or margin < 0.05,
            strategy_used="fallback",
            top_classes=[{"class": c, "score": s} for c, s in sorted_classes[:5]]
        )
    
    def process_zip(self, zip_path: str) -> List[PredictionResult]:
        """Process all files in a ZIP archive."""
        self._load_resources()
        
        print(f"Opening ZIP: {zip_path}")
        results = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = [f for f in zf.namelist() 
                    if not f.endswith('/') and Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS]
            
            print(f"Found {len(files)} files to process")
            
            for file_path in tqdm(files, desc="Processing"):
                try:
                    content = zf.read(file_path)
                    text = self._parse_content(file_path, content)
                    
                    if text:
                        result = self._predict(file_path, text)
                    else:
                        result = PredictionResult(
                            file_path=file_path,
                            file_name=Path(file_path).name,
                            file_type=Path(file_path).suffix.lower(),
                            predicted_class="PARSE_ERROR",
                            confidence=0.0,
                            margin=0.0,
                            is_uncertain=True,
                            strategy_used="none",
                            top_classes=[]
                        )
                    results.append(result)
                except Exception as e:
                    results.append(PredictionResult(
                        file_path=file_path,
                        file_name=Path(file_path).name,
                        file_type=Path(file_path).suffix.lower(),
                        predicted_class="ERROR",
                        confidence=0.0,
                        margin=0.0,
                        is_uncertain=True,
                        strategy_used="none",
                        top_classes=[]
                    ))
        
        self.results = results
        return results
    
    def process_directory(self, dir_path: str) -> List[PredictionResult]:
        """Process all files in a directory."""
        self._load_resources()
        
        print(f"Scanning directory: {dir_path}")
        results = []
        
        files = []
        for root, _, filenames in os.walk(dir_path):
            for f in filenames:
                if Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, f))
        
        print(f"Found {len(files)} files to process")
        
        for file_path in tqdm(files, desc="Processing"):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                text = self._parse_content(file_path, content)
                
                if text:
                    result = self._predict(file_path, text)
                else:
                    result = PredictionResult(
                        file_path=file_path,
                        file_name=Path(file_path).name,
                        file_type=Path(file_path).suffix.lower(),
                        predicted_class="PARSE_ERROR",
                        confidence=0.0,
                        margin=0.0,
                        is_uncertain=True,
                        strategy_used="none",
                        top_classes=[]
                    )
                results.append(result)
            except Exception as e:
                pass
        
        self.results = results
        return results
    
    def save_csv(self, output_path: str):
        """Save results to CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'file_name', 'predicted_class', 'confidence', 'margin', 'is_uncertain', 'strategy'])
            for r in self.results:
                writer.writerow([r.file_path, r.file_name, r.predicted_class, 
                               f"{r.confidence:.4f}", f"{r.margin:.4f}", r.is_uncertain, r.strategy_used])
        print(f"CSV saved to {output_path}")
    
    def save_json(self, output_path: str):
        """Save detailed results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(self.results),
            "summary": {
                "primary_strategy": sum(1 for r in self.results if r.strategy_used == "primary"),
                "fallback_strategy": sum(1 for r in self.results if r.strategy_used == "fallback"),
                "uncertain": sum(1 for r in self.results if r.is_uncertain),
                "errors": sum(1 for r in self.results if r.predicted_class in ["ERROR", "PARSE_ERROR", "UNKNOWN"])
            },
            "predictions": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"JSON saved to {output_path}")
    
    def print_summary(self):
        """Print summary statistics."""
        total = len(self.results)
        primary = sum(1 for r in self.results if r.strategy_used == "primary")
        fallback = sum(1 for r in self.results if r.strategy_used == "fallback")
        uncertain = sum(1 for r in self.results if r.is_uncertain)
        errors = sum(1 for r in self.results if r.predicted_class in ["ERROR", "PARSE_ERROR", "UNKNOWN"])
        
        class_counts = defaultdict(int)
        for r in self.results:
            class_counts[r.predicted_class] += 1
        
        print("\n" + "=" * 60)
        print("BATCH INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total files: {total}")
        print(f"High confidence (primary): {primary} ({100*primary/total:.1f}%)")
        print(f"Lower confidence (fallback): {fallback} ({100*fallback/total:.1f}%)")
        print(f"Uncertain predictions: {uncertain} ({100*uncertain/total:.1f}%)")
        print(f"Errors: {errors} ({100*errors/total:.1f}%)")
        print()
        print("Top 10 predicted classes:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {count} ({100*count/total:.1f}%)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch inference on documents")
    parser.add_argument("--input", required=True, help="Path to ZIP file or directory")
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Path to Qdrant data")
    parser.add_argument("--output-json", help="Output JSON file path")
    parser.add_argument("--output-csv", help="Output CSV file path")
    args = parser.parse_args()
    
    if not args.output_json and not args.output_csv:
        args.output_json = "predictions.json"
    
    inference = BatchInference(args.qdrant_path)
    
    if args.input.endswith('.zip'):
        inference.process_zip(args.input)
    else:
        inference.process_directory(args.input)
    
    if args.output_csv:
        inference.save_csv(args.output_csv)
    if args.output_json:
        inference.save_json(args.output_json)
    
    inference.print_summary()


if __name__ == "__main__":
    main()
