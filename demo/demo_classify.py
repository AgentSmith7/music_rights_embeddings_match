#!/usr/bin/env python3
"""
Quick Classification Demo

Classify a single file or directory of files using the trained model.

Usage:
    python demo_classify.py --file /path/to/document.csv
    python demo_classify.py --directory /path/to/documents/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install sentence-transformers qdrant-client")
    sys.exit(1)


class DocumentClassifier:
    """Simple document classifier using embedding similarity."""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf', '.txt', '.tsv', '.tab'}
    
    def __init__(self, qdrant_path: str, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "auto"):
        self.qdrant_path = qdrant_path
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if self._has_cuda() else "cpu")
        self.model = None
        self.client = None
        self.collection_name = "music_rights"
        
    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_model(self):
        if self.model is None:
            print(f"Loading model {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded.")
    
    def _load_qdrant(self):
        if self.client is None:
            print(f"Connecting to Qdrant at {self.qdrant_path}...")
            self.client = QdrantClient(path=self.qdrant_path)
            print("Connected.")
    
    def _parse_file(self, file_path: str) -> str:
        """Parse file into text representation."""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.csv':
            return self._parse_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return self._parse_excel(file_path)
        elif ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.txt', '.tsv', '.tab']:
            return self._parse_text(file_path)
        else:
            return ""
    
    def _parse_csv(self, path: str) -> str:
        import pandas as pd
        try:
            df = pd.read_csv(path, nrows=50, on_bad_lines='skip')
            cols = list(df.columns[:30])
            sample = df.head(10).to_string(index=False, max_colwidth=50)
            return f"CSV Schema: {', '.join(cols)}\nSample Data:\n{sample}"
        except Exception as e:
            return f"CSV file (parse error: {e})"
    
    def _parse_excel(self, path: str) -> str:
        import pandas as pd
        try:
            df = pd.read_excel(path, nrows=50)
            cols = list(df.columns[:30])
            sample = df.head(10).to_string(index=False, max_colwidth=50)
            return f"Excel Schema: {', '.join(str(c) for c in cols)}\nSample Data:\n{sample}"
        except Exception as e:
            return f"Excel file (parse error: {e})"
    
    def _parse_pdf(self, path: str) -> str:
        try:
            import fitz
            doc = fitz.open(path)
            text_parts = []
            for i in range(min(3, len(doc))):
                text_parts.append(doc[i].get_text()[:2000])
            doc.close()
            return "\n".join(text_parts)[:4096]
        except Exception as e:
            return f"PDF file (parse error: {e})"
    
    def _parse_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()[:4096]
        except:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    return f.read()[:4096]
            except Exception as e:
                return f"Text file (parse error: {e})"
    
    def classify(self, file_path: str, top_k: int = 10) -> Dict[str, Any]:
        """Classify a single file."""
        self._load_model()
        self._load_qdrant()
        
        # Parse file
        text = self._parse_file(file_path)
        if not text:
            return {
                "file": file_path,
                "error": "Could not parse file",
                "predicted_class": None,
                "confidence": 0.0
            }
        
        # Embed
        embedding = self.model.encode(text, normalize_embeddings=True)
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=top_k
        )
        
        if not results.points:
            return {
                "file": file_path,
                "error": "No matches found",
                "predicted_class": None,
                "confidence": 0.0
            }
        
        # Aggregate scores (weighted method)
        class_results = defaultdict(list)
        for r in results.points:
            class_label = r.payload["class_label"]
            class_results[class_label].append(r.score)
        
        class_scores = {}
        for cls, scores in class_results.items():
            max_score = max(scores)
            avg_score = np.mean(scores)
            count = len(scores) / top_k
            class_scores[cls] = 0.5 * max_score + 0.3 * avg_score + 0.2 * count
        
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_score = sorted_classes[0]
        second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = top_score - second_score
        
        return {
            "file": file_path,
            "predicted_class": top_class,
            "confidence": round(top_score, 4),
            "margin": round(margin, 4),
            "is_uncertain": margin < 0.10 or top_score < 0.70,
            "top_classes": [{"class": c, "score": round(s, 4)} for c, s in sorted_classes[:5]]
        }
    
    def classify_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Classify all supported files in a directory."""
        results = []
        files = []
        
        for root, _, filenames in os.walk(dir_path):
            for f in filenames:
                if Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, f))
        
        print(f"Found {len(files)} files to classify")
        
        for i, file_path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Classifying {Path(file_path).name}...")
            result = self.classify(file_path)
            results.append(result)
            
            if result.get("predicted_class"):
                status = "UNCERTAIN" if result.get("is_uncertain") else "OK"
                print(f"  -> {result['predicted_class']} (conf: {result['confidence']:.2f}) [{status}]")
            else:
                print(f"  -> ERROR: {result.get('error')}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Classify music rights documents")
    parser.add_argument("--file", help="Path to a single file to classify")
    parser.add_argument("--directory", help="Path to directory of files to classify")
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Path to Qdrant data")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Embedding model")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=10, help="Number of neighbors to retrieve")
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.error("Either --file or --directory must be specified")
    
    classifier = DocumentClassifier(args.qdrant_path, args.model, args.device)
    
    if args.file:
        result = classifier.classify(args.file, args.top_k)
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"File: {result['file']}")
        if result.get("predicted_class"):
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Margin: {result['margin']:.4f}")
            print(f"Uncertain: {result['is_uncertain']}")
            print("\nTop 5 Classes:")
            for item in result.get("top_classes", []):
                print(f"  {item['class']}: {item['score']:.4f}")
        else:
            print(f"Error: {result.get('error')}")
        print("=" * 60)
    
    elif args.directory:
        results = classifier.classify_directory(args.directory)
        
        # Summary
        total = len(results)
        successful = sum(1 for r in results if r.get("predicted_class"))
        uncertain = sum(1 for r in results if r.get("is_uncertain"))
        
        print("\n" + "=" * 60)
        print("BATCH CLASSIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total files: {total}")
        print(f"Successfully classified: {successful}")
        print(f"Uncertain predictions: {uncertain}")
        print(f"Errors: {total - successful}")
        
        # Class distribution
        class_counts = defaultdict(int)
        for r in results:
            if r.get("predicted_class"):
                class_counts[r["predicted_class"]] += 1
        
        print("\nClass Distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
