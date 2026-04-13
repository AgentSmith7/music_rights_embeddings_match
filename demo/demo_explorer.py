#!/usr/bin/env python3
"""
Interactive Explorer Demo

Interactively explore the vector database and find similar documents.

Usage:
    python demo_explorer.py --qdrant-path /path/to/qdrant_data
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install sentence-transformers qdrant-client")
    sys.exit(1)


class VectorExplorer:
    """Interactive vector database explorer."""
    
    def __init__(self, qdrant_path: str, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.qdrant_path = qdrant_path
        self.model_name = model_name
        self.model = None
        self.client = None
        self.collection_name = "music_rights"
    
    def _load(self):
        if self.model is None:
            print("Loading model...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded")
        
        if self.client is None:
            print(f"Connecting to Qdrant at {self.qdrant_path}...")
            self.client = QdrantClient(path=self.qdrant_path)
            print("Connected")
    
    def get_stats(self):
        """Get collection statistics."""
        self._load()
        
        info = self.client.get_collection(self.collection_name)
        print("\n" + "=" * 60)
        print("COLLECTION STATISTICS")
        print("=" * 60)
        print(f"Collection: {self.collection_name}")
        print(f"Total vectors: {info.points_count:,}")
        print(f"Vector dimension: {info.config.params.vectors.size}")
        print("=" * 60)
        
        return info
    
    def search_by_text(self, query: str, top_k: int = 10):
        """Search for similar documents by text query."""
        self._load()
        
        print(f"\nSearching for: '{query}'")
        
        embedding = self.model.encode(query, normalize_embeddings=True)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=top_k
        )
        
        print(f"\nFound {len(results.points)} results:")
        print("-" * 60)
        
        for i, r in enumerate(results.points, 1):
            print(f"\n{i}. Score: {r.score:.4f}")
            print(f"   Class: {r.payload.get('class_label', 'N/A')}")
            print(f"   File: {r.payload.get('file_name', 'N/A')}")
            print(f"   Type: {r.payload.get('file_type', 'N/A')}")
            if 'source_path' in r.payload:
                print(f"   Path: {r.payload['source_path'][:80]}...")
        
        return results
    
    def get_class_distribution(self, sample_size: int = 1000):
        """Get class distribution from the collection."""
        self._load()
        
        # Scroll through points to get distribution
        print("\nAnalyzing class distribution...")
        
        class_counts = defaultdict(int)
        offset = None
        total = 0
        
        while total < sample_size:
            batch_size = min(100, sample_size - total)
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
            
            for r in results:
                class_counts[r.payload.get("class_label", "UNKNOWN")] += 1
                total += 1
        
        print(f"\nClass distribution (sampled {total} documents):")
        print("-" * 60)
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total
            print(f"  {cls}: {count} ({pct:.1f}%)")
        
        return class_counts
    
    def find_similar_to_class(self, class_name: str, top_k: int = 5):
        """Find documents most similar to a given class centroid."""
        self._load()
        
        # Get a sample from the class
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter={
                "must": [{"key": "class_label", "match": {"value": class_name}}]
            },
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        if not results:
            print(f"No documents found for class: {class_name}")
            return
        
        print(f"\nSample documents from class '{class_name}':")
        print("-" * 60)
        for i, r in enumerate(results[:5], 1):
            print(f"{i}. {r.payload.get('file_name', 'N/A')}")
    
    def interactive_mode(self):
        """Run interactive exploration mode."""
        self._load()
        
        print("\n" + "=" * 60)
        print("INTERACTIVE VECTOR EXPLORER")
        print("=" * 60)
        print("Commands:")
        print("  search <query>  - Search by text")
        print("  stats           - Show collection stats")
        print("  classes         - Show class distribution")
        print("  class <name>    - Show samples from class")
        print("  quit            - Exit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == "quit" or cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "search":
                    if arg:
                        self.search_by_text(arg)
                    else:
                        print("Usage: search <query>")
                elif cmd == "stats":
                    self.get_stats()
                elif cmd == "classes":
                    self.get_class_distribution()
                elif cmd == "class":
                    if arg:
                        self.find_similar_to_class(arg)
                    else:
                        print("Usage: class <class_name>")
                else:
                    # Treat as search query
                    self.search_by_text(user_input)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive vector database explorer")
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Path to Qdrant data")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Embedding model")
    parser.add_argument("--search", help="Direct search query (non-interactive)")
    args = parser.parse_args()
    
    explorer = VectorExplorer(args.qdrant_path, args.model)
    
    if args.search:
        explorer.search_by_text(args.search)
    else:
        explorer.interactive_mode()


if __name__ == "__main__":
    main()
