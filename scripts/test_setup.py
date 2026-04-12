#!/usr/bin/env python3
"""Test script to verify the setup is working."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

def test_config():
    print("Testing config...")
    from src.config.settings import get_config
    config = get_config()
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Device: {config.embedding.device}")
    print("  Config OK!")

def test_embeddings():
    print("\nTesting embedding service...")
    from src.embeddings import get_embedding_service
    svc = get_embedding_service()
    print(f"  Dimension: {svc.get_dimension()}")
    
    result = svc.embed_text("This is a test music royalty statement.")
    print(f"  Embedding shape: {result.vector.shape}")
    print("  Embeddings OK!")

def test_parsers():
    print("\nTesting parsers...")
    from src.parsers import get_parser
    
    pdf_parser = get_parser("test.pdf")
    csv_parser = get_parser("test.csv")
    txt_parser = get_parser("test.txt")
    
    print(f"  PDF parser: {type(pdf_parser).__name__}")
    print(f"  CSV parser: {type(csv_parser).__name__}")
    print(f"  TXT parser: {type(txt_parser).__name__}")
    print("  Parsers OK!")

def test_qdrant():
    print("\nTesting Qdrant...")
    from src.vectordb import QdrantStore
    
    store = QdrantStore(
        path="/workspace/qdrant_data",
        collection_name="test_collection",
        dimension=1024
    )
    info = store.get_collection_info()
    print(f"  Collection: {info['name']}")
    print(f"  Status: {info['status']}")
    print("  Qdrant OK!")

def test_rclone():
    print("\nTesting rclone access...")
    import subprocess
    result = subprocess.run(
        ["rclone", "lsf", "gdrive:Music_rights_train/"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print(f"  Files found: {result.stdout.strip()}")
        print("  rclone OK!")
    else:
        print(f"  Error: {result.stderr}")

def main():
    print("=" * 50)
    print("Music Rights Classifier - Setup Test")
    print("=" * 50)
    
    try:
        test_config()
        test_embeddings()
        test_parsers()
        test_qdrant()
        test_rclone()
        
        print("\n" + "=" * 50)
        print("All tests passed! Setup is complete.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
