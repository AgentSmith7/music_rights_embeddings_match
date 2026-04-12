#!/usr/bin/env python3
"""Test embedding model on GPU."""

import time
from sentence_transformers import SentenceTransformer

# Try multiple models to find one that works
MODELS_TO_TRY = [
    ('BAAI/bge-large-en-v1.5', False),
    ('intfloat/e5-large-v2', False),
    ('sentence-transformers/all-mpnet-base-v2', False),
]

model = None
for model_name, trust_remote in MODELS_TO_TRY:
    print(f"Trying {model_name}...")
    try:
        start = time.time()
        model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote,
            device='cuda'
        )
        print(f"SUCCESS: {model_name} loaded in {time.time() - start:.1f}s")
        break
    except Exception as e:
        print(f"FAILED: {e}")
        continue

if model is None:
    print("No model worked!")
    exit(1)
print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Test single embedding
text = "This is a music royalty statement from Believe Digital showing streaming revenue."
start = time.time()
embedding = model.encode(text)
print(f"Single embedding: {embedding.shape}, took {(time.time() - start)*1000:.1f}ms")

# Test batch embedding
texts = [
    "Spotify streaming royalty report Q1 2024",
    "Artist: John Smith, Track: Summer Nights, Plays: 1,234,567",
    "Distribution agreement between Label X and Artist Y",
    "Mechanical royalty payment for song composition",
    "Performance rights organization statement",
] * 20  # 100 texts

start = time.time()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
elapsed = time.time() - start
print(f"Batch of {len(texts)} embeddings: {embeddings.shape}")
print(f"Throughput: {len(texts)/elapsed:.1f} texts/sec")

print("\nEmbedding model test PASSED!")
