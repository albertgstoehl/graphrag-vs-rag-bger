#!/usr/bin/env python3
"""Bulk-load numpy embedding shards into Qdrant.

Usage: python scripts/load_qdrant.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())

from datasets import load_from_disk
from qdrant_client import QdrantClient, models

EMBED_DIR = os.environ.get("EMBED_DIR", "/data/thesis/embeddings")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "bger"
BATCH_SIZE = 1000  # vectors per upload call

DATASETS = [
    "swiss_rulings_chunked",
    "swiss_leading_decisions_chunked",
]


def load_dataset(ds_name: str):
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    emb_dir = os.path.join(EMBED_DIR, ds_name)
    print(f"\n{'='*60}", flush=True)
    print(f"Load: {ds_name} → Qdrant ({QDRANT_URL})", flush=True)
    print(f"{'='*60}", flush=True)

    # Load dataset for metadata
    path = f"data/prechunked/{ds_name}"
    print(f"Loading {path}...", flush=True)
    ds = load_from_disk(path)
    total = len(ds)

    # Load all embedding shards
    shards = sorted([f for f in os.listdir(emb_dir) if f.endswith(".npy")])
    print(f"Loading {len(shards)} embedding shards...", flush=True)
    all_embs = np.concatenate([np.load(os.path.join(emb_dir, s)) for s in shards])
    print(f"Loaded {all_embs.shape[0]} embeddings ({all_embs.shape[1]}d)", flush=True)

    assert all_embs.shape[0] >= total, f"Embedding count {all_embs.shape[0]} < dataset {total}"

    # Create collection (only on first dataset)
    if not client.collection_exists(COLLECTION):
        print(f"Creating collection '{COLLECTION}' with indexing disabled...", flush=True)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=all_embs.shape[1],
                distance=models.Distance.COSINE,
            ),
            # Disable indexing during bulk upload for max speed
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
    else:
        info = client.get_collection(COLLECTION)
        print(f"Collection exists: {info.points_count} points", flush=True)

    # Bulk upload
    t0 = time.time()
    stored = 0

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        chunk = ds[start:end]
        keys = list(chunk.keys())
        batch = [dict(zip(keys, vals)) for vals in zip(*chunk.values())]

        points = []
        for i, doc in enumerate(batch):
            idx = start + i
            point_id = stored + i  # use sequential int IDs for speed
            points.append(models.PointStruct(
                id=point_id,
                vector=all_embs[idx].tolist(),
                payload={
                    "chunk_id": doc.get("chunk_id", doc["decision_id"]),
                    "text": doc["text"],
                    "decision_id": str(doc.get("decision_id", "")),
                    "chunk_index": int(doc.get("chunk_index", 0)),
                    "num_chunks": int(doc.get("num_chunks", 1)),
                    "file_number": str(doc.get("file_number") or ""),
                    "date": str(doc.get("date") or ""),
                    "language": str(doc.get("language") or ""),
                    "court": str(doc.get("court") or ""),
                    "source": ds_name,
                },
            ))

        client.upsert(collection_name=COLLECTION, points=points, wait=False)
        stored += len(points)

        if stored % 50000 < BATCH_SIZE:
            elapsed = time.time() - t0
            rate = stored / elapsed if elapsed > 0 else 0
            eta_m = (total - end) / rate / 60 if rate > 0 else 0
            print(f"  {stored:>8d}/{total} | {rate:.0f}/s | ETA {eta_m:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone loading {ds_name}: {stored} in {elapsed:.0f}s ({stored/elapsed:.0f}/s)", flush=True)

    return stored


def main():
    client = QdrantClient(url=QDRANT_URL, timeout=300)

    # Delete existing collection for clean start
    if client.collection_exists(COLLECTION):
        print(f"Deleting existing collection '{COLLECTION}'...", flush=True)
        client.delete_collection(COLLECTION)

    total_stored = 0
    for ds_name in DATASETS:
        total_stored += load_dataset(ds_name)

    # Re-enable indexing now that all data is loaded
    print("\nRe-enabling HNSW indexing...", flush=True)
    client.update_collection(
        collection_name=COLLECTION,
        optimizer_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )

    print(f"\nTotal: {total_stored} vectors loaded. HNSW index building in background.", flush=True)

    # Wait for indexing to complete
    print("Waiting for indexing to complete...", flush=True)
    while True:
        info = client.get_collection(COLLECTION)
        if info.status == models.CollectionStatus.GREEN:
            print(f"Indexing complete! Collection status: {info.status}", flush=True)
            break
        time.sleep(5)
        print(f"  Status: {info.status}, points: {info.points_count}, indexed: {info.indexed_vectors_count}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
