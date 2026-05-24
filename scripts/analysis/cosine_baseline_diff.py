#!/usr/bin/env python3
"""
cosine_baseline_diff.py
=======================
Validates the claim that cosine similarity in the BGE-M3 embedding space is
NOT discriminative for citation relevance in Swiss federal court rulings.

For each sampled query, compute the mean cosine similarity between the
query's chunk-0 vector and the chunk-0 vectors of (a) its cited rulings
(ground truth) and (b) randomly chosen valid rulings. The per-query
difference Delta = mean(cos_cited) - mean(cos_random) is aggregated across
the sample.

If the claim holds, |Delta| is well below 0.01, meaning a system that only
ranks by cosine cannot reliably distinguish cited from random rulings.

USAGE:
    QDRANT_HOST=localhost python scripts/analysis/cosine_baseline_diff.py
    SAMPLE_SIZE=200 python scripts/analysis/cosine_baseline_diff.py
"""

import json
import os
import random
import statistics
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION = "bger"

EVAL_DIR = Path(os.environ.get("EVAL_DIR", "data/eval"))
QUERIES_PATH = EVAL_DIR / "eval_queries.jsonl"
VALID_IDS_PATH = EVAL_DIR / "valid_ids.json"

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "200"))
RANDOM_PER_QUERY = int(os.environ.get("RANDOM_PER_QUERY", "20"))
BATCH = 64
SEED = 42


def fetch_chunk0_vectors(client: QdrantClient, decision_ids: list[str]) -> dict[str, np.ndarray]:
    """Batch-fetch chunk-0 vectors for the given decision IDs."""
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(decision_ids), BATCH):
        batch_ids = decision_ids[i : i + BATCH]
        scroll, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="decision_id", match=MatchAny(any=batch_ids)),
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            ]),
            with_vectors=True,
            with_payload=["decision_id"],
            limit=len(batch_ids),
        )
        for point in scroll:
            did = point.payload["decision_id"]
            out[did] = np.asarray(point.vector, dtype=np.float32)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}, collection={COLLECTION}")
    print(f"Sample size: {SAMPLE_SIZE} queries, {RANDOM_PER_QUERY} random per query")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60, check_compatibility=False)

    queries = [json.loads(l) for l in QUERIES_PATH.open()]
    valid_ids = json.loads(VALID_IDS_PATH.read_text())
    print(f"Loaded {len(queries)} queries, {len(valid_ids)} valid IDs")

    rng = random.Random(SEED)
    sample = rng.sample(queries, min(SAMPLE_SIZE, len(queries)))

    # Collect all decision IDs we need
    needed: set[str] = set()
    for q in sample:
        needed.add(q["query_id"])
        for cid in q.get("ground_truth_cases", []):
            needed.add(cid)
    # add random pool
    random_pool = rng.sample(valid_ids, min(SAMPLE_SIZE * RANDOM_PER_QUERY, len(valid_ids)))
    needed.update(random_pool)

    print(f"Fetching chunk-0 vectors for {len(needed)} decision IDs ...")
    vecs = fetch_chunk0_vectors(client, sorted(needed))
    print(f"Retrieved {len(vecs)} vectors")

    deltas = []
    cited_means = []
    random_means = []
    rng2 = random.Random(SEED + 1)

    for q in sample:
        qid = q["query_id"]
        if qid not in vecs:
            continue
        qv = vecs[qid]

        cited = [c for c in q.get("ground_truth_cases", []) if c in vecs]
        if not cited:
            continue
        cos_cited = [cosine(qv, vecs[c]) for c in cited]

        # Per-query random sample from the random_pool, exclude self and cited
        excl = {qid, *cited}
        pool = [r for r in random_pool if r in vecs and r not in excl]
        rs = rng2.sample(pool, min(RANDOM_PER_QUERY, len(pool)))
        if not rs:
            continue
        cos_random = [cosine(qv, vecs[r]) for r in rs]

        c_mean = statistics.mean(cos_cited)
        r_mean = statistics.mean(cos_random)
        cited_means.append(c_mean)
        random_means.append(r_mean)
        deltas.append(c_mean - r_mean)

    if not deltas:
        print("No valid queries produced a delta, aborting")
        return

    print(f"\nResult over n={len(deltas)} queries:")
    print(f"  mean cosine(q, cited)    = {statistics.mean(cited_means):.4f}")
    print(f"  mean cosine(q, random)   = {statistics.mean(random_means):.4f}")
    print(f"  mean delta (cited-random) = {statistics.mean(deltas):.4f}")
    print(f"  median delta              = {statistics.median(deltas):.4f}")
    print(f"  stdev delta               = {statistics.stdev(deltas):.4f}")
    print(f"  min / max                  = {min(deltas):.4f} / {max(deltas):.4f}")


if __name__ == "__main__":
    main()
