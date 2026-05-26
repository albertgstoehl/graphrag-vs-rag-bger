#!/usr/bin/env python3
"""Clean standalone embedding script for aiserver01.

Phase 1: Embed all chunks via async aiohttp → save as numpy files to disk.
Phase 2: Bulk-load numpy files into ChromaDB (separate step, run via --load).
"""
import os, sys, time, asyncio, argparse
import numpy as np

# Must run from /data/thesis/workspace
sys.path.insert(0, os.getcwd())

import aiohttp
from datasets import load_from_disk
import chromadb
from chromadb.config import Settings

# ── Config ──────────────────────────────────────────────────────────────
BATCH_SIZE = 352          # 11 GPUs × 32 texts = 352 (skip GPU 2)
MAX_PER_CALL = 32         # texts per TEI endpoint call
CHROMA_MAX = 5000         # ChromaDB max per add() call
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/data/thesis/chromadb")
EMBED_DIR = os.environ.get("EMBED_DIR", "/data/thesis/embeddings")
EMBED_TIMEOUT = int(os.environ.get("EMBED_TIMEOUT_SECONDS", "120"))

# Ports 8010-8021, skip 8012 (GPU 2 broken, no container)
_DEFAULT_PORTS = [p for p in range(8010, 8022) if p != 8012]
_EMBED_BASE_URLS = os.environ.get(
    "EMBED_BASE_URL", ",".join(f"http://localhost:{p}" for p in _DEFAULT_PORTS)
).split(",")
ENDPOINTS = [f"{u.strip().rstrip('/')}/v1/embeddings" for u in _EMBED_BASE_URLS]

DATASETS = [
    "swiss_rulings_chunked",
    "swiss_leading_decisions_chunked",
]


# ── Async embedding ─────────────────────────────────────────────────────

async def embed_batch_async(session: aiohttp.ClientSession, texts: list[str]) -> list[list[float]]:
    """Embed texts using all TEI endpoints in parallel via aiohttp."""
    chunks = [texts[i:i + MAX_PER_CALL] for i in range(0, len(texts), MAX_PER_CALL)]
    num_ep = len(ENDPOINTS)

    async def fetch_with_fallback(ep_idx: int, inputs: list[str]) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=EMBED_TIMEOUT)
        primary_ep = ENDPOINTS[ep_idx % num_ep]
        for attempt in range(3):
            ep = primary_ep if attempt == 0 else ENDPOINTS[(ep_idx + attempt) % num_ep]
            try:
                async with session.post(ep, json={"input": inputs}, timeout=timeout) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
                    data = sorted(payload["data"], key=lambda x: x.get("index", 0))
                    return [d["embedding"] for d in data]
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5)

    tasks = [fetch_with_fallback(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    all_embeddings = []
    for r in results:
        all_embeddings.extend(r)
    return all_embeddings


# ── Phase 1: Embed → disk ──────────────────────────────────────────────

SAVE_EVERY = 100  # save a numpy shard every N batches (~35,200 vectors, ~137MB)

async def run_embed(ds_name: str):
    """Embed all chunks and save embeddings as numpy shards."""
    out_dir = os.path.join(EMBED_DIR, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Embed: {ds_name} → {out_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    # Check for existing shards (resume support). Use ACTUAL shard sizes,
    # not the SAVE_EVERY*BATCH_SIZE assumption — silent batch skips during
    # earlier runs leave shorter shards, and the assumption drifts the
    # dataset cursor past chunks that were never embedded.
    existing_shards = sorted([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    start_from = sum(
        np.load(os.path.join(out_dir, s), mmap_mode="r").shape[0]
        for s in existing_shards
    )
    if start_from > 0:
        print(f"Resuming: found {len(existing_shards)} shards covering {start_from} chunks", flush=True)

    path = f"data/prechunked/{ds_name}"
    print(f"Loading {path}...", flush=True)
    ds = load_from_disk(path)
    total = len(ds)
    print(f"Loaded {total} chunks", flush=True)

    if start_from >= total:
        print("Already complete!", flush=True)
        return

    embedded = 0
    batch_num = 0
    shard_idx = len(existing_shards)
    acc_embs = []
    t0 = time.time()

    async with aiohttp.ClientSession() as session:
        for start in range(start_from, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_num += 1

            chunk = ds[start:end]
            keys = list(chunk.keys())
            batch = [dict(zip(keys, vals)) for vals in zip(*chunk.values())]
            texts = [d["text"] for d in batch]

            t_batch = time.time()

            try:
                embeddings = await embed_batch_async(session, texts)
            except Exception as e:
                print(f"  Batch {batch_num} | ERROR: {e} | retrying...", flush=True)
                await asyncio.sleep(5)
                try:
                    embeddings = await embed_batch_async(session, texts)
                except Exception as e2:
                    # Record skipped chunk_ids so a recovery pass can pick
                    # them up. Silent skips here in 2026-03 caused a 3.3%
                    # ingest gap (~21k decisions) that was not noticed
                    # until thesis-time forensics.
                    skip_log = os.path.join(out_dir, "skipped_chunks.log")
                    with open(skip_log, "a") as fh:
                        for d in batch:
                            fh.write(f"{d.get('chunk_id', '')}\n")
                    print(f"  Batch {batch_num} | SKIP: {e2} | "
                          f"{len(batch)} chunk_ids logged to {skip_log}",
                          flush=True)
                    continue

            t_embed = time.time() - t_batch
            acc_embs.extend(embeddings)
            embedded += len(texts)

            # Save shard
            if batch_num % SAVE_EVERY == 0 or end >= total:
                arr = np.array(acc_embs, dtype=np.float32)
                shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.npy")
                np.save(shard_path, arr)
                print(f"  Saved {shard_path}: {arr.shape}", flush=True)
                shard_idx += 1
                acc_embs = []

            elapsed = time.time() - t0
            rate = embedded / elapsed if elapsed > 0 else 0
            eta_m = (total - end) / rate / 60 if rate > 0 else 0

            if batch_num <= 5 or batch_num % 25 == 0:
                print(f"  Batch {batch_num:5d} | {end:>8d}/{total} | embed={t_embed:.2f}s | {rate:.0f}/s | ETA {eta_m:.0f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone embedding {ds_name}: {embedded} in {elapsed:.0f}s ({embedded/elapsed:.0f}/s)", flush=True)


# ── Phase 2: Load numpy → ChromaDB ─────────────────────────────────────

def run_load(ds_name: str):
    """Load saved embeddings + dataset into ChromaDB."""
    emb_dir = os.path.join(EMBED_DIR, ds_name)

    print(f"\n{'='*60}", flush=True)
    print(f"Load: {ds_name} → ChromaDB ({PERSIST_DIR})", flush=True)
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

    # ChromaDB
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection(
        name="bger",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": 50000,
            "hnsw:sync_threshold": 200000,
            "hnsw:num_threads": 12,
        },
    )
    existing = coll.count()
    print(f"ChromaDB: {existing} existing vectors", flush=True)

    t0 = time.time()
    stored = 0
    for start in range(0, total, CHROMA_MAX):
        end = min(start + CHROMA_MAX, total)

        chunk = ds[start:end]
        keys = list(chunk.keys())
        batch = [dict(zip(keys, vals)) for vals in zip(*chunk.values())]

        ids = [d.get("chunk_id", d["decision_id"]) for d in batch]
        embs = all_embs[start:end].tolist()

        coll.add(
            ids=ids,
            embeddings=embs,
            documents=[d["text"] for d in batch],
            metadatas=[{
                "decision_id": str(d.get("decision_id", "")),
                "chunk_index": int(d.get("chunk_index", 0)),
                "num_chunks": int(d.get("num_chunks", 1)),
                "file_number": str(d.get("file_number") or ""),
                "date": str(d.get("date") or ""),
                "language": str(d.get("language") or ""),
                "court": str(d.get("court") or ""),
            } for d in batch],
        )
        stored += len(batch)

        if stored % 50000 < CHROMA_MAX:
            elapsed = time.time() - t0
            rate = stored / elapsed if elapsed > 0 else 0
            eta_m = (total - end) / rate / 60 if rate > 0 else 0
            print(f"  {stored:>8d}/{total} | {rate:.0f}/s | ETA {eta_m:.0f}m", flush=True)

    elapsed = time.time() - t0
    final = coll.count()
    print(f"\nDone loading {ds_name}: {stored} stored in {elapsed:.0f}s ({stored/elapsed:.0f}/s) | total: {final}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Phase 2: load embeddings into ChromaDB")
    args = parser.parse_args()

    if args.load:
        for ds_name in DATASETS:
            run_load(ds_name)
    else:
        for ds_name in DATASETS:
            await run_embed(ds_name)

    print(f"\n{'='*60}", flush=True)
    print("All phases complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
