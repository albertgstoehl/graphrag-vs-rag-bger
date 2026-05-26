#!/usr/bin/env python3
"""Phase 1 of the bulk embedding pipeline.

Reads a pre-chunked HuggingFace Arrow dataset, embeds every chunk via async
HTTP calls to one or more TEI endpoints, and persists the resulting vectors
as sharded numpy files under ``EMBED_DIR``. The companion script
``load_qdrant.py`` consumes those shards and writes the points (vectors +
payload) into a Qdrant collection.

Historical note, an earlier iteration of this script also contained a
ChromaDB writer. The project moved to Qdrant for production use and that
branch was removed in May 2026.
"""
import os, sys, time, asyncio
import numpy as np

# Must run from /data/thesis/workspace
sys.path.insert(0, os.getcwd())

import aiohttp
from datasets import load_from_disk

# ── Config ──────────────────────────────────────────────────────────────
BATCH_SIZE = 352          # 11 GPUs × 32 texts = 352 (skip GPU 2)
MAX_PER_CALL = 32         # texts per TEI endpoint call
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


# ── Main ────────────────────────────────────────────────────────────────

async def main():
    for ds_name in DATASETS:
        await run_embed(ds_name)

    print(f"\n{'='*60}", flush=True)
    print("Embedding complete. Next step: scripts/embedding/load_qdrant.py", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
