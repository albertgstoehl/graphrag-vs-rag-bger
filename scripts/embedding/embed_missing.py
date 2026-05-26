#!/usr/bin/env python3
"""Embed and upload missing swiss_rulings_chunked decisions into the Qdrant
`bger` collection.

Background: during the 2026-03 bulk embedding run, `embed_clean.py` silently
dropped batches whenever TEI was momentarily unreachable. The resume logic
also assumed every existing shard contained exactly SAVE_EVERY*BATCH_SIZE
chunks. Combined, those bugs left ~21'244 swiss_rulings decisions out of
Qdrant (3.3 %), without recording which chunk_ids were lost.

This script reconstructs the gap by:
  1. Scrolling Qdrant for every (decision_id, chunk_index) currently present.
  2. Reading the local prechunked arrow dataset to find chunk_ids that should
     exist but don't.
  3. Embedding those missing chunks via TEI, retrying on transient failures
     and logging chunk_ids that exceed the retry budget so they can be picked
     up by a follow-up run.
  4. Upserting points into Qdrant with the same payload schema as the
     historical points (decision_id, chunk_id, chunk_index, num_chunks,
     file_number, date, date_ms, language, court, source).

Run on aiserver01 where TEI (port 8010) and Qdrant (port 6333) and the
prechunked dataset all live.
"""
import argparse
import json
import os
import sys
import time
import uuid
from typing import Iterable

import pyarrow.ipc as ipc
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    PointStruct,
)

DEFAULT_DATASET = "/data/thesis/workspace/data/prechunked/swiss_rulings_chunked"
DEFAULT_QDRANT = "http://127.0.0.1:6333"
DEFAULT_TEI = "http://127.0.0.1:8010/embed"
COLLECTION = "bger"
SOURCE_LABEL = "swiss_rulings_chunked"
BATCH = 32
RETRY_LIMIT = 5
RETRY_BACKOFF = 4.0


def embed_batch(tei_url: str, texts: list[str]) -> list[list[float]]:
    """Call TEI /embed with retry. Raises if RETRY_LIMIT is exceeded."""
    last_exc: Exception | None = None
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.post(
                tei_url, json={"inputs": texts}, timeout=120
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            sleep_s = RETRY_BACKOFF * (2**attempt)
            print(
                f"  TEI failed (attempt {attempt+1}/{RETRY_LIMIT}): "
                f"{exc} — sleeping {sleep_s:.0f}s",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"TEI exhausted retries: {last_exc}") from last_exc


def load_prechunked(path: str):
    """Iterate over (decision_id, chunk_index, payload_dict) of every chunk."""
    arrow_files = sorted(
        f for f in os.listdir(path) if f.endswith(".arrow")
    )
    for fn in arrow_files:
        with open(os.path.join(path, fn), "rb") as fh:
            try:
                t = ipc.open_file(fh).read_all()
            except Exception:
                fh.seek(0)
                t = ipc.open_stream(fh).read_all()
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        n = t.num_rows
        for i in range(n):
            yield {k: cols[k][i] for k in cols}


def chunks_to_upload(
    dataset_path: str,
    missing_dids: set[str],
    already_present: set[tuple[str, int]],
):
    """Yield prechunked rows whose decision_id is missing AND whose
    (decision_id, chunk_index) isn't already in Qdrant."""
    for row in load_prechunked(dataset_path):
        did = row.get("decision_id")
        ci = row.get("chunk_index")
        if did in missing_dids and (did, ci) not in already_present:
            yield row


def collect_existing(client: QdrantClient) -> set[tuple[str, int]]:
    """Scroll every (decision_id, chunk_index) currently in the bger
    collection. Slow but exhaustive, the only safe way to be idempotent."""
    print("Scrolling existing Qdrant points (this takes a few minutes)…", flush=True)
    present: set[tuple[str, int]] = set()
    offset = None
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="source",
                                     match=MatchValue(value=SOURCE_LABEL))]
            ),
            with_payload=["decision_id", "chunk_index"],
            with_vectors=False,
            limit=10000,
            offset=offset,
        )
        for p in points:
            d = p.payload.get("decision_id")
            ci = p.payload.get("chunk_index")
            if d is not None and ci is not None:
                present.add((d, int(ci)))
        if offset is None:
            break
        if len(present) % 200000 < 10000:
            print(f"  …{len(present)} pairs scrolled "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"Existing pairs: {len(present)} ({time.time()-t0:.0f}s)", flush=True)
    return present


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT)
    ap.add_argument("--tei", default=DEFAULT_TEI)
    ap.add_argument("--missing-ids-file", required=True,
                    help="JSON list of missing decision_ids OR JSON object "
                         "with key 'swiss_rulings'")
    ap.add_argument("--skip-log", default="/tmp/embed_missing_skipped.log")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many chunks (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do everything except the Qdrant upsert")
    args = ap.parse_args()

    raw = json.load(open(args.missing_ids_file))
    if isinstance(raw, dict):
        missing_dids = set(raw.get("swiss_rulings", []))
    else:
        missing_dids = set(raw)
    print(f"Missing decision_ids: {len(missing_dids)}", flush=True)

    client = QdrantClient(url=args.qdrant, timeout=120)
    existing = collect_existing(client)

    queue: list[dict] = []
    n_yield = 0
    for row in chunks_to_upload(args.dataset, missing_dids, existing):
        queue.append(row)
        n_yield += 1
        if args.limit and n_yield >= args.limit:
            break
    print(f"Chunks to embed and upload: {len(queue)}", flush=True)
    if not queue:
        print("Nothing to do.", flush=True)
        return 0

    embedded = 0
    uploaded = 0
    skipped_log = open(args.skip_log, "a")
    t0 = time.time()

    for start in range(0, len(queue), BATCH):
        batch = queue[start : start + BATCH]
        texts = [r["text"] for r in batch]
        try:
            vecs = embed_batch(args.tei, texts)
        except Exception as exc:
            for r in batch:
                skipped_log.write(f"{r['chunk_id']}\t{exc}\n")
            skipped_log.flush()
            print(f"  Batch {start//BATCH+1} | SKIP {len(batch)} "
                  f"chunks: {exc}", flush=True)
            continue
        embedded += len(batch)

        points = []
        for row, vec in zip(batch, vecs):
            date_str = row.get("date") or ""
            try:
                date_ms = int(date_str) if date_str else None
            except ValueError:
                date_ms = None
            payload = {
                "decision_id": row["decision_id"],
                "chunk_id": row["chunk_id"],
                "chunk_index": int(row["chunk_index"]),
                "num_chunks": int(row.get("num_chunks") or 1),
                "text": row["text"],
                "file_number": row.get("file_number") or "",
                "date": date_str,
                "language": row.get("language") or "",
                "court": row.get("court") or "",
                "source": SOURCE_LABEL,
            }
            if date_ms is not None:
                payload["date_ms"] = date_ms
            # Use a deterministic UUIDv5 derived from chunk_id so re-runs of
            # this same script are idempotent at the point-id level.
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, row["chunk_id"]))
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        if not args.dry_run:
            client.upsert(collection_name=COLLECTION, points=points)
        uploaded += len(points)

        if (start // BATCH) % 20 == 0 or start + BATCH >= len(queue):
            rate = embedded / (time.time() - t0) if time.time() - t0 > 0 else 0
            eta_m = (len(queue) - start - BATCH) / max(rate, 0.1) / 60
            print(f"  Batch {start//BATCH+1:5d} | "
                  f"embedded {embedded}/{len(queue)} | "
                  f"uploaded {uploaded} | "
                  f"rate {rate:.0f}/s | ETA {eta_m:.1f}m",
                  flush=True)

    skipped_log.close()
    print(f"\nDone. Embedded {embedded}, uploaded {uploaded}, "
          f"skipped logged to {args.skip_log}", flush=True)
    print(f"Total time: {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
