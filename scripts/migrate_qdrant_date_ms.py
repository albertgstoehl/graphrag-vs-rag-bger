"""One-shot migration: add integer `date_ms` payload field to all points
in the `bger` Qdrant collection, then create an integer payload index.

v3: thread-pool over `date_index.json` (605 K decisions). Each worker
issues `set_payload` with a `decision_id == X` filter (decision_id is
already indexed in Qdrant, so the filter is O(log n)). Per-call latency
~30 ms — single-threaded gives 32/s; with 32 worker threads we expect
~600-1000/s and ~10-20 min total wall-clock.

Idempotent: re-running re-writes the same `date_ms` values.
"""

import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, Range, PayloadSchemaType,
)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION = "bger"
DATE_INDEX_PATH = os.environ.get("DATE_INDEX_PATH", "data/eval/date_index.json")
N_WORKERS = int(os.environ.get("MIGRATE_WORKERS", "32"))


_progress_lock = Lock()
_progress = {"done": 0, "failed": 0, "skipped": 0}


def write_one(client: QdrantClient, did: str, date_ms: int) -> None:
    if date_ms <= 0:
        with _progress_lock:
            _progress["skipped"] += 1
        return
    try:
        client.set_payload(
            collection_name=COLLECTION,
            payload={"date_ms": date_ms},
            points=Filter(must=[
                FieldCondition(key="decision_id", match=MatchValue(value=did))
            ]),
            wait=False,
        )
        with _progress_lock:
            _progress["done"] += 1
    except Exception as e:
        with _progress_lock:
            _progress["failed"] += 1
            if _progress["failed"] < 5:
                print(f"[migrate] WARN failed for {did[:8]}: {e}", flush=True)


def main():
    print(f"[migrate] loading {DATE_INDEX_PATH} ...", flush=True)
    with open(DATE_INDEX_PATH) as f:
        date_index = json.load(f)
    total = len(date_index)
    print(f"[migrate] {total} decision→date entries, {N_WORKERS} workers", flush=True)

    # Each worker has its own QdrantClient (httpx clients are not thread-safe).
    clients = [
        QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False, timeout=30)
        for _ in range(N_WORKERS)
    ]
    info = clients[0].get_collection(COLLECTION)
    print(f"[migrate] collection={COLLECTION} points={info.points_count}", flush=True)

    start = time.time()
    last_log = start

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        # Round-robin assign clients to tasks via the worker index.
        futures = []
        items = list(date_index.items())
        for i, (did, date_ms) in enumerate(items):
            client = clients[i % N_WORKERS]
            futures.append(ex.submit(write_one, client, did, int(date_ms)))

        # Progress loop
        for j, fut in enumerate(as_completed(futures)):
            now = time.time()
            if now - last_log >= 5:
                with _progress_lock:
                    done = _progress["done"]
                    failed = _progress["failed"]
                    skipped = _progress["skipped"]
                processed = done + failed + skipped
                elapsed = now - start
                rate = processed / elapsed if elapsed else 0
                eta = (total - processed) / rate if rate else 0
                print(
                    f"[migrate] {processed}/{total} ({100*processed/total:.1f}%) "
                    f"rate={rate:.0f}/s eta={eta:.0f}s "
                    f"done={done} skipped={skipped} failed={failed}",
                    flush=True,
                )
                last_log = now

    elapsed = time.time() - start
    with _progress_lock:
        print(
            f"[migrate] writes complete: done={_progress['done']} "
            f"skipped={_progress['skipped']} failed={_progress['failed']} "
            f"in {elapsed:.0f}s",
            flush=True,
        )

    # Barrier: synchronous wait=True on a single decision to flush the queue.
    print("[migrate] barrier (wait=True)...", flush=True)
    first_did, first_date = next(iter(date_index.items()))
    clients[0].set_payload(
        collection_name=COLLECTION,
        payload={"date_ms": int(first_date)},
        points=Filter(must=[
            FieldCondition(key="decision_id", match=MatchValue(value=first_did))
        ]),
        wait=True,
    )
    print("[migrate] flushed.", flush=True)

    print("[migrate] creating integer payload index on date_ms...", flush=True)
    t0 = time.time()
    clients[0].create_payload_index(
        collection_name=COLLECTION,
        field_name="date_ms",
        field_schema=PayloadSchemaType.INTEGER,
    )
    print(f"[migrate] index created in {time.time()-t0:.0f}s.", flush=True)

    threshold_2001 = 978307200000
    t0 = time.time()
    count = clients[0].count(
        collection_name=COLLECTION,
        count_filter=Filter(must=[
            FieldCondition(key="date_ms", range=Range(lt=threshold_2001))
        ]),
        exact=True,
    ).count
    print(
        f"[migrate] verify: {count} points have date_ms < 2001 "
        f"(filter latency {time.time()-t0:.2f}s)",
        flush=True,
    )

    sample = clients[0].scroll(
        collection_name=COLLECTION,
        limit=3,
        scroll_filter=Filter(must=[
            FieldCondition(key="date_ms", range=Range(lt=threshold_2001))
        ]),
        with_payload=["decision_id", "date", "date_ms"],
    )
    print("[migrate] sample (date_ms < 2001):", flush=True)
    for p in sample[0]:
        print(f"  {p.payload}", flush=True)

    print(f"[migrate] done in {time.time()-start:.0f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
