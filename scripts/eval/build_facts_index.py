#!/usr/bin/env python3
"""
build_facts_index.py
====================
Extracts the `facts` field for every Bundesgerichts-Decision in the
`rcds/swiss_doc2doc_ir` dataset and writes a `decision_id → facts` index
to `data/eval/facts_index.jsonl`.

Why this artefact exists
------------------------
Stage 1 of the evaluation pipeline (`01_sample_queries.py`) used to derive
the query text by concatenating the first three Qdrant chunks of the source
decision. Those chunks contain the FULL decision text including the
`considerations` section, which in Swiss court rulings carries the explicit
BGE citations to precedent cases. Putting that text into the query embedding
leaks the ground-truth labels into the query and inflates retrieval metrics.

The HF dataset stores `facts` and `considerations` as separate string
columns. By materialising just the `facts` column into a lookup, Stage 1 can
use a leak-free query text without re-introducing a runtime HuggingFace
dependency.

Output schema (one JSON object per line):
    {"decision_id": "<uuid>", "language": "de|fr|it", "facts": "<text>"}

Usage:
    .venv/bin/python scripts/eval/build_facts_index.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("build_facts_index")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path(os.environ.get(
    "FACTS_INDEX_PATH", str(REPO_ROOT / "data" / "eval" / "facts_index.jsonl")
))
DATASET_NAME = "rcds/swiss_doc2doc_ir"
KEEP_LANGUAGES = {"de", "fr", "it"}


def main() -> int:
    log.info("Loading dataset %s ...", DATASET_NAME)
    t0 = time.time()
    ds = load_dataset(DATASET_NAME)
    log.info("Loaded in %.1fs. Splits: %s", time.time() - t0, list(ds.keys()))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_lang = 0
    seen_ids: set = set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for split_name, split in ds.items():
            log.info("Processing split %s (%d rows) ...", split_name, len(split))
            for row in tqdm(split, desc=split_name, unit="rows"):
                did = row.get("decision_id")
                if not did or did in seen_ids:
                    continue
                lang = (row.get("language") or "").lower()
                if lang not in KEEP_LANGUAGES:
                    skipped_lang += 1
                    continue
                facts = row.get("facts") or ""
                out.write(json.dumps({
                    "decision_id": did,
                    "language": lang,
                    "facts": facts,
                }, ensure_ascii=False) + "\n")
                seen_ids.add(did)
                written += 1

    log.info("Wrote %d entries to %s", written, OUTPUT_PATH)
    log.info("  skipped (wrong language): %d", skipped_lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
