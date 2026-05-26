"""Centralized HuggingFace dataset loading with local cache support.

Downloads datasets once to data/hf_cache/ and loads from disk on subsequent runs.
Falls back to streaming if local cache is not available.
"""

import os
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset

HF_CACHE_DIR = Path("data/hf_cache")

# All datasets used in the POC
DATASETS = {
    "swiss_rulings": "rcds/swiss_rulings",
    "swiss_doc2doc_ir": "rcds/swiss_doc2doc_ir",
    "swiss_legislation": "rcds/swiss_legislation",
}


def _local_path(name: str) -> Path:
    return HF_CACHE_DIR / name


def download_all():
    """Download all datasets to local disk. Run once."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, hf_id in DATASETS.items():
        dest = _local_path(name)
        if dest.exists():
            print(f"  {name}: already cached at {dest}")
            continue
        print(f"  {name}: downloading {hf_id}...")
        ds = load_dataset(hf_id, split="train")
        ds.save_to_disk(str(dest))
        print(f"  {name}: saved ({len(ds)} rows)")


def load(name: str, streaming_fallback: bool = True) -> Dataset:
    """Load a dataset by short name. Uses local cache if available.

    Args:
        name: One of 'swiss_rulings', 'swiss_doc2doc_ir', 'swiss_legislation'.
        streaming_fallback: If True and no local cache, stream from HF.

    Returns:
        Dataset (or IterableDataset if streaming).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    local = _local_path(name)
    if local.exists():
        return load_from_disk(str(local))

    hf_id = DATASETS[name]
    if streaming_fallback:
        print(f"  WARNING: {name} not cached locally, streaming from HF (slow)...")
        print(f"  Run: python -m src.poc.dataset_loader  to download first.")
        return load_dataset(hf_id, split="train", streaming=True)

    raise FileNotFoundError(
        f"Local cache for {name} not found at {local}. "
        f"Run: python -m src.poc.dataset_loader"
    )


if __name__ == "__main__":
    print("Downloading HuggingFace datasets to local cache...")
    download_all()
    print("Done! Datasets cached in", HF_CACHE_DIR)
