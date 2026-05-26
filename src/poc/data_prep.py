import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")

# gemini-embedding-001 max input is 2048 tokens
CHUNK_SIZE = 1024      # tokens per chunk
CHUNK_OVERLAP = 128    # token overlap between chunks


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_encoder.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token-based chunks.

    Returns list of text chunks. Short texts (<= chunk_size) return as single chunk.
    """
    tokens = _encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_encoder.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def iter_all_rulings_chunked(batch_size: int = 512):
    """Yield batches of chunks from ALL swiss_rulings. Memory-efficient.

    Each chunk has decision_id, chunk_index, and the chunk text.
    Yields: (chunk_batch: list[dict], stats: dict)
    """
    from src.poc.dataset_loader import load

    print("  Loading dataset from disk...", flush=True)
    dataset = load("swiss_rulings")
    total_rows = len(dataset) if hasattr(dataset, '__len__') else "unknown"
    print(f"  Dataset loaded: {total_rows} rows", flush=True)

    batch = []
    skipped_empty = 0
    total_chunks = 0
    processed = 0

    for item in dataset:
        processed += 1
        if processed % 50000 == 0:
            print(f"  Row {processed}/{total_rows} | {skipped_empty} empty, {total_chunks} chunks total", flush=True)

        full_text = item.get("full_text", "")
        if not full_text:
            skipped_empty += 1
            continue

        decision_id = item.get("decision_id", "unknown")
        meta = {
            "file_number": item.get("file_number", ""),
            "date": item.get("date", ""),
            "language": item.get("language", ""),
            "court": item.get("court", ""),
        }

        chunks = chunk_text(full_text)
        for idx, chunk in enumerate(chunks):
            total_chunks += 1
            batch.append({
                "decision_id": decision_id,
                "chunk_id": f"{decision_id}_c{idx}",
                "chunk_index": idx,
                "num_chunks": len(chunks),
                "text": chunk,
                **meta,
            })

            if len(batch) >= batch_size:
                yield batch, {"processed": processed, "total": total_rows, "empty": skipped_empty, "chunks": total_chunks}
                batch = []

    if batch:
        yield batch, {"processed": processed, "total": total_rows, "empty": skipped_empty, "chunks": total_chunks}

    print(f"  DONE: {processed} rows, {skipped_empty} empty, {total_chunks} chunks", flush=True)


def load_bger_sample(
    limit: int = 500,
    max_tokens: int = 7000,
    only_ids: set[str] | None = None,
) -> list[dict]:
    """Load BGer decisions under max_tokens from HuggingFace."""
    from src.poc.dataset_loader import load

    dataset = load("swiss_rulings")

    docs = []
    for item in dataset:
        if item.get("court") != "CH_BGer":
            continue
        if only_ids is not None and item.get("decision_id") not in only_ids:
            continue

        text = item.get("full_text", "")
        if not text or count_tokens(text) >= max_tokens:
            continue

        docs.append({
            "decision_id": item.get("decision_id", "unknown"),
            "file_number": item.get("file_number", ""),
            "date": item.get("date", ""),
            "language": item.get("language", ""),
            "court": item.get("court", ""),
            "text": text,
        })

        if len(docs) >= limit:
            break

    return docs
