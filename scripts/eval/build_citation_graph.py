#!/usr/bin/env python3
"""
build_citation_graph.py
=======================
Builds a NetworkX directed citation graph from the swiss_doc2doc_ir dataset.

Dataset: rcds/swiss_doc2doc_ir (Apache Arrow format, HuggingFace cache)
 - Each row represents a Swiss Federal Supreme Court ruling (BGer case)
 - `decision_id`: UUID string identifying the ruling (source node, type=ruling)
 - `cited_rulings`: Python-list-encoded string of UUID strings → case-to-case edges
 - `laws`: Python-list-encoded string of UUID strings → case-to-law edges

Graph structure (NetworkX DiGraph):
 - Nodes: all unique document IDs (rulings + law articles)
   - node attr `source`: "ruling" or "law"
 - Edges: directed, from citing case to cited document
   - edge attr `type`: "case_to_case" or "case_to_law"

Output:
 - Prints summary statistics to stdout
 - Saves graph as pickle to data/eval/citation_graph.pkl (override via OUTPUT_DIR)

Usage:
    python scripts/eval/build_citation_graph.py
"""

import ast
import os
import pickle
import time

import networkx as nx
import pyarrow.ipc as ipc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the cached HuggingFace dataset shards for rcds/swiss_doc2doc_ir.
# Override via DATASET_DIR; default points into the local HF cache.
DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.expanduser(
        "~/.cache/huggingface/datasets/rcds___swiss_doc2doc_ir"
        "/default/0.0.0/1c3a1e5200f8577d485bdb6ba7f17b17ce5d0d79"
    ),
)

# All five Arrow shards (train × 3, validation, test)
ARROW_SHARDS = [
    "swiss_doc2doc_ir-train-00000-of-00003.arrow",
    "swiss_doc2doc_ir-train-00001-of-00003.arrow",
    "swiss_doc2doc_ir-train-00002-of-00003.arrow",
    "swiss_doc2doc_ir-validation.arrow",
    "swiss_doc2doc_ir-test.arrow",
]

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/eval")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "citation_graph.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_list_field(raw: str) -> list[str]:
    """
    Parse a field that is stored as a Python-repr list string, e.g.:
        "['uuid-1', 'uuid-2']"
    Returns an empty list on any parse failure or empty/null input.
    """
    if not raw:
        return []
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return result
        return []
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_graph() -> nx.DiGraph:
    """Read all Arrow shards and construct the citation DiGraph."""
    G = nx.DiGraph()

    total_rows = 0
    n_case_to_case = 0
    n_case_to_law = 0
    n_skipped_self = 0  # self-loop edges (citing_id == cited_id) — we skip these

    start = time.time()

    for shard_name in ARROW_SHARDS:
        shard_path = os.path.join(DATASET_DIR, shard_name)
        print(f"  Loading shard: {shard_name} ...", flush=True)

        with open(shard_path, "rb") as fh:
            reader = ipc.open_stream(fh)
            table = reader.read_all()

        n_rows = table.num_rows
        total_rows += n_rows

        for i in range(n_rows):
            decision_id: str = table["decision_id"][i].as_py()
            if not decision_id:
                continue

            # Ensure the source ruling node exists with its metadata
            if not G.has_node(decision_id):
                G.add_node(decision_id, source="ruling")
            elif G.nodes[decision_id].get("source") != "ruling":
                # Node was previously added as a "law" target; correct it
                G.nodes[decision_id]["source"] = "ruling"

            # --- case-to-case edges ---
            cited_rulings_raw: str = table["cited_rulings"][i].as_py()
            for cited_id in parse_list_field(cited_rulings_raw):
                if not cited_id or cited_id == decision_id:
                    n_skipped_self += 1
                    continue
                if not G.has_node(cited_id):
                    # Cited ruling may not appear as a source row (out-of-corpus)
                    G.add_node(cited_id, source="ruling")
                G.add_edge(decision_id, cited_id, type="case_to_case")
                n_case_to_case += 1

            # --- case-to-law edges ---
            laws_raw: str = table["laws"][i].as_py()
            for law_id in parse_list_field(laws_raw):
                if not law_id or law_id == decision_id:
                    n_skipped_self += 1
                    continue
                if not G.has_node(law_id):
                    G.add_node(law_id, source="law")
                # If the law_id was already tagged as a ruling (unlikely but
                # possible due to shared UUID space), keep the ruling tag.
                G.add_edge(decision_id, law_id, type="case_to_law")
                n_case_to_law += 1

        print(f"    → {n_rows} rows processed ({total_rows} total so far)", flush=True)

    elapsed = time.time() - start

    print(f"\nGraph construction complete in {elapsed:.1f}s")
    print(f"  Source rows processed : {total_rows:,}")
    print(f"  Self-loop edges skipped: {n_skipped_self:,}")
    print(f"\n=== Graph Statistics ===")
    print(f"  Total nodes           : {G.number_of_nodes():,}")
    print(f"  Total edges           : {G.number_of_edges():,}")
    print(f"  case_to_case edges    : {n_case_to_case:,}")
    print(f"  case_to_law edges     : {n_case_to_law:,}")

    # Node breakdown by source type
    ruling_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("source") == "ruling")
    law_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("source") == "law")
    print(f"  Ruling nodes          : {ruling_nodes:,}")
    print(f"  Law-article nodes     : {law_nodes:,}")

    return G


def save_graph(G: nx.DiGraph, path: str) -> None:
    """Pickle the graph to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"\nGraph saved to: {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Building citation graph from swiss_doc2doc_ir …\n")
    G = build_graph()
    save_graph(G, OUTPUT_PATH)
    print("\nDone.")
