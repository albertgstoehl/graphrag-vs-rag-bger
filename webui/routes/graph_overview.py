"""Knowledge-graph overview page.

Reads the citation_graph.pkl via the inspector's existing lazy loader
(`webui.inspector_data.graph()`) so this page does not re-load the pickle
into a second process-wide copy. Computes the aggregate stats once on
first request and caches the result for the pod's lifetime — the graph
is read-only at runtime.
"""

from __future__ import annotations

from collections import Counter
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import inspector_data
from ..app_state import templates
from .inspector import _hf_dataset_url, _source_link


router = APIRouter()

_STATS_CACHE: dict | None = None
_STATS_LOCK = Lock()


def _compute_stats() -> dict:
    """Aggregate counters shown in the four header cards on /graph.

    Walks the graph once, so it stays O(n_nodes + n_edges) and runs in
    well under a second on the 158k-node citation graph.
    """
    g = inspector_data.graph()
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    in_mean = (n_edges / n_nodes) if n_nodes else 0.0
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "directed": g.is_directed(),
        "in_mean": in_mean,
        "out_mean": in_mean,
    }


def stats() -> dict:
    global _STATS_CACHE
    if _STATS_CACHE is None:
        with _STATS_LOCK:
            if _STATS_CACHE is None:
                _STATS_CACHE = _compute_stats()
    return _STATS_CACHE


@router.get("/graph", response_class=HTMLResponse)
def graph_overview(request: Request):
    return templates.TemplateResponse(
        request, "graph_overview.html", {"stats": stats()},
    )


@router.get("/api/graph-overview")
def api_graph_overview():
    return JSONResponse(stats())


def _enrich_node(g, nid: str, meta: dict) -> dict:
    """Cytoscape node descriptor, enriched with Qdrant metadata so the
    tooltip can show the same fields as the per-query inspector."""
    graph_attrs = g.nodes[nid] if nid in g else {}
    m = meta or {}
    return {
        "data": {
            "id": nid,
            "label": m.get("file_number") or nid[:8],
            "source": graph_attrs.get("source") or "unknown",
            "indegree":  int(g.in_degree(nid)) if nid in g else 0,
            "outdegree": int(g.out_degree(nid)) if nid in g else 0,
            "file_number": m.get("file_number"),
            "date": m.get("date_ms"),
            "language": m.get("language"),
            "court": m.get("court"),
            "qdrant_source": m.get("source"),
            "text_excerpt": (m.get("text") or "")[:300],
            "source_link": _source_link(
                m.get("file_number"), m.get("date_ms"),
                m.get("court"), m.get("source"),
                m.get("language") or "de",
            ),
            "hf_url": _hf_dataset_url("rcds/swiss_rulings", nid),
        }
    }


def _edge_payload(src: str, dst: str, etype: str) -> dict:
    return {
        "data": {
            "id": f"{src}->{dst}",
            "source": src, "target": dst, "type": etype or "",
        }
    }


@router.get("/api/graph-overview/subgraph")
def api_subgraph(seed: str | None = None,
                 top_n: int = 30,
                 citers_per_seed: int = 5,
                 max_nodes: int = 250):
    """Cytoscape-compatible subgraph view.

    Two modes:
    - `seed` absent: take the top-`top_n` most-cited rulings and add up to
      `citers_per_seed` predecessors for each. Caps at `max_nodes` total to
      keep the layout responsive.
    - `seed` set: that node + its in-neighbors (citers) and out-neighbors
      (citees), capped at `max_nodes`.
    """
    g = inspector_data.graph()
    node_ids: list[str] = []
    seen: set[str] = set()
    edges: list[dict] = []

    def _add_node(nid: str):
        if nid in seen:
            return
        seen.add(nid)
        node_ids.append(nid)

    def _add_edge(src: str, dst: str):
        if src not in g or dst not in g:
            return
        if not g.has_edge(src, dst):
            return
        etype = g.edges[src, dst].get("type") or ""
        edges.append(_edge_payload(src, dst, etype))

    if seed:
        if seed not in g:
            raise HTTPException(404, f"node {seed!r} not in graph")
        _add_node(seed)
        # In-neighbours = citers of seed (predecessors)
        for p in g.predecessors(seed):
            if len(seen) >= max_nodes:
                break
            _add_node(p)
            _add_edge(p, seed)
        # Out-neighbours = what seed cites (successors)
        for s in g.successors(seed):
            if len(seen) >= max_nodes:
                break
            _add_node(s)
            _add_edge(seed, s)
    else:
        top_n = max(1, min(top_n, 100))
        citers_per_seed = max(0, min(citers_per_seed, 20))
        seeds = sorted(g.in_degree(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for sid, _deg in seeds:
            if len(seen) >= max_nodes:
                break
            _add_node(sid)
        for sid, _deg in seeds:
            if len(seen) >= max_nodes:
                break
            preds = list(g.predecessors(sid))
            preds.sort(key=lambda n: g.in_degree(n) + g.out_degree(n), reverse=True)
            for p in preds[:citers_per_seed]:
                if len(seen) >= max_nodes:
                    break
                _add_node(p)
                _add_edge(p, sid)
        # After all nodes are placed, add any seed→seed citation edges
        seed_ids = {sid for sid, _ in seeds}
        for sid in seed_ids:
            for s in g.successors(sid):
                if s in seed_ids:
                    _add_edge(sid, s)

    # Batch-fetch Qdrant metadata once for all selected nodes (cache hits
    # after the boot-time preload of bger chunk-0 metadata).
    meta = inspector_data.metadata_for(node_ids)
    nodes = [_enrich_node(g, nid, meta.get(nid, {})) for nid in node_ids]

    return JSONResponse({
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "seed": seed,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "top_n": top_n,
            "citers_per_seed": citers_per_seed,
            "truncated": len(nodes) >= max_nodes,
        },
    })
