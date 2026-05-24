"""Per-query inspector: 2-column detail page with side-by-side
citation-graph visualisations for any two systems.

The per-query *table* used to live at /inspector?tab=metrics |?tab=pool
but has been folded into the aggregate metric subpages — clicking a
matrix cell, a /metrics row, or a graph-nearness bar opens an inline
per-query drilldown panel right there. This module now owns only the
single-query detail view and its JSON graph endpoint.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import inspector_data, metrics_reader
from ..app_state import templates


router = APIRouter()


SYSTEMS = ["rag", "rag_smart", "emb_1hop", "emb_2hop", "graph_1hop", "graph_2hop"]
RANKINGS = ["cosine", "cross_encoder", "indegree"]
K_VALUES = [5, 10, 15, 20]


@router.get("/inspector")
def inspector_table_redirect(tab: str = "metrics",
                             ranking: str = "cross_encoder",
                             k: int = 10):
    """The per-query table moved into the aggregate subpages. Bounce old
    bookmarks to the matching drilldown:

      /inspector?tab=metrics → /metrics?row=graph_1hop|<ranking>_<k>
      /inspector?tab=pool    → /metrics/recall-ceiling?cell=graph_1hop|post_cap

    Default-anchor system is graph_1hop because it's the citation-graph
    arm under test in the thesis — the comparison most users land here
    for. The thesis figures' inspector links re-resolve to a populated
    panel rather than a dead URL.
    """
    if ranking not in RANKINGS or k not in K_VALUES:
        ranking, k = "cross_encoder", 10
    if tab == "pool":
        return RedirectResponse(
            url="/metrics/recall-ceiling?cell=graph_1hop%7Cpost_cap",
            status_code=302,
        )
    return RedirectResponse(
        url=f"/metrics?row=graph_1hop%7C{ranking}_{k}",
        status_code=302,
    )


@router.get("/inspector/{query_id}", response_class=HTMLResponse)
def inspector_detail(request: Request, query_id: str,
                     left: str = "graph_1hop", right: str = "emb_1hop",
                     view: str = "ranked"):
    """Per-query detail view: 2-column layout with side-by-side graphs.

    `left` and `right` pick which system's graph each panel renders.
    The graph data itself is fetched async via /inspector/{query_id}/graph
    so the page renders immediately and graphs hydrate after.
    """
    q = metrics_reader.query_by_id(query_id)
    if not q:
        raise HTTPException(404, f"Query {query_id} not found")

    # The Doc2Doc-IR dataset records each citation occurrence in the source
    # decision separately, so `ground_truth_cases` may list the same precedent
    # multiple times (a court can cite the same BGE 12× in one judgment). All
    # downstream metrics already work on `set(...)` semantics; the UI must
    # match that — otherwise the "X / Y indexed" line shows 24 / 68 instead
    # of the correct 24 / 24.
    gt_ids = list(dict.fromkeys(q.get("ground_truth_cases", [])))  # dedup, keep order

    # Parallelise the two slow data sources. metadata_for hits Qdrant
    # (cold ~75 ms, warm 0 ms via cache); retrieved_for opens 45 jsonl
    # files lazily (~130 ms). They are independent of each other.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_meta = pool.submit(inspector_data.metadata_for, gt_ids + [query_id])
        f_retrieved = pool.submit(metrics_reader.retrieved_for, query_id)
        gt_meta = f_meta.result()
        retrieved = f_retrieved.result()
    query_meta = gt_meta.pop(query_id, None)

    # An "indexed" GT is exactly one whose chunk-0 metadata came back —
    # metadata_for filters to chunk_index=0 which always exists for every
    # indexed decision. The old gt_indexed_status() did one Qdrant
    # count() round-trip per id (~30 ms each, ~234 ms for a typical
    # query) and gave us the same answer.
    gt_indexed = {gid: gid in gt_meta for gid in gt_ids}
    # Per-system rank lookup for each GT id, picked at the chosen ranking/k.
    # We pre-compute the most useful one (cross_encoder, k=20) for the side
    # panel; other (ranking,k) views are reachable by reloading.
    gt_rank_table = []
    for gid in gt_ids:
        m = gt_meta.get(gid) or {}
        row = {"id": gid, "indexed": gt_indexed.get(gid, False),
               "meta": gt_meta.get(gid),
               "source_link": _source_link(
                   m.get("file_number"), m.get("date_ms"),
                   m.get("court"), m.get("source"),
                   m.get("language") or "de",
               ) if gt_meta.get(gid) else None}
        for s in SYSTEMS:
            ranks_per_cfg = {}
            for r in RANKINGS:
                for kv in K_VALUES:
                    key = f"{s}_{r}_{kv}"
                    if key in retrieved:
                        ids_list = retrieved[key]
                        if gid in ids_list:
                            ranks_per_cfg[f"{r}_{kv}"] = ids_list.index(gid) + 1
            row[s] = ranks_per_cfg
        gt_rank_table.append(row)

    query_source_link = None
    if query_meta:
        query_source_link = _source_link(
            query_meta.get("file_number"), query_meta.get("date_ms"),
            query_meta.get("court"), query_meta.get("source"),
            query_meta.get("language") or "de",
        )

    return templates.TemplateResponse(
        request, "inspector_detail.html",
        {"q": q, "query_meta": query_meta, "gt_rank_table": gt_rank_table,
         "query_source_link": query_source_link,
         "n_indexed": sum(1 for v in gt_indexed.values() if v),
         "n_total_gt": len(gt_ids),
         "left": left, "right": right, "view": view,
         "systems": SYSTEMS,
         "hf_url": _hf_dataset_url},
    )


def _hf_dataset_url(dataset: str, decision_id: str, splits=("train", "validation", "test")) -> str:
    """Build a HuggingFace dataset-viewer URL that filters to a single
    decision_id. Uses the SQL console (`?sql_console=true&sql=...`) because
    `?search=` and `?q=` are silently ignored on Data-Studio-enabled
    datasets like `rcds/swiss_doc2doc_ir`. UNIONs across all splits
    so the link works regardless of which split holds the row.

    First click on a UNION-ALL link can take ~10-15 s while DuckDB-WASM
    scans the parquet files; subsequent clicks reuse the browser cache.
    """
    from urllib.parse import quote
    parts = [
        f"SELECT '{s}' AS split, * FROM {s} WHERE decision_id = '{decision_id}'"
        for s in splits
    ]
    sql = " UNION ALL ".join(parts)
    return (
        f"https://huggingface.co/datasets/{dataset}/viewer/default/{splits[0]}"
        f"?sql_console=true&sql={quote(sql, safe='')}"
    )


def _source_link(file_number: Optional[str], date_ms: Optional[int],
                 court: Optional[str], source: Optional[str],
                 lang: str = "de") -> Optional[dict]:
    """Best public URL for a node, with the hostname as a label.

    Returns ``{"url": ..., "host": ...}`` or None.

    Coverage:
      • CH_BGer (case-numbered)            → bger.ch aza:// deep link
      • CH_BGE  (BGE collection)           → bger.ch atf:// deep link
      • Swiss federal laws (court='ch',
        source='swiss_legislation_chunked') → fedlex.admin.ch search
      • everything else with a file_number → entscheidsuche.ch search
    """
    import re
    from urllib.parse import quote
    if lang not in ("de", "fr", "it"):
        lang = "de"

    # CH_BGer: case-numbered ruling, deep link via aza://
    # The Bundesgericht site canonicalises file_numbers as
    #   "{chamber_num}-{4-digit-year}"
    # where chamber_num replaces every `.` with `-` (so `6P.149` → `6P-149`)
    # but keeps `_` (so `4D_71`, `I_799`, `B_8` stay). 2-digit case years
    # ("/03", "/01") expand to 4 digits anchored to the decision year, e.g.
    # decision 2004 + filing year `03` → 2003. Confirmed across all three
    # observed CH_BGer file_number families:
    #   • modern post-2007       4D_71/2011    → aza://28-03-2012-4D_71-2011
    #   • old-style dot          6P.149/2005   → aza://28-04-2006-6P-149-2005
    #   • EVG single-letter      I_799/03      → aza://26-04-2004-I_799-2003
    if court == "CH_BGer" and file_number and date_ms:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(int(date_ms) / 1000, tz=timezone.utc)
            slash = file_number.rfind("/")
            if slash > 0:
                chamber, year_part = file_number[:slash], file_number[slash+1:]
                if year_part.isdigit():
                    if len(year_part) == 2:
                        # Anchor 2-digit year to the decision year's century.
                        century = (dt.year // 100) * 100
                        candidate = century + int(year_part)
                        if candidate > dt.year:
                            candidate -= 100
                        year_part = f"{candidate:04d}"
                    chamber_norm = chamber.replace(".", "-")
                    docid = (f"aza://{dt.day:02d}-{dt.month:02d}-{dt.year}"
                             f"-{chamber_norm}-{year_part}")
                    return {
                        "url": (f"https://www.bger.ch/ext/eurospider/live/{lang}"
                                f"/php/aza/http/index.php?lang={lang}"
                                f"&type=show_document"
                                f"&highlight_docid={quote(docid, safe='')}"),
                        "host": "bger.ch",
                    }
        except Exception:
            pass

    # CH_BGE: leading-decision collection, deep link via atf://
    # File numbers come in three observed flavours:
    #   "BGE 121 III 408"   (~19%)   — Doc2Doc-IR style with spaces
    #   "BGE_134_I_83"      (~81%)   — swiss_leading_decisions normalisation
    #   "BGE_103_Ia_191"    (older)  — section letter suffix on the roman num
    if court == "CH_BGE" and file_number:
        m = re.match(r"^BGE[\s_]+(\d+)[\s_]+([IVX]+[ab]?)[\s_]+(\d+)$",
                     file_number.strip())
        if m:
            volume, roman, page = m.groups()
            docid = f"atf://{volume}-{roman}-{page}:{lang}"
            return {
                "url": (f"https://www.bger.ch/ext/eurospider/live/{lang}"
                        f"/php/clir/http/index.php?lang={lang}"
                        f"&type=show_document&highlight_docid={quote(docid, safe='')}"),
                "host": "bger.ch",
            }

    # Swiss federal laws → fedlex search by SR number (file_number is the
    # SR classification, e.g. "173.110" = Bundesgerichtsgesetz).
    if source == "swiss_legislation_chunked" and file_number:
        return {
            "url": f"https://www.fedlex.admin.ch/{lang}/search?text={quote(file_number)}",
            "host": "fedlex.admin.ch",
        }

    # Fallback: entscheidsuche.ch indexes BGE + cantonal courts and works
    # for anything with a usable file_number.
    if file_number:
        return {
            "url": f"https://entscheidsuche.ch/?text={quote(file_number)}",
            "host": "entscheidsuche.ch",
        }

    return None


POOL_NODE_CAP = 300  # don't visualise more than this — cytoscape gets unhappy


@router.get("/inspector/{query_id}/graph")
def inspector_graph(query_id: str, system: str = "graph_1hop",
                    ranking: str = "cross_encoder", k: int = 20,
                    view: str = "ranked"):
    """JSON endpoint feeding cytoscape.js.

    `view='ranked'`: query + the system's top-k retrieved at the chosen
    ranking + any GT not already covered. Answers 'what does the
    pipeline actually output for this query'.

    `view='pool'`: query + the full candidate pool of the system + any
    GT not already covered. Answers 'is the GT even reachable through
    this system's candidate generation' — i.e. visualises Pool-Recall.

    Both views add citation edges from `citation_graph.pkl` between any
    pair of selected nodes. Pool view caps total nodes at POOL_NODE_CAP
    by sub-sampling non-GT pool candidates if necessary.
    """
    if system not in SYSTEMS:
        raise HTTPException(400, f"unknown system {system}")
    if view not in ("ranked", "pool"):
        raise HTTPException(400, f"unknown view {view}")

    # Run all five JSONL-scan lookups concurrently — each is a linear
    # file scan, ~30–130 ms each, totally independent of each other.
    # Sequential they sum to ~500 ms; parallel they finish in the
    # slowest one's time (~150 ms typical).
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_q = pool.submit(metrics_reader.query_by_id, query_id)
        f_retr = pool.submit(metrics_reader.retrieved_for, query_id)
        f_pool = pool.submit(metrics_reader.pool_for, query_id)
        f_seeds = pool.submit(metrics_reader.seeds_for, query_id)
        f_ce = pool.submit(metrics_reader.cross_encoder_scores_for, query_id)
        q = f_q.result()
        retrieved = f_retr.result()
        pool_for_query = f_pool.result()
        seed_ids_for_q = f_seeds.result()
        ce_scores = f_ce.result()

    if not q:
        raise HTTPException(404, f"query {query_id} not found")
    gt = set(q.get("ground_truth_cases", []))  # set drops the Doc2Doc duplicates

    cfg_key = f"{system}_{ranking}_{k}"
    retrieved_ids = retrieved.get(cfg_key, [])

    # Pool for THIS system (used for cosine scores in either view, and
    # as the node set when view='pool').
    pool_records = pool_for_query.get(system, [])
    cosine_map: dict[str, float] = {}
    pool_ids: set[str] = set()
    for c in pool_records:
        if isinstance(c, dict):
            did = c.get("decision_id")
            if did:
                pool_ids.add(did)
                cosine_map[did] = c.get("score", 0.0)
        elif isinstance(c, str):
            pool_ids.add(c)

    if view == "ranked":
        node_ids = {query_id, *retrieved_ids, *gt}
        config = {"view": view, "system": system, "ranking": ranking,
                  "k": k, "n_retrieved": len(retrieved_ids),
                  "n_pool": len(pool_ids), "n_gt": len(gt),
                  "n_seeds": len(seed_ids_for_q),
                  "n_gt_via_seed": len(gt & seed_ids_for_q),
                  "n_gt_via_expansion_only": len((gt & pool_ids) - seed_ids_for_q)}
    else:
        # view == 'pool': always include query + all GT + pool ∩ GT, then
        # add as many non-GT pool members as fit under the cap.
        gt_in_pool = pool_ids & gt
        gt_missing = gt - pool_ids
        pool_other = list(pool_ids - gt)
        capacity = POOL_NODE_CAP - 1 - len(gt_in_pool) - len(gt_missing)
        sampled_other = pool_other[:max(0, capacity)]
        node_ids = {query_id, *gt, *gt_in_pool, *sampled_other}
        config = {"view": view, "system": system,
                  "n_pool": len(pool_ids), "n_gt": len(gt),
                  "n_seeds": len(seed_ids_for_q),
                  "n_pool_hits": len(gt_in_pool),
                  "n_gt_via_seed": len(gt & seed_ids_for_q),
                  "n_gt_via_expansion_only": len(gt_in_pool - seed_ids_for_q),
                  "n_pool_displayed": len(sampled_other),
                  "pool_truncated": len(pool_other) > len(sampled_other)}

    meta = inspector_data.metadata_for(node_ids)
    edges = inspector_data.edges_within(node_ids)
    # ce_scores already fetched above in the parallel block

    nodes = []
    for did in node_ids:
        m = meta.get(did, {})
        is_query = did == query_id
        is_gt = did in gt
        is_in_pool = did in pool_ids
        is_retrieved = did in retrieved_ids
        is_seed = did in seed_ids_for_q
        rank = retrieved_ids.index(did) + 1 if is_retrieved else None

        if is_query:
            kind = "query"
        elif view == "ranked":
            if is_retrieved and is_gt:
                kind = "hit"
            elif is_retrieved and not is_gt:
                kind = "miss"
            elif is_gt and not is_retrieved:
                kind = "missing_gt"
            else:
                kind = "other"
        else:  # view == 'pool'
            if is_in_pool and is_gt:
                kind = "hit"  # green: GT in the pool
            elif is_gt and not is_in_pool:
                kind = "missing_gt"  # red: GT not even reachable
            else:
                kind = "pool_other"  # small grey: pool noise

        nodes.append({
            "data": {
                "id": did,
                "label": m.get("file_number") or did[:8],
                "kind": kind,
                "is_seed": is_seed,
                "rank": rank,
                "in_pool": is_in_pool,
                "indegree": inspector_data.in_degree(did),
                "language": m.get("language"),
                "date": m.get("date_ms"),
                "file_number": m.get("file_number"),
                "text_excerpt": m.get("text", "")[:300],
                "cosine": cosine_map.get(did),
                "cross_encoder": ce_scores.get(did),
                "hf_url": _hf_dataset_url("rcds/swiss_rulings", did),
                "source_link": _source_link(
                    m.get("file_number"), m.get("date_ms"),
                    m.get("court"), m.get("source"),
                    m.get("language") or "de",
                ),
            }
        })
    edge_data = [
        {"data": {"id": f"{u}->{v}", "source": u, "target": v}}
        for u, v in edges
    ]
    return JSONResponse({
        "elements": {"nodes": nodes, "edges": edge_data},
        "config": config,
    })
