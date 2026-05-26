"""Ad-hoc retrieval on a user-supplied query.

Demo route for a single custom Sachverhalt: the user types (or picks an
example) text, we embed it via TEI, seed via Qdrant, expand four ways
(graph/embedding × 1-hop/2-hop), rank with the chosen strategy and
render the five top-k lists side by side. Nothing is persisted — this
is purely for a live demo of the five retrieval systems.

Implementation note: the pipeline functions live in
`scripts/eval/02_run_retrieval.py` (a filename that starts with a digit,
so it isn't importable as a package member). We load it via importlib
the same way `webui/executor.py` does and call the exported helpers
directly. We do NOT reuse `run_retrieval()` — that function is the
batch driver and writes JSONL files. The per-query work loop inside
it is what we replicate, minimised to one query and one ranking.
"""

from __future__ import annotations

import importlib.util
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient

from .. import inspector_data
from ..app_state import templates
from .inspector import _source_link


router = APIRouter()


# Five example Sachverhalte spanning distinct legal areas. Picked from
# eval_queries.jsonl so they look authentic; trimmed for brevity. These
# are demo prompts, the user can edit them freely.
EXAMPLE_QUERIES: list[dict] = [
    {
        "id": "strafrecht",
        "label": "Strafrecht — Entschädigung nach Verfahrenseinstellung",
        "lang": "de",
        "text": (
            "Sachverhalt: A. Auf Anzeige vom 14. September 2016 hin eröffnete "
            "die Staatsanwaltschaft Nidwalden ein Strafverfahren gegen A._ "
            "wegen Steuerbetrugs, stellte dieses aber am 30. Oktober 2019 unter "
            "Kostenauflage an den Beanzeigten und ohne Ausrichtung einer "
            "Entschädigung ein. Dessen dagegen erhobene Beschwerde wies das "
            "Obergericht des Kantons Nidwalden am 5. März 2020 ab. "
            "B. Mit Beschwerde in Strafsachen beantragt A._, ihm seien für das "
            "Strafverfahren Fr. 16'893.10 Entschädigung und Fr. 2'000.-- "
            "Genugtuung zuzusprechen. Eventualiter sei die Sache an das "
            "Obergericht zurückzuweisen."
        ),
    },
    {
        "id": "verkehr",
        "label": "Strassenverkehr — Geschwindigkeitsüberschreitung",
        "lang": "de",
        "text": (
            "Sachverhalt: A. Das Bezirksamt Muri verurteilte X._ mit Strafbefehl "
            "vom 3. November 2009 wegen einfacher Verletzung von Verkehrsregeln "
            "begangen durch Überschreiten der Höchstgeschwindigkeit zu einer "
            "Busse von Fr. 700.--. Auf Einsprache von X._ hin bestätigte das "
            "Gerichtspräsidium Muri am 16. März 2010 diesen Strafbefehl. "
            "B. Das Obergericht des Kantons Aargau wies die von X._ erhobene "
            "Berufung am 23. Dezember 2010 ab. C. X._ führt Beschwerde in "
            "Strafsachen. Er beantragt, die Urteile seien aufzuheben. Er sei "
            "von Schuld und Strafe freizusprechen."
        ),
    },
    {
        "id": "steuer",
        "label": "Steuerrecht — Steuererlass und Begründungspflicht",
        "lang": "de",
        "text": (
            "Sachverhalt: A. X._, welche für das Jahr 2006 noch direkte "
            "Bundessteuern in der Höhe von 119.35 Franken schuldete, ersuchte "
            "die Steuerverwaltung des Kantons Bern erfolglos um Gewährung "
            "eines Steuererlasses (Verfügung vom 10. Dezember 2007). "
            "B. Am 8. Januar 2008 hat X._ gegen den abschlägigen "
            "Erlassentscheid subsidiäre Verfassungsbeschwerde beim "
            "Bundesgericht eingereicht. Sie beanstandet vorab, dass ihr "
            "Erlassgesuch anders als in den Vorjahren abgewiesen worden sei, "
            "obschon sich ihre finanziellen Verhältnisse nicht verändert "
            "hätten. Weiter rügt sie, dass der Entscheid jeder Begründung "
            "entbehrt."
        ),
    },
    {
        "id": "iv",
        "label": "Sozialversicherung — Aufhebung einer IV-Rente",
        "lang": "de",
        "text": (
            "Sachverhalt: Mit Verfügung vom 27. September 2012 hob die "
            "IV-Stelle des Kantons Zürich die S._ seit Februar 2009 "
            "ausgerichtete halbe Invalidenrente auf Ende Oktober 2012 hin "
            "wiedererwägungsweise auf. Das Sozialversicherungsgericht des "
            "Kantons Zürich wies die dagegen erhobene Beschwerde mit "
            "Entscheid vom 11. Februar 2014 ab. S._ führt Beschwerde ans "
            "Bundesgericht mit dem Antrag, es sei ihr über Ende 2012 hinaus "
            "weiterhin eine halbe, eventuell eine Viertelsrente zuzusprechen."
        ),
    },
    {
        "id": "familie",
        "label": "Familienrecht — Unentgeltliche Rechtspflege in Scheidung",
        "lang": "de",
        "text": (
            "Sachverhalt: Zwischen A._ und B._ ist vor dem Kantonsgericht "
            "Schwyz im Berufungsstadium das Scheidungsverfahren hängig. "
            "Mit Entscheid vom 23. April 2019 wies das Kantonsgericht das "
            "Gesuch des Ehemannes um unentgeltliche Rechtspflege ab und "
            "verlangte von ihm (bei einem Fr. 7 Mio. übersteigenden "
            "Streitwert) einen Kostenvorschuss von Fr. 35'000.--. Dagegen "
            "hat der Ehemann am 24. Mai 2019 beim Bundesgericht eine "
            "Beschwerde eingereicht mit den Begehren um Aufhebung und "
            "Neubeurteilung durch das Kantonsgericht, eventuell um "
            "Erteilung der unentgeltlichen Rechtspflege."
        ),
    },
]


SYSTEMS = [
    ("rag", "RAG"),
    ("emb_1hop", "Emb 1-hop"),
    ("emb_2hop", "Emb 2-hop"),
    ("graph_1hop", "Graph 1-hop"),
    ("graph_2hop", "Graph 2-hop"),
]
RANKINGS = ["cosine", "indegree", "cross_encoder"]
K_VALUES = [5, 10, 15, 20]


@lru_cache(maxsize=1)
def _pipeline_module():
    """Load scripts/eval/02_run_retrieval.py as a module.

    Same trick as `webui/executor.py`. Done once and cached for the
    process lifetime.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "eval" / "02_run_retrieval.py"
    if not script_path.exists():
        # Container layout: scripts/eval is at /app/scripts/eval.
        script_path = Path("/app/scripts/eval/02_run_retrieval.py")
    spec = importlib.util.spec_from_file_location("retrieval_mod", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refresh_pipeline_env(mod) -> None:
    """Re-read TEI/Qdrant env vars into module-level constants.

    The pipeline module captures these at import time; /settings changes
    update os.environ but not the already-loaded module. We re-bind them
    on every custom query so a host edit in /settings takes effect on
    the next click without needing a pod restart.
    """
    mod.TEI_HOST = os.environ.get("TEI_HOST", mod.TEI_HOST)
    mod.TEI_PORTS = [int(p) for p in os.environ.get(
        "TEI_PORTS", ",".join(str(p) for p in mod.TEI_PORTS)
    ).split(",") if p.strip()]
    rerank_urls = os.environ.get("TEI_RERANK_URLS", "")
    if rerank_urls:
        mod.TEI_RERANK_URLS = [u.strip() for u in rerank_urls.split(",") if u.strip()]


def _qdrant_client() -> QdrantClient:
    # Reuse the inspector_data client so we share the connection (and
    # benefit from /settings cache_clear on host changes).
    return inspector_data.qdrant()


def _detect_language(text: str) -> str:
    """Crude trigram-free language guess for de/fr/it.

    The pipeline only filters embedding sources by `swiss_rulings_chunked`
    and `swiss_leading_decisions_chunked`, both multilingual, so the
    language label is only used to drive the bger.ch link locale and to
    hint at example-fit. A heavy langid dependency would be overkill for
    a demo route.
    """
    t = " " + text.lower() + " "
    score = {
        "de": sum(t.count(w) for w in (" der ", " die ", " und ", " mit ", " ist ", " vom ")),
        "fr": sum(t.count(w) for w in (" le ", " la ", " et ", " avec ", " est ", " du ")),
        "it": sum(t.count(w) for w in (" il ", " la ", " e ", " con ", " del ", " della ")),
    }
    return max(score, key=score.get) if any(score.values()) else "de"


@router.get("/custom-query", response_class=HTMLResponse)
def custom_query_form(request: Request):
    return templates.TemplateResponse(
        request,
        "custom_query_form.html",
        {
            "examples": EXAMPLE_QUERIES,
            "rankings": RANKINGS,
            "k_values": K_VALUES,
            "default_ranking": "indegree",
            "default_k": 20,
        },
    )


@router.post("/custom-query", response_class=HTMLResponse)
def custom_query_run(
    request: Request,
    query_text: Annotated[str, Form()],
    language: Annotated[str, Form()] = "auto",
    ranking: Annotated[str, Form()] = "indegree",
    k: Annotated[int, Form()] = 20,
):
    query_text = (query_text or "").strip()
    if not query_text:
        return templates.TemplateResponse(
            request,
            "custom_query_form.html",
            {
                "examples": EXAMPLE_QUERIES,
                "rankings": RANKINGS,
                "k_values": K_VALUES,
                "default_ranking": ranking,
                "default_k": k,
                "error": "Bitte einen Sachverhalt eingeben.",
            },
            status_code=400,
        )
    if ranking not in RANKINGS:
        ranking = "indegree"
    if k not in K_VALUES:
        k = 20
    if language not in ("auto", "de", "fr", "it"):
        language = "auto"
    detected = _detect_language(query_text) if language == "auto" else language

    mod = _pipeline_module()
    _refresh_pipeline_env(mod)
    client = _qdrant_client()
    graph = inspector_data.graph()

    timings: dict[str, float] = {}
    error: Optional[str] = None
    by_system: dict[str, list[dict]] = {}
    seed_docs: list = []

    try:
        # Step 1: TEI embed. Probe configured endpoints lazily — the user
        # might be running locally against an unreachable cluster.
        t0 = time.monotonic()
        endpoint = mod.find_live_tei_endpoint()
        if endpoint is None:
            raise RuntimeError(
                "TEI embedding endpoint not reachable. Check /settings → "
                "aiserver host + TEI embed port."
            )
        vec = mod.embed_texts([query_text], endpoint)[0]
        timings["embed"] = time.monotonic() - t0

        # Step 2: Qdrant ANN seeds. No date filter — custom queries don't
        # come with a publication date, the demo wants the full corpus.
        t0 = time.monotonic()
        ruling_sources = [mod.RULINGS_SOURCE, mod.LEADING_SOURCE]
        seed_docs = mod.qdrant_search(
            client, vec, ruling_sources, limit=mod.SEED_K,
            exclude_decision_id=None,
            query_date_ms=0,
        )
        timings["seeds"] = time.monotonic() - t0
        seed_ids = [d["decision_id"] for d in seed_docs]
        seed_score_map = {d["decision_id"]: d["score"] for d in seed_docs}

        # Step 3: Four expansions (graph 1/2-hop, emb 1/2-hop) in parallel.
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_g1 = ex.submit(mod.expand_graph_1hop, seed_ids, graph,
                             mod.MAX_1HOP_CANDIDATES)
            f_g2 = ex.submit(mod.expand_graph_2hop, seed_ids, graph,
                             mod.MAX_2HOP_CANDIDATES)
            f_e1 = ex.submit(mod.expand_emb_1hop, seed_docs, client,
                             mod.MAX_1HOP_CANDIDATES)
            graph_1hop_ids = f_g1.result()
            graph_2hop_ids = f_g2.result()
            emb_1hop_ids = f_e1.result()
        # emb_2hop reuses the 1-hop result, so it runs sequentially.
        emb_2hop_ids = mod.expand_emb_2hop(
            seed_docs, emb_1hop_ids, client, mod.MAX_2HOP_CANDIDATES,
        )
        timings["expand"] = time.monotonic() - t0

        def make_candidates(ids_to_add) -> list:
            merged: dict = dict(seed_score_map)
            for did in sorted(ids_to_add):
                if did not in merged:
                    merged[did] = 0.0
            return [{"decision_id": d, "score": s} for d, s in merged.items()]

        candidates_by_system = {
            "rag":        [{"decision_id": d["decision_id"], "score": d["score"]}
                           for d in seed_docs],
            "emb_1hop":   make_candidates(emb_1hop_ids),
            "emb_2hop":   make_candidates(emb_2hop_ids),
            "graph_1hop": make_candidates(graph_1hop_ids),
            "graph_2hop": make_candidates(graph_2hop_ids),
        }

        # Step 4: rank — three strategies, but we only run the selected one.
        ce_score_map: Optional[dict] = None
        if ranking == "cross_encoder":
            t0 = time.monotonic()
            ce_urls = mod.load_cross_encoder()
            if ce_urls is None:
                error = ("Cross-Encoder endpoint nicht erreichbar — "
                         "Ranking auf In-Degree zurückgefallen.")
                ranking = "indegree"
            else:
                # Union of all candidate ids → one rerank call.
                union_score: dict = {}
                for cands in candidates_by_system.values():
                    for c in cands:
                        union_score.setdefault(c["decision_id"], c["score"])
                text_cache: dict = {}
                mod.fetch_chunk_texts(list(union_score.keys()),
                                      client, cache=text_cache)
                union_with_text = sorted(
                    (
                        {"decision_id": d, "score": s, "text": text_cache.get(d, "")}
                        for d, s in union_score.items()
                        if text_cache.get(d, "").strip()
                    ),
                    key=lambda r: r["decision_id"],
                )
                ranked_union = mod.rank_by_cross_encoder(
                    query_text, union_with_text, ce_urls
                )
                ce_score_map = {r["decision_id"]: r["score"] for r in ranked_union}
            timings["rerank"] = time.monotonic() - t0

        t0 = time.monotonic()
        for system, _label in SYSTEMS:
            cands = candidates_by_system[system]
            if ranking == "cosine":
                ranked = mod.rank_by_cosine(cands)
            elif ranking == "indegree":
                ranked = mod.rank_by_indegree(cands, graph)
            else:
                ranked = sorted(
                    [{"decision_id": c["decision_id"],
                      "score": (ce_score_map or {}).get(c["decision_id"], -1e9)}
                     for c in cands],
                    key=lambda x: x["score"], reverse=True,
                )
            by_system[system] = ranked[:k]
        timings["rank"] = time.monotonic() - t0

        # Step 5: enrich top-k with metadata for display.
        t0 = time.monotonic()
        all_ids: set[str] = set()
        for results in by_system.values():
            all_ids.update(r["decision_id"] for r in results)
        meta_map = inspector_data.metadata_for(all_ids)
        timings["meta"] = time.monotonic() - t0

        results_by_system = []
        for system, label in SYSTEMS:
            rows = []
            for rank_idx, r in enumerate(by_system[system], start=1):
                did = r["decision_id"]
                m = meta_map.get(did) or {}
                rows.append({
                    "rank": rank_idx,
                    "decision_id": did,
                    "score": r["score"],
                    "file_number": m.get("file_number"),
                    "date_ms": m.get("date_ms"),
                    "language": m.get("language"),
                    "court": m.get("court"),
                    "source": m.get("source"),
                    "text": (m.get("text") or "")[:200],
                    "indegree": graph.in_degree(did) if did in graph else 0,
                    "source_link": _source_link(
                        m.get("file_number"), m.get("date_ms"),
                        m.get("court"), m.get("source"),
                        m.get("language") or detected,
                    ),
                })
            results_by_system.append({"key": system, "label": label, "rows": rows})

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    timings["total"] = sum(timings.values())

    return templates.TemplateResponse(
        request,
        "custom_query_results.html",
        {
            "query_text": query_text,
            "language": language,
            "detected_language": detected,
            "ranking": ranking,
            "k": k,
            "rankings": RANKINGS,
            "k_values": K_VALUES,
            "examples": EXAMPLE_QUERIES,
            "n_seeds": len(seed_docs),
            "results_by_system": locals().get("results_by_system", []),
            "timings": timings,
            "error": error,
        },
    )
