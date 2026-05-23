# SPEC — Public Repo `graphrag-vs-rag-bger`

**Status:** Planungs-Skizze. Repo wird nach Abgabe der Bachelorarbeit auf GitHub `albertgstoehl/graphrag-vs-rag-bger` öffentlich gemacht.

**Zweck:** Reproduzierbarkeits-Artefakt für die Bachelorarbeit "Citation-Graph-Retrieval vs. Embedding-Retrieval auf Schweizer Bundesgerichtsentscheiden". Enthält genau jene Dateien, die in der Arbeit namentlich referenziert werden, plus eine Setup-Anleitung.

## Was rein muss

Alle Dateien sind aus dem privaten Repo `kg-rag-legal` zu übernehmen, ohne ops-spezifische Konfiguration (Hostnames, IP-Adressen, Tailscale-Konfig).

### Pipeline-Code
- `scripts/eval/01_sample_queries.py` — Stage 1, Query-Sampling
- `scripts/eval/02_run_retrieval.py` — Stage 2, Retrieval gegen 5 Systeme
- `scripts/eval/03_compute_metrics.py` — Stage 3, Metrik-Berechnung
- `scripts/eval/build_citation_graph.py` — Citation-Graph-Build aus rcds/swiss_doc2doc_ir
- `scripts/eval/build_valid_ids.py` — V ∩ G Schnittmengen-Filter
- `scripts/eval/build_facts_index.py` — Facts-Index für HF-freies Stage 1
- `scripts/migrate_qdrant_date_ms.py` — Qdrant Payload-Migration für indizierten Datums-Range-Filter

### Bootstrap-Artefakte
- `data/eval/citation_graph.pkl` — NetworkX DiGraph, 159k Knoten, 1.6M Kanten (per Git LFS)
- `data/eval/valid_ids.json` — 131'734 Decision-IDs des V ∩ G Schnitts
- `data/eval/date_index.json` — Decision-ID zu ms-Timestamp Map
- `data/eval/eval_queries.jsonl` — Query-Set mit GT
- `data/eval/facts_index.jsonl` — Facts-Texte pro Decision-ID

### Web-UI
- `webui/` — vollständiges FastAPI-Verzeichnis mit Templates, statischen Assets, Inspector

### Resultate
- `data/eval/results/` — pro System: `*_pool.jsonl`, `*_layers.jsonl`, plus `cross_encoder_scores.jsonl`
- `data/eval/metrics/` — 60 `per_query_<system>_<ranking>_<k>.jsonl` Dateien plus die Aggregat-CSVs

### Thesis
- `thesis/` — vollständige Markdown-Quellen, references.json, build.yaml
- `build/draft/thesis.pdf` — finaler Build (oder Link zu GitHub-Release)

### Figures
- `thesis/figures/` — alle für die Arbeit generierten PNGs
- Plus die Python-Skripte, die sie erzeugen, falls als separater Ordner vorhanden

## Was raus muss

- Hostnames (`aiserver01-1`, `ubuntu-4gb-hel1-1`)
- IP-Adressen (`100.116.242.70`, `192.168.0.100`)
- Tailscale-Konfig, `~/.openfang/`-Artefakte
- k3s/k0s deploy-Skripte aus `~/.openfang/apps/kg-rag-control/`
- SSH-Konfig-Hinweise

Statt konkreter Hosts/Ports nur die generische Architektur ausweisen (Qdrant + zwei TEI-Instanzen + Pipeline-Client), wie auch in der Arbeit selbst dokumentiert.

## Setup-Anleitung (für die README)

Strukturierung sollte so aussehen.

```
1. Voraussetzungen
   - Python 3.11+
   - Docker oder Kubernetes für Modell-Services
   - Min. 32 GB RAM, GPU mit mind. 24 GB VRAM für Reranker

2. Modell-Services starten
   - Qdrant 1.x (Docker)
   - TEI mit BAAI/bge-m3 (Embeddings)
   - TEI mit BAAI/bge-reranker-v2-m3 (Reranker, mehrere Replicas optional)
   - Endpunkt-URLs in `webui/services.py` oder via /settings UI eintragen

3. Korpus indexieren
   - Qdrant-Collection `bger` mit BGE-M3-Embeddings der rcds/swiss_rulings + rcds/swiss_leading_decisions Chunks aufbauen
   - Datums-Payload `date_ms` per `scripts/migrate_qdrant_date_ms.py` setzen

4. Bootstrap-Artefakte laden
   - Aus dem Repo via Git LFS oder neu bauen mit `build_citation_graph.py`, `build_valid_ids.py`, `build_facts_index.py`

5. Pipeline starten
   - Web-UI auf Port 8000 starten und `Run` klicken
   - Oder direkt: `python scripts/eval/01_sample_queries.py && python scripts/eval/02_run_retrieval.py && python scripts/eval/03_compute_metrics.py`

6. Resultate inspizieren
   - Web-UI `/metrics` für Aggregat-Tabellen
   - Web-UI `/inspector/<query_id>` für Per-Query-Details
```

## Lizenz

MIT oder CC-BY-SA für die Inhalte. Vor Veröffentlichung mit Betreuung und ZHAW-IPR klären (Merkblatt Ziff. 14, 15).

## Nicht-Ziel

Das Repo ist explizit kein produktiver Service. Es ist die Reproduzierbarkeits-Beilage zur Arbeit. Wer mit der Pipeline weitermachen will, wird auf die in Kapitel 7 vorgeschlagenen Folgeschritte verwiesen.
