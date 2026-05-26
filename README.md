# graphrag-vs-rag-bger

Reproduktions-Artefakt zur ZHAW-Bachelorarbeit *AI-Agent für den Rechtsdienst* von Albert Gstöhl (FS 2026, Studiengang Wirtschaftsinformatik, ZHAW School of Management and Law, Betreuung Benjamin Kühnis).

Die Arbeit vergleicht fünf Retrieval-Architekturen unter identischen Bedingungen auf 12'678 Bundesgerichts-Sachverhalten (je 4'226 pro Amtssprache) gegen die in den Urteilen zitierten Leitentscheide als Ground Truth. Das PDF der Arbeit liegt als GitHub-Release `v1.0-thesis` an diesem Repo.

## Vergleichene Systeme

- **RAG** (Baseline, 60 ANN-Seeds aus BGE-M3)
- **Embedding-1Hop** und **Embedding-2Hop** (kNN-Expansion im Vektorraum als Kontrollbedingung)
- **GraphRAG-1Hop** und **GraphRAG-2Hop** (Expansion entlang Zitationskanten im Citation-Graph)

Drei Ranking-Strategien pro System (Cosine, Cross-Encoder, In-Degree), je vier k-Werte (5, 10, 15, 20).

## Headline-Befund

Die zentrale Hypothese, dass Zitations-Traversierung bei gleicher Pool-Grösse bessere Kandidaten liefert als Embedding-Nachbarschaft, wird bestätigt.

| Kennzahl                                   | Wert                    |
| ------------------------------------------ | ----------------------- |
| Pool-Recall-Faktor GraphRAG-1Hop vs. RAG   | 14× (CI 13.1 bis 15.1)  |
| Ranked Recall@20 GraphRAG-1Hop vs. RAG     | 10.4×                   |
| Pool-Recall-Ceiling GraphRAG-1Hop          | 47.4 Prozent            |

Vollständige Tabellen, Boxplots und sprachweise Aufschlüsselung in Kapitel 5 der Arbeit.

## Repository-Layout

```
scripts/
  eval/
    01_sample_queries.py        Stage 1, stratifiziertes Query-Sampling
    02_run_retrieval.py         Stage 2, Retrieval gegen 5 Systeme
    03_compute_metrics.py       Stage 3, IR-Metriken plus Graph-Nearness
    build_citation_graph.py     Citation-Graph aus rcds/swiss_doc2doc_ir bauen
    build_valid_ids.py          V geschnitten G berechnen (Qdrant intersect Graph)
    build_facts_index.py        facts-Index für HF-freies Stage 1
    04_run_pipeline.sh          End-to-End-Pipeline-Runner
  migrate_qdrant_date_ms.py     Qdrant payload-Migration für date_ms-Index
  embedding/
    embed_clean.py              Bulk-Embed via TEI, schreibt Numpy-Shards
    load_qdrant.py              Numpy-Shards in Qdrant uploaden
    embed_missing.py            Recovery für Ingest-Lücken (idempotent)
    upload_snapshot_hf.sh       Einmalig, Qdrant-Snapshot auf HF Hub publizieren
    restore_qdrant.sh           Snapshot von HF Hub laden und in Qdrant restoren

webui/                          FastAPI + HTMX UI, Inspector, Live-Logs

data/eval/                      Bootstrap-Artefakte (Git LFS)
  citation_graph.pkl            NetworkX DiGraph, 158'881 Knoten, 1.6M Kanten
  valid_ids.json                V geschnitten G, 131'734 Decision-IDs
  date_index.json               decision_id zu ms-Timestamp Map
  eval_queries.jsonl            Query-Set mit Ground Truth
  facts_index.jsonl             facts-Texte pro Decision-ID
  metrics/                      60 per_query_*.jsonl plus 4 Aggregat-CSVs
  results/                      *_layers.jsonl für die Recall-Ceiling-Wasserfälle

plan/meetings/                  Advisor-Meeting-Notizen
```

Schwere Roh-Artefakte (Pool-JSONLs pro System, geteilte Cross-Encoder-Scores, Embeddings) liegen separat als Release-Asset `v1.0-thesis-data`. Diese sind nur nötig, wenn jemand Ranking-Strategien neu evaluieren möchte. Für die in der Arbeit berichteten Tabellen reichen die per_query-Metriken aus `data/eval/metrics/`.

## Voraussetzungen

- Python 3.11 oder neuer
- Docker (oder Kubernetes) für die Modell-Services
- GPU mit min. 24 GB VRAM für den Cross-Encoder
- min. 32 GB RAM auf dem Pipeline-Client für den Citation-Graph
- Git LFS für die Bootstrap-Artefakte

Hardware-Setup des Original-Runs: 8× RTX 3090 (je 24 GB) auf einem Kubernetes-Cluster, BGE-M3 plus vier Reranker-Replicas (eine pro GPU für die Cross-Encoder-Parallelisierung). Ein einzelner GPU mit 24 GB reicht für die Reproduktion, dann ohne Reranker-Fan-out.

## Quick Start

1. Repo klonen plus LFS-Inhalte ziehen

   Voraussetzung: das `git-lfs`-Systempaket muss vorhanden sein
   (`apt install git-lfs` auf Debian/Ubuntu, `pacman -S git-lfs` auf Arch).

   ```bash
   git lfs install
   git clone https://github.com/albertgstoehl/graphrag-vs-rag-bger.git
   cd graphrag-vs-rag-bger
   git lfs pull
   ```

2. Modell-Services starten

   ```bash
   # Qdrant (Pin auf v1.13.0, identisch zum Snapshot-Produzenten)
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.13.0

   # TEI Embed (BGE-M3)
   docker run -d --name tei-embed --gpus all -p 8010:80 \
     -v $HOME/.cache/huggingface:/data \
     ghcr.io/huggingface/text-embeddings-inference:1.7 \
     --model-id BAAI/bge-m3

   # TEI Rerank (bge-reranker-v2-m3)
   docker run -d --name tei-rerank --gpus all -p 8011:80 \
     -v $HOME/.cache/huggingface:/data \
     ghcr.io/huggingface/text-embeddings-inference:1.7 \
     --model-id BAAI/bge-reranker-v2-m3
   ```

3. Korpus indexieren

   Die Qdrant-Collection `bger` enthält rund 3 Millionen BGE-M3-Embeddings der `rcds/swiss_rulings` plus `rcds/swiss_leading_decisions`-Chunks. Zwei Pfade stehen zur Auswahl.

   **Schnellpfad, Snapshot von Hugging Face wiederherstellen.** Der zur Arbeit gehörende Qdrant-Snapshot ist als Hugging-Face-Dataset publiziert (`albertstudy/graphrag-vs-rag-bger-snapshot`, rund 24 GiB). Das Restore-Skript lädt die Datei, prüft den SHA256 gegen den im Repo eingecheckten Manifest und spielt sie via Qdrant-`/snapshots/upload`-Endpoint in eine laufende Instanz ein.

   ```bash
   pip install -U "huggingface_hub[cli]"
   hf auth login   # einmalig, mit Read-Token von hf.co/settings/tokens
   ./scripts/embedding/restore_qdrant.sh
   ```

   Der Restore dauert auf einer SSD etwa 5 Minuten, der Download je nach Anbindung 10 bis 30 Minuten. Anschliessend ist die Collection mit identischem Inhalt wie im finalen Lauf der Arbeit bereit.

   **Langpfad, Embeddings selbst rechnen.** Wer Zugang zu einer GPU hat und das volle Pipeline durchlaufen will, baut die Embeddings via TEI selbst. Dauert auf einer einzelnen RTX 3090 etwa 8 bis 12 Stunden.

   ```bash
   python scripts/embedding/embed_clean.py     # 2k-Token-Chunks via TEI embedden
   python scripts/embedding/load_qdrant.py     # Numpy-Shards in Qdrant uploaden
   python scripts/embedding/embed_missing.py \ # Optional, schliesst nachträgliche Lücken
       --missing-ids-file <pfad>
   python scripts/migrate_qdrant_date_ms.py    # date_ms-Payload-Index nachziehen
   ```

   Das `embed_missing.py`-Skript ist die korrigierte Nachindex-Variante, die im Mai 2026 die im ursprünglichen Bulk-Lauf verlorenen 3.3 Prozent der `swiss_rulings`-Decisions ergänzt hat. Im publizierten Snapshot ist diese Lücke bereits geschlossen.

4. Bootstrap-Artefakte sind bereits in `data/eval/` per Git LFS. Optional neu bauen:

   ```bash
   python scripts/eval/build_citation_graph.py
   python scripts/eval/build_valid_ids.py
   python scripts/eval/build_facts_index.py
   ```

5. Pipeline starten

   ```bash
   bash scripts/eval/04_run_pipeline.sh
   ```

   Oder bequemer über die Web-UI:

   ```bash
   pip install -r webui/requirements.txt
   uvicorn webui.app:app --host 127.0.0.1 --port 8000
   ```

   Dann `http://127.0.0.1:8000/` öffnen, unter `/settings` Host und Ports anpassen, im Dashboard `Run` klicken.

   Achtung beim Smoketest: `01_sample_queries.py --n <klein>` schreibt nach `data/eval/eval_queries.jsonl` und überschreibt damit die committed 12'678-Query-Baseline. Vor kleineren Test-Läufen entweder die Datei sichern oder `EVAL_DIR=/tmp/eval-smoke` setzen. Rankings und k-Werte sind in `02_run_retrieval.py` aktuell hartcodiert (alle drei Rankings × vier k-Werte), das Script akzeptiert `--systems` und `--dry-run`, aber kein `--rankings`-Flag.

## Resultate inspizieren

- `/metrics` für Aggregat-Tabellen (Recall, NDCG, MRR, Graph-Nearness pro System × Ranking × k)
- `/metrics/recall-ceiling` für die Pool-Recall-Wasserfälle
- `/metrics/graph-nearness` für die Graph-Nearness-Verteilung
- `/inspector/{query_id}` für Per-Query-Drilldown mit Graph-Visualisierung

## PDF der Arbeit

Die finale Bachelorarbeit liegt als Release-Asset unter
[Releases / v1.0-thesis](https://github.com/albertgstoehl/graphrag-vs-rag-bger/releases/tag/v1.0-thesis).

## Zitieren

```bibtex
@thesis{gstohl2026graphrag,
  author = {Gst{\"o}hl, Albert},
  title  = {AI-Agent für den Rechtsdienst},
  school = {ZHAW School of Management and Law},
  year   = {2026},
  type   = {Bachelorarbeit},
  url    = {https://github.com/albertgstoehl/graphrag-vs-rag-bger}
}
```

## Lizenz

[MIT](LICENSE) für den Code. Die Arbeit selbst (PDF im Release) untersteht der ZHAW-Publikationsregelung.

## Kontakt

Albert Gstöhl, `albert@gstoehl.dev`
