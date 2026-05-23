# graphrag-vs-rag-bger

Reproduktions-Artefakt zur ZHAW-Bachelorarbeit *Citation-Graph-Retrieval vs. Embedding-Retrieval auf Schweizer Bundesgerichtsentscheiden* von Albert Gstöhl (FS 2026, Studiengang Wirtschaftsinformatik, ZHAW School of Management and Law, Betreuung Benjamin Kühnis).

## Was die Arbeit untersucht

Die Arbeit vergleicht fünf Retrieval-Architekturen unter identischen Bedingungen auf 12'678 Bundesgerichts-Sachverhalten (je 4'226 pro Amtssprache) gegen die in den Urteilen zitierten Leitentscheide als Ground Truth.

- **RAG** (Baseline, 60 ANN-Seeds aus BGE-M3)
- **Embedding-1Hop / Embedding-2Hop** (kNN-Expansion im Vektorraum als Kontrollbedingung)
- **GraphRAG-1Hop / GraphRAG-2Hop** (Expansion entlang Zitationskanten im Citation-Graphen)

Die zentrale Hypothese, dass Zitations-Traversierung bei gleicher Pool-Grösse bessere Kandidaten liefert als Embedding-Nachbarschaft, wird bestätigt (Pool-Recall-Faktor 14, gepaarter Bootstrap-CI 13.1 bis 15.1).

## Was in dieses Repo gehört

Dieses Repository wird nach Abgabe der Bachelorarbeit mit dem vollständigen Reproduktions-Material aufgefüllt. Der Plan ist in [SPEC.md](SPEC.md) dokumentiert und umfasst:

- Pipeline-Code (Stages 1 bis 3, Citation-Graph-Build, FastAPI-Web-UI)
- Bootstrap-Artefakte (`citation_graph.pkl`, `valid_ids.json`, `date_index.json`, `eval_queries.jsonl`)
- Resultate (Per-Query-JSONL pro System × Ranking × k)
- Thesis-Quellen (Markdown + LaTeX-Vorlage + Build-Skripte)

## Lizenz

[MIT](LICENSE).

## Kontakt

Albert Gstöhl, `albert@gstoehl.dev`
