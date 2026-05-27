"""Aufschlüsselung der Ground-Truth-Drops zwischen swiss_doc2doc_ir und eval_queries.jsonl.

Reproduziert die in Kapitel 6 (Selbstverstärkung der Leitentscheid-Präferenz im Ranking)
zitierten Zahlen:
- 46.6 Prozent der Original-GT-Einträge werden vom strict-GT-Filter entfernt
- 30.6 Prozent davon wegen fehlendem date_ms (Closed-World-Datumsfilter)
- 15.9 Prozent davon wegen fehlendem Qdrant-Embedding (V∩G-Filter)
- Pro Query reduziert sich die durchschnittliche GT-Menge von 8.92 auf 4.77.

Ausführung:
    python -m scripts.analysis.gt_filter_breakdown
"""
import ast
import json

from datasets import load_dataset


EVAL_QUERIES = "data/eval/eval_queries.jsonl"
VALID_IDS = "data/eval/valid_ids.json"
DATE_INDEX = "data/eval/date_index.json"


def main():
    eval_queries = []
    with open(EVAL_QUERIES) as f:
        for line in f:
            eval_queries.append(json.loads(line))

    orig_gt = {}
    for split in ["train", "validation", "test"]:
        ds = load_dataset("rcds/swiss_doc2doc_ir", split=split)
        for item in ds:
            cited = item.get("cited_rulings") or "[]"
            try:
                gts = ast.literal_eval(cited)
            except Exception:
                gts = []
            orig_gt[item["decision_id"]] = set(gts)

    with open(VALID_IDS) as f:
        valid_ids = set(json.load(f))
    with open(DATE_INDEX) as f:
        date_index = json.load(f)

    reasons = {
        "kept": 0,
        "date_missing": 0,
        "not_in_VG": 0,
        "temporal_fail": 0,
        "other": 0,
    }
    total = 0

    for q in eval_queries:
        qid = q["query_id"]
        qdate = q.get("date_ms", 0)
        if qid not in orig_gt:
            continue
        filtered = set(q["ground_truth_cases"])
        for gt in orig_gt[qid]:
            total += 1
            if gt in filtered:
                reasons["kept"] += 1
            elif gt not in valid_ids:
                reasons["not_in_VG"] += 1
            elif gt not in date_index:
                reasons["date_missing"] += 1
            elif date_index[gt] >= qdate:
                reasons["temporal_fail"] += 1
            else:
                reasons["other"] += 1

    print(f"queries: {len(eval_queries)}")
    print(f"total GT entries: {total}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:7d} ({100*v/total:5.1f} %)")
    n_comparable = sum(1 for q in eval_queries if q["query_id"] in orig_gt)
    avg_orig = sum(len(orig_gt[q["query_id"]]) for q in eval_queries
                   if q["query_id"] in orig_gt) / n_comparable
    avg_filt = sum(len(q["ground_truth_cases"]) for q in eval_queries) / len(eval_queries)
    print(f"\navg original GT/query: {avg_orig:.2f}")
    print(f"avg filtered GT/query: {avg_filt:.2f}")


if __name__ == "__main__":
    main()
