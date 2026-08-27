"""
Reads the latest full test-run report (output/latest.json) and writes a
much smaller, easy-to-review file containing just each question's
question/answer/citations/robustness/precision-recall — for manual
review without wading through the full raw_response payloads.

Usage:
    python extract_qa.py            # reads output/latest.json
    python extract_qa.py run_2026-08-27T08-42-10-816976+00-00.json
"""

import json
import sys
from pathlib import Path

import config

INPUT_NAME = sys.argv[1] if len(sys.argv) > 1 else "latest.json"
INPUT_PATH = config.OUTPUT_DIR / INPUT_NAME
OUTPUT_PATH = config.OUTPUT_DIR / f"qa_extract__{INPUT_PATH.stem}.json"


def extract_record(r):
    return {
        "question_id": r.get("question_id"),
        "question": r.get("question"),
        "status": r.get("status"),
        "answer": r.get("answer"),
        "citation_count": r.get("citation_count"),
        "citations": [
            {"document_name": c.get("document_name"), "chunk_id": c.get("chunk_id")}
            for c in (r.get("citations") or [])
        ],
        "llm_evaluation": r.get("llm_evaluation"),
        "precision_recall_at_10": r.get("precision_recall_at_10"),
        "robustness_check": r.get("robustness_check"),
        "best_variant": r.get("best_variant"),
    }


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    extracted = {
        "source_file": INPUT_PATH.name,
        "summary": data.get("summary"),
        "questions": [extract_record(r) for r in results],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"Read {len(results)} questions from {INPUT_PATH}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
