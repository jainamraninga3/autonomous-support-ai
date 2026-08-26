"""
Extracts just {question, answer} pairs out of a full test-run report
(../output/run_<timestamp>.json or ../output/latest.json) into a small,
simple JSON file — for when you just want the Q&A content without all
the timing/evaluation/raw-response detail.

Usage:
    python extract_qa.py                # reads ../output/latest.json
    python extract_qa.py run_....json   # reads a specific report file
"""

import json
import sys

import config


def main():
    source_name = sys.argv[1] if len(sys.argv) > 1 else "latest.json"
    source_path = config.OUTPUT_DIR / source_name
    if not source_path.exists():
        print(f"Report file not found: {source_path}")
        sys.exit(1)

    with open(source_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    qa_pairs = []
    for r in report.get("results", []):
        qa_pairs.append({
            "question_id": r.get("question_id"),
            "question": r.get("question"),
            "answer": r.get("answer"),
        })

    out_path = config.OUTPUT_DIR / "qa_extracted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"source_report": source_name, "qa_pairs": qa_pairs}, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(qa_pairs)} question/answer pairs from {source_path.name}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
