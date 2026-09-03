"""
Standalone RAG chat test harness.

Sends every question in ../input/questions.json to POST /api/v1/chat
ONE AT A TIME (never in parallel), measures real response time, and asks
an LLM (Groq, openai/gpt-oss-120b) to independently judge the quality of
each answer (relevance, groundedness vs. its citations, completeness,
clarity, hallucination risk).

For each question, ALSO runs it through POST /api/v1/rag/ask once per
entry in config.RERANK_VARIANTS — that endpoint exposes retrieval knobs
/api/v1/chat doesn't (alpha = BM25-vs-vector weighting, rerank on/off),
so this produces several differently-retrieved answers to the same
question, each independently judged, so you can compare which retrieval/
reranking strategy actually answers best. Disable via
config.ENABLE_RERANK_VARIANTS = False.

Writes one big, detailed JSON report to ../output/.

This script sends NOTHING until you run it yourself:
    python test_runner.py

Each request (both /api/v1/chat and each /api/v1/rag/ask variant call)
respects config.REQUEST_TIMEOUT_SECONDS — None (the default) means wait
indefinitely for a response; set it to a number of seconds to abort and
mark a question TIMEOUT instead.

Scope note: this only measures what's observable from outside the API
(HTTP response time, the returned answer, and the returned citations).
It cannot see internal server-side numbers that these endpoints don't
return in their response (e.g. per-stage retrieval/reranking relevance
scores or individual LLM-call latencies inside the LangGraph workflow)
— doing that would require instrumenting the backend itself, which this
harness deliberately does not touch.
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone

import requests

try:
    from groq import Groq
except ImportError:
    print(
        "Missing dependency: run `pip install -r requirements.txt` in this "
        "folder first (needs `requests` and `groq`)."
    )
    sys.exit(1)

import config


EVAL_SYSTEM_PROMPT = """You are a strict, impartial evaluator of a RAG \
(Retrieval-Augmented Generation) chatbot's answers. You are given the \
user's question, the exact response time in seconds, the chatbot's \
answer, and the citations (source document + page) the chatbot returned \
as evidence for that answer.

Score the answer on these dimensions, each 1-10 (10 = best):
- relevance: does the answer actually address the question asked?
- groundedness: is the answer plausibly supported by its own citations \
(if citations exist)? If the answer makes claims with NO citations \
backing them, groundedness must be low. If there were zero citations \
AND the question clearly needed document knowledge, groundedness must \
be low.
- completeness: does the answer fully cover what was asked, or is it \
partial/vague?
- clarity: is the answer well-written, direct, and easy to understand?

Also assess:
- hallucination_risk: "low", "medium", or "high" — your judgment of \
whether the answer likely contains information not actually supported \
by its citations.
- speed_assessment: one short phrase judging the response_time_seconds \
value (e.g. "fast", "acceptable", "slow") — under 3s is fast, 3-8s is \
acceptable, over 8s is slow.

Respond with ONLY a single valid JSON object, no markdown fences, no \
extra text, in exactly this shape:
{
  "relevance_score": <1-10 integer>,
  "groundedness_score": <1-10 integer>,
  "completeness_score": <1-10 integer>,
  "clarity_score": <1-10 integer>,
  "overall_score": <1-10 integer, your overall judgment>,
  "hallucination_risk": "low|medium|high",
  "speed_assessment": "<short phrase>",
  "strengths": "<one or two sentences>",
  "weaknesses": "<one or two sentences>",
  "summary": "<one sentence overall verdict>"
}
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def describe_reranking_method(rerank, alpha):
    """Human-readable label for a retrieval/reranking configuration —
    e.g. "Hybrid search (alpha=0.5) + BGE-Reranker-v2-M3"."""
    if alpha == 0.0:
        base = "BM25-only search"
    elif alpha == 1.0:
        base = "Vector-only search"
    else:
        base = f"Hybrid search (alpha={alpha})"
    return f"{base} + BGE-Reranker-v2-M3" if rerank else f"{base}, no reranking"


def compute_precision_recall(citations, question_entry, k=None):
    """Precision@k / Recall@k against ground truth optionally provided in
    questions.json — "relevant_chunk_ids" (exact chunk_id match, precise)
    takes priority over "relevant_document_names" (document-level, coarser
    but easier to author by hand). Returns None if the question provides
    neither — this NEVER fabricates a score for an ungraded question.
    """
    k = k or config.PRECISION_RECALL_K
    chunk_ground_truth = question_entry.get("relevant_chunk_ids")
    doc_ground_truth = question_entry.get("relevant_document_names")

    if chunk_ground_truth:
        retrieved = [c.get("chunk_id") for c in (citations or [])]
        relevant = set(chunk_ground_truth)
        match_field = "chunk_id"
    elif doc_ground_truth:
        retrieved = [c.get("document_name") for c in (citations or [])]
        relevant = set(doc_ground_truth)
        match_field = "document_name"
    else:
        return None

    retrieved_at_k = retrieved[:k]
    # Precision: fraction of the top-k retrieved items that are relevant.
    hits = sum(1 for item in retrieved_at_k if item in relevant)
    precision_at_k = round(hits / len(retrieved_at_k), 3) if retrieved_at_k else 0.0
    # Recall: fraction of the DISTINCT relevant items found at least once
    # in the top-k. Using raw `hits` here would overcount for document-
    # level ground truth (e.g. 10 chunks all from the same 1 relevant
    # document would wrongly give hits=10, recall=10/1=10.0). Counting
    # distinct relevant items actually retrieved keeps recall in [0, 1]
    # for both chunk-level and document-level ground truth.
    distinct_hits = len(relevant & set(retrieved_at_k))
    recall_at_k = round(distinct_hits / len(relevant), 3) if relevant else None

    return {
        "k": k,
        "match_field": match_field,
        "ground_truth": sorted(relevant),
        "retrieved_at_k": retrieved_at_k,
        "hits": hits,
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
    }


def check_robustness(question_entry, answer_text, citations, answer_source=None):
    """Verify (not assume) that the backend's classify/refuse/fallback
    rules actually fired the way `expected_type` in questions.json says
    they should. Skipped (checked=False) if `expected_type` is absent —
    never grades a question nobody labeled.

    Prefers the backend's own `answer_source` field over string-matching
    the reply text. Current values: 'rag', 'off_topic', 'general_fallback',
    'unverified_fallback', 'small_talk'.

    `expected_type: "off_topic"` expects 'off_topic' — the fixed
    out-of-scope decline, sent without any LLM call.

    `expected_type: "uncovered"` accepts either fallback source:
    'general_fallback' (the documents didn't cover it) or
    'unverified_fallback' (they did, but the answer failed verification).
    Which one it was is reported per question; it doesn't change pass/fail.
    """
    expected = question_entry.get("expected_type")
    if not expected:
        return {
            "expected_type": None,
            "checked": False,
            "note": "No expected_type set for this question — add 'off_topic', "
            "'uncovered', or 'covered' to questions.json to enable this check.",
        }

    answer_text = answer_text or ""
    has_citations = bool(citations)

    # Both fallback paths answer a COMPANY question without the
    # documents. Grouped for pass/fail; the exact value is still reported
    # per question.
    fallback_sources = {"general_fallback", "unverified_fallback"}

    if answer_source is not None:
        is_off_topic = answer_source == "off_topic"
        is_fallback = answer_source in fallback_sources
        is_rag = answer_source == "rag"
        check_basis = f"answer_source ({answer_source})"
    else:
        is_off_topic = config.OUT_OF_SCOPE_MARKER in answer_text
        is_fallback = config.GENERAL_FALLBACK_MARKER in answer_text
        is_rag = not is_off_topic and not is_fallback
        check_basis = "marker_text_match (no answer_source in response — older backend?)"

    if expected == "off_topic":
        passed = is_off_topic and not has_citations
        detail = "out-of-scope decline fired correctly, no citations" if passed else (
            "expected the fixed out-of-scope decline with no citations — did not get it"
        )
    elif expected == "uncovered":
        passed = is_fallback and not has_citations
        detail = f"answered outside the documents ({answer_source or 'marker match'}), no citations" if passed else (
            "expected a general-knowledge answer with no citations — did not get it"
        )
    elif expected == "covered":
        passed = has_citations and is_rag
        detail = "grounded, cited answer as expected" if passed else (
            "expected a grounded answer with real citations — did not get it"
        )
    else:
        return {
            "expected_type": expected,
            "checked": False,
            "note": f"Unknown expected_type {expected!r} — use 'off_topic', 'uncovered', or 'covered'.",
        }

    return {"expected_type": expected, "checked": True, "passed": passed, "detail": detail, "check_basis": check_basis}


def load_questions():
    if not config.INPUT_FILE.exists():
        print(f"Input file not found: {config.INPUT_FILE}")
        sys.exit(1)
    with open(config.INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    if not questions:
        print(f"No questions found in {config.INPUT_FILE}")
        sys.exit(1)
    if config.RETEST_ONLY_IDS:
        wanted = set(config.RETEST_ONLY_IDS)
        questions = [q for q in questions if q.get("id") in wanted]
        print(f"RETEST_ONLY_IDS set — filtering to {len(questions)} question(s): {sorted(wanted)}")
        if not questions:
            print(f"None of the requested ids {sorted(wanted)} were found in {config.INPUT_FILE}")
            sys.exit(1)
    return questions


def _post_json(url, payload):
    """POST payload to url, returning a uniform result dict.

    Shared by both the /api/v1/chat call and the /api/v1/rag/ask
    (reranking-variant) calls — same timeout/error handling either way.
    """
    started = time.perf_counter()
    started_iso = now_iso()
    try:
        resp = requests.post(
            url,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json=payload,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started
        finished_iso = now_iso()

        if resp.status_code != 200:
            return {
                "status": "ERROR",
                "http_status": resp.status_code,
                "response_time_seconds": round(elapsed, 3),
                "started_at": started_iso,
                "finished_at": finished_iso,
                "error": f"Non-200 response: {resp.text[:2000]}",
                "raw_response": None,
            }

        body = resp.json()
        return {
            "status": "SUCCESS",
            "http_status": resp.status_code,
            "response_time_seconds": round(elapsed, 3),
            "started_at": started_iso,
            "finished_at": finished_iso,
            "error": None,
            "raw_response": body,
        }
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - started
        timeout_desc = "no timeout limit is set" if config.REQUEST_TIMEOUT_SECONDS is None else f"{config.REQUEST_TIMEOUT_SECONDS}s"
        return {
            "status": "TIMEOUT",
            "http_status": None,
            "response_time_seconds": round(elapsed, 3),
            "started_at": started_iso,
            "finished_at": now_iso(),
            "error": f"No response received ({timeout_desc}) — request aborted.",
            "raw_response": None,
        }
    except requests.exceptions.ConnectionError as exc:
        elapsed = time.perf_counter() - started
        return {
            "status": "ERROR",
            "http_status": None,
            "response_time_seconds": round(elapsed, 3),
            "started_at": started_iso,
            "finished_at": now_iso(),
            "error": f"Connection error (is the backend running on {url}?): {exc}",
            "raw_response": None,
        }
    except Exception as exc:  # noqa: BLE001 - report any unexpected failure into the report, don't crash the run
        elapsed = time.perf_counter() - started
        return {
            "status": "ERROR",
            "http_status": None,
            "response_time_seconds": round(elapsed, 3),
            "started_at": started_iso,
            "finished_at": now_iso(),
            "error": f"Unexpected error: {exc}",
            "raw_response": None,
        }


def call_chat_api(question_text, conversation_id):
    payload = {"message": question_text}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return _post_json(config.CHAT_API_URL, payload)


def call_rag_ask_api(question_text, variant):
    payload = {
        "question": question_text,
        "limit": variant.get("limit", 30),
        "alpha": variant["alpha"],
        "rerank": variant["rerank"],
        "top_k": variant.get("top_k", 10),
    }
    return _post_json(config.RAG_ASK_API_URL, payload)


def evaluate_with_llm(groq_clients, question_text, answer, citations, response_time_seconds):
    """Ask the Groq judge model to score one answer. Tries each client in
    `groq_clients` in order, falling through to the next on a rate-limit
    (429) error — same idea as the backend's own multi-key fallback, so
    one exhausted key doesn't stop every question from being judged.
    Returns (evaluation_dict, eval_time_seconds, error)."""
    user_prompt = json.dumps(
        {
            "question": question_text,
            "response_time_seconds": response_time_seconds,
            "answer": answer or "",
            "citations": citations or [],
            "citation_count": len(citations or []),
        },
        ensure_ascii=False,
    )

    started = time.perf_counter()
    last_error = None
    for client in groq_clients:
        try:
            completion = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=config.GROQ_EVAL_TIMEOUT_SECONDS,
            )
            raw_text = completion.choices[0].message.content.strip()
            # Strip accidental markdown fences if the model adds them anyway.
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].strip()
            evaluation = json.loads(raw_text)
            eval_time = time.perf_counter() - started
            return evaluation, round(eval_time, 3), None
        except Exception as exc:  # noqa: BLE001 - evaluation failure shouldn't kill the run
            last_error = exc
            if "rate_limit" in str(exc) or "429" in str(exc):
                continue  # try the next key
            break  # non-rate-limit error — no point retrying with another key
    eval_time = time.perf_counter() - started
    return None, round(eval_time, 3), f"Evaluation failed: {last_error}"


def build_summary(results):
    total = len(results)
    success = [r for r in results if r["status"] == "SUCCESS"]
    timeouts = [r for r in results if r["status"] == "TIMEOUT"]
    errors = [r for r in results if r["status"] == "ERROR"]

    response_times = [r["response_time_seconds"] for r in results]
    scored = [r for r in success if r.get("llm_evaluation")]

    summary = {
        "total_questions": total,
        "success_count": len(success),
        "timeout_count": len(timeouts),
        "error_count": len(errors),
        "avg_response_time_seconds": round(statistics.mean(response_times), 3) if response_times else None,
        "min_response_time_seconds": round(min(response_times), 3) if response_times else None,
        "max_response_time_seconds": round(max(response_times), 3) if response_times else None,
        "median_response_time_seconds": round(statistics.median(response_times), 3) if response_times else None,
    }

    if scored:
        for dim in ("relevance_score", "groundedness_score", "completeness_score", "clarity_score", "overall_score"):
            vals = [r["llm_evaluation"][dim] for r in scored if isinstance(r["llm_evaluation"].get(dim), (int, float))]
            summary[f"avg_{dim}"] = round(statistics.mean(vals), 2) if vals else None
        risk_counts = {}
        for r in scored:
            risk = r["llm_evaluation"].get("hallucination_risk", "unknown")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        summary["hallucination_risk_breakdown"] = risk_counts

        # Scores broken down by expected_type (covered/uncovered/off_topic).
        # A single blended average is misleading: a "covered" question that
        # gets a real grounded answer and an "off_topic" question that
        # CORRECTLY refuses are both good outcomes, but the judge scores a
        # correct refusal low ("fails to answer the question") since it has
        # no way to know refusing was the right call. Judge answer quality
        # only within "covered" questions, where a real answer is expected.
        by_type_scores = {}
        for r in scored:
            expected = (r.get("robustness_check") or {}).get("expected_type")
            if not expected:
                continue
            by_type_scores.setdefault(expected, []).append(r["llm_evaluation"].get("overall_score"))
        if by_type_scores:
            summary["avg_overall_score_by_expected_type"] = {
                t: round(statistics.mean(v for v in vals if isinstance(v, (int, float))), 2)
                for t, vals in by_type_scores.items()
                if any(isinstance(v, (int, float)) for v in vals)
            }
            summary["avg_overall_score_note"] = (
                "The blended avg_overall_score above mixes 'covered' questions (where a real "
                "grounded answer is expected) with 'off_topic'/'uncovered' ones (where refusing "
                "or disclosing is the CORRECT behavior, but the judge doesn't know that and scores "
                "it low). avg_overall_score_by_expected_type separates these — judge 'covered' "
                "answer quality using its own number, not the blended one."
            )
    else:
        summary["note"] = "No questions were successfully evaluated by the LLM judge."

    # --- Precision@10 / Recall@10 for the /api/v1/chat pipeline, only
    # across questions that provided ground truth (relevant_document_names
    # or relevant_chunk_ids in questions.json) ---
    chat_pr = [r["precision_recall_at_10"] for r in results if r.get("precision_recall_at_10")]
    if chat_pr:
        summary["chat_pipeline_precision_at_10"] = round(
            statistics.mean(pr["precision_at_10"] for pr in chat_pr), 3
        )
        recalls = [pr["recall_at_10"] for pr in chat_pr if pr["recall_at_10"] is not None]
        summary["chat_pipeline_recall_at_10"] = round(statistics.mean(recalls), 3) if recalls else None
        summary["chat_pipeline_precision_recall_question_count"] = len(chat_pr)
    else:
        summary["chat_pipeline_precision_recall_note"] = (
            "No question provided ground truth (relevant_document_names / relevant_chunk_ids) — "
            "precision/recall not computed for any question."
        )

    # --- Robustness: did each question's actual behavior match its
    # expected_type (off_topic/uncovered/covered), where labeled? ---
    checked = [r["robustness_check"] for r in results if r.get("robustness_check", {}).get("checked")]
    if checked:
        by_type = {}
        for rc in checked:
            t = rc["expected_type"]
            bucket = by_type.setdefault(t, {"passed": 0, "failed": 0})
            bucket["passed" if rc["passed"] else "failed"] += 1
        for bucket in by_type.values():
            bucket["total"] = bucket["passed"] + bucket["failed"]
            bucket["rate"] = round(bucket["passed"] / bucket["total"], 3) if bucket["total"] else None
        summary["robustness_summary"] = {
            "total_checked": len(checked),
            "total_passed": sum(1 for rc in checked if rc["passed"]),
            "total_failed": sum(1 for rc in checked if not rc["passed"]),
            "by_expected_type": by_type,
        }
    else:
        summary["robustness_note"] = (
            "No question set expected_type — robustness rule-compliance not checked for any question."
        )

    # --- Reranking/retrieval variant comparison, aggregated across all questions ---
    variant_stats = {}  # name -> {scores, times, wins, precisions, recalls}
    for r in results:
        for vr in r.get("rerank_variants", []):
            name = vr["variant_name"]
            stats = variant_stats.setdefault(
                name, {"scores": [], "times": [], "wins": 0, "precisions": [], "recalls": [], "method": vr.get("reranking_method")}
            )
            if vr["status"] == "SUCCESS":
                stats["times"].append(vr["response_time_seconds"])
                if vr.get("llm_evaluation") and isinstance(vr["llm_evaluation"].get("overall_score"), (int, float)):
                    stats["scores"].append(vr["llm_evaluation"]["overall_score"])
                if vr.get("precision_recall_at_10"):
                    stats["precisions"].append(vr["precision_recall_at_10"]["precision_at_10"])
                    if vr["precision_recall_at_10"]["recall_at_10"] is not None:
                        stats["recalls"].append(vr["precision_recall_at_10"]["recall_at_10"])
        best = r.get("best_variant")
        if best and best["name"] in variant_stats:
            variant_stats[best["name"]]["wins"] += 1
        elif best and best["name"] == "chat_pipeline (/api/v1/chat)":
            variant_stats.setdefault(
                "chat_pipeline (/api/v1/chat)",
                {"scores": [], "times": [], "wins": 0, "precisions": [], "recalls": [], "method": None},
            )
            variant_stats["chat_pipeline (/api/v1/chat)"]["wins"] += 1

    if variant_stats:
        summary["rerank_variant_comparison"] = {
            name: {
                "reranking_method": s["method"],
                "avg_overall_score": round(statistics.mean(s["scores"]), 2) if s["scores"] else None,
                "avg_response_time_seconds": round(statistics.mean(s["times"]), 3) if s["times"] else None,
                "times_best_for_a_question": s["wins"],
                "avg_precision_at_10": round(statistics.mean(s["precisions"]), 3) if s["precisions"] else None,
                "avg_recall_at_10": round(statistics.mean(s["recalls"]), 3) if s["recalls"] else None,
            }
            for name, s in variant_stats.items()
        }

    return summary


def main():
    questions = load_questions()
    groq_clients = [Groq(api_key=key) for key in config.GROQ_JUDGE_API_KEYS]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(questions)} question(s) from {config.INPUT_FILE}")
    timeout_display = "none (waits indefinitely)" if config.REQUEST_TIMEOUT_SECONDS is None else f"{config.REQUEST_TIMEOUT_SECONDS}s"
    print(f"Target API: {config.CHAT_API_URL}  |  per-question timeout: {timeout_display}")
    print(f"Judge model: {config.GROQ_MODEL} (Groq)\n")

    results = []
    last_conversation_id = None
    run_started_iso = now_iso()
    run_started = time.perf_counter()

    try:
        for idx, q in enumerate(questions, start=1):
            q_id = q.get("id", idx)
            q_text = q.get("question", "").strip()
            if not q_text:
                print(f"[{idx}/{len(questions)}] id={q_id} — skipped (empty question)")
                continue

            conversation_id = q.get("conversation_id")
            if not conversation_id and config.CHAIN_CONVERSATIONS:
                conversation_id = last_conversation_id

            print(f"[{idx}/{len(questions)}] id={q_id} — asking: {q_text!r}")
            api_result = call_chat_api(q_text, conversation_id)
            print(f"    -> status={api_result['status']}  time={api_result['response_time_seconds']}s")

            record = {
                "question_id": q_id,
                "question": q_text,
                "conversation_id_used": conversation_id,
                "status": api_result["status"],
                "http_status": api_result["http_status"],
                "response_time_seconds": api_result["response_time_seconds"],
                "started_at": api_result["started_at"],
                "finished_at": api_result["finished_at"],
                "error": api_result["error"],
                "answer": None,
                "citations": [],
                "citation_count": 0,
                "returned_conversation_id": None,
                "llm_evaluation": None,
                "llm_evaluation_time_seconds": None,
                "llm_evaluation_error": None,
                "raw_response": api_result["raw_response"],
            }

            if api_result["status"] == "SUCCESS" and api_result["raw_response"]:
                body = api_result["raw_response"]
                record["answer"] = body.get("reply")
                record["citations"] = body.get("citations", [])
                record["citation_count"] = len(record["citations"])
                record["returned_conversation_id"] = body.get("conversation_id")
                record["answer_source"] = body.get("answer_source")
                record["retrieved_chunk_count"] = body.get("retrieved_chunk_count")
                last_conversation_id = record["returned_conversation_id"] or last_conversation_id

                print("    -> asking Groq judge model to evaluate this answer...")
                evaluation, eval_time, eval_error = evaluate_with_llm(
                    groq_clients, q_text, record["answer"], record["citations"], api_result["response_time_seconds"]
                )
                record["llm_evaluation"] = evaluation
                record["llm_evaluation_time_seconds"] = eval_time
                record["llm_evaluation_error"] = eval_error
                if evaluation:
                    print(f"    -> overall_score={evaluation.get('overall_score')} "
                          f"hallucination_risk={evaluation.get('hallucination_risk')} "
                          f"(eval took {eval_time}s)")
                else:
                    print(f"    -> evaluation failed: {eval_error}")

                record["precision_recall_at_10"] = compute_precision_recall(record["citations"], q)
                if record["precision_recall_at_10"]:
                    pr = record["precision_recall_at_10"]
                    print(f"    -> precision@10={pr['precision_at_10']} recall@10={pr['recall_at_10']} "
                          f"(matched by {pr['match_field']})")

                record["robustness_check"] = check_robustness(
                    q, record["answer"], record["citations"], body.get("answer_source")
                )
                if record["robustness_check"]["checked"]:
                    rc = record["robustness_check"]
                    verdict = "PASS" if rc["passed"] else "FAIL"
                    print(f"    -> robustness [{rc['expected_type']}]: {verdict} — {rc['detail']}")
            else:
                record["precision_recall_at_10"] = None
                record["robustness_check"] = check_robustness(q, None, [])

            record["rerank_variants"] = []
            if config.ENABLE_RERANK_VARIANTS:
                for variant in config.RERANK_VARIANTS:
                    method_label = describe_reranking_method(variant["rerank"], variant["alpha"])
                    print(f"    -> [variant: {variant['name']}] {method_label} — querying /api/v1/rag/ask "
                          f"for question {q_id!r} ({q_text!r})...")
                    variant_result = call_rag_ask_api(q_text, variant)
                    print(f"       status={variant_result['status']}  time={variant_result['response_time_seconds']}s")

                    variant_record = {
                        "variant_name": variant["name"],
                        "reranking_method": method_label,
                        "config": {
                            "rerank": variant["rerank"],
                            "alpha": variant["alpha"],
                            "limit": variant.get("limit", 30),
                            "top_k": variant.get("top_k", 10),
                        },
                        "status": variant_result["status"],
                        "http_status": variant_result["http_status"],
                        "response_time_seconds": variant_result["response_time_seconds"],
                        "error": variant_result["error"],
                        "answer": None,
                        "citations": [],
                        "citation_count": 0,
                        "was_answerable": None,
                        "llm_evaluation": None,
                        "llm_evaluation_time_seconds": None,
                        "llm_evaluation_error": None,
                        "precision_recall_at_10": None,
                        "raw_response": variant_result["raw_response"],
                    }

                    if variant_result["status"] == "SUCCESS" and variant_result["raw_response"]:
                        vbody = variant_result["raw_response"]
                        variant_record["answer"] = vbody.get("answer")
                        variant_record["citations"] = vbody.get("citations", [])
                        variant_record["citation_count"] = len(variant_record["citations"])
                        variant_record["was_answerable"] = vbody.get("was_answerable")

                        v_eval, v_eval_time, v_eval_error = evaluate_with_llm(
                            groq_clients,
                            q_text,
                            variant_record["answer"],
                            variant_record["citations"],
                            variant_result["response_time_seconds"],
                        )
                        variant_record["llm_evaluation"] = v_eval
                        variant_record["llm_evaluation_time_seconds"] = v_eval_time
                        variant_record["llm_evaluation_error"] = v_eval_error
                        if v_eval:
                            print(f"       overall_score={v_eval.get('overall_score')} "
                                  f"hallucination_risk={v_eval.get('hallucination_risk')}")

                        variant_record["precision_recall_at_10"] = compute_precision_recall(
                            variant_record["citations"], q
                        )
                        if variant_record["precision_recall_at_10"]:
                            pr = variant_record["precision_recall_at_10"]
                            print(f"       precision@10={pr['precision_at_10']} recall@10={pr['recall_at_10']}")

                    record["rerank_variants"].append(variant_record)

                # Rank this question's variants (chat answer + every rerank variant) by overall_score.
                candidates = []
                if record.get("llm_evaluation"):
                    candidates.append(("chat_pipeline (/api/v1/chat)", record["llm_evaluation"].get("overall_score")))
                for vr in record["rerank_variants"]:
                    if vr.get("llm_evaluation"):
                        candidates.append((vr["variant_name"], vr["llm_evaluation"].get("overall_score")))
                scored_candidates = [c for c in candidates if isinstance(c[1], (int, float))]
                if scored_candidates:
                    best = max(scored_candidates, key=lambda c: c[1])
                    record["best_variant"] = {"name": best[0], "overall_score": best[1]}
                    print(f"    -> best for this question: {best[0]} (overall_score={best[1]})")

            results.append(record)
            print()
    except KeyboardInterrupt:
        print("\nInterrupted by user — saving partial results collected so far...\n")

    total_run_time = round(time.perf_counter() - run_started, 3)
    summary = build_summary(results)
    summary["total_run_time_seconds"] = total_run_time
    summary["run_started_at"] = run_started_iso
    summary["run_finished_at"] = now_iso()
    summary["target_api"] = config.CHAT_API_URL
    summary["judge_model"] = config.GROQ_MODEL
    summary["per_question_timeout_seconds"] = config.REQUEST_TIMEOUT_SECONDS

    report = {"summary": summary, "results": results}

    timestamp = run_started_iso.replace(":", "-").replace(".", "-")
    out_file = config.OUTPUT_DIR / f"run_{timestamp}.json"
    latest_file = config.OUTPUT_DIR / "latest.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nFull detailed report written to:\n  {out_file}\n  {latest_file}")


if __name__ == "__main__":
    main()
