# rag_chat_test

Standalone test harness for `POST /api/v1/chat`. Lives entirely outside
`backend/app` — it only talks to the running backend over HTTP, like any
other client, and never imports or modifies backend source code.

It does **not** run on its own. It only sends requests when you
explicitly run the script.

## What it does

1. Reads every question from `input/questions.json`.
2. Sends them to `http://localhost:8000/api/v1/chat` **one at a time**
   (never in parallel). By default there's no timeout — the runner waits
   until the API answers, however long that takes
   (`config.REQUEST_TIMEOUT_SECONDS = None`); set it to a number of
   seconds if you want a hard cutoff instead (a timed-out question is
   marked `TIMEOUT` and the runner moves to the next one).
3. For every question that gets a real answer, sends the question +
   answer + citations to an LLM judge (Groq, `openai/gpt-oss-120b`) that
   scores it on relevance, groundedness (vs. its own citations),
   completeness, clarity, and hallucination risk.
4. **Reranking/retrieval variants:** `/api/v1/chat` runs one fixed
   pipeline with no way to change retrieval settings per request. So for
   each question, the same question is *also* sent to
   `POST /api/v1/rag/ask` once per entry in `config.RERANK_VARIANTS` —
   that endpoint exposes the actual retrieval knobs (`alpha`:
   BM25-vs-vector weighting, `rerank`: on/off, `top_k`). Out of the box
   there are 4 variants: no rerank (hybrid), rerank + hybrid, rerank +
   vector-only, rerank + BM25-only. Each variant is tagged with a
   human-readable `reranking_method` label (e.g. "Hybrid search
   (alpha=0.5) + BGE-Reranker-v2-M3") and printed to the console per
   question as it runs. Each variant's answer is independently judged by
   the same LLM, so every question ends up with several
   differently-retrieved answers to compare, plus a computed "best
   variant for this question" based on the judge's `overall_score`. Edit
   or extend the list in `config.py` (`RERANK_VARIANTS`), or turn this
   off entirely with `ENABLE_RERANK_VARIANTS = False`.
5. **Precision@10 / Recall@10:** for any question that provides ground
   truth in `questions.json` (see "Adding questions" below), computes
   how many of the top-10 returned citations are actually relevant
   (precision) and how much of the known-relevant set was found
   (recall) — for BOTH the `/api/v1/chat` answer and every reranking
   variant, so you can see which retrieval strategy actually retrieves
   the right material, not just which one the LLM judge scored highest.
   Never fabricated: a question with no ground truth reports `null`, not
   a guessed score.
6. **Robustness checks:** for any question that sets `expected_type` in
   `questions.json` (`"off_topic"`, `"uncovered"`, or `"covered"`),
   verifies the backend's classify/refuse/fallback rules actually fired
   the way they should — e.g. an `"off_topic"` question must get the
   fixed out-of-scope refusal with no citations, not a real answer. This
   directly checks the security/scope rules are enforced, not just that
   *an* answer came back. Skipped (not graded) for any question that
   doesn't set `expected_type` — ambiguous questions are left ungraded
   rather than guessed at.
7. Writes one detailed JSON report per run to `output/run_<timestamp>.json`
   (plus `output/latest.json`, always the most recent run) containing,
   per question: response time, HTTP status, the full raw API response,
   the chat answer, citations, the LLM judge's full evaluation
   (relevance/groundedness/completeness/clarity/overall scores,
   hallucination risk), precision@10/recall@10, and the robustness
   check result, PLUS every reranking variant's own answer/citations/
   timing/evaluation/reranking-method-label/precision-recall and the
   picked best variant — plus an aggregate summary (avg/min/max/median
   response time, success/timeout/error counts, average scores per
   dimension, hallucination-risk breakdown, average precision/recall,
   a robustness pass/fail breakdown by expected_type, and a per-variant
   comparison table: average score, average response time, average
   precision/recall, and how many questions each variant "won").

## What it can't measure

This only observes what each endpoint's HTTP response actually exposes:
the answer/reply, its citations, and total round-trip time. It has no
visibility into internal server-side numbers these endpoints don't
return (e.g. per-stage retrieval/reranking *relevance scores*, or the
latency of each individual LLM call inside the LangGraph workflow) —
getting those would require instrumenting the backend itself, which is
out of scope here. The reranking variants above compare *retrieval
strategies* (which chunks get used and in what order) by their effect on
final answer quality — not raw internal similarity/rerank scores.

## Setup

```bash
cd backend/rag_chat_test/main_script
pip install -r requirements.txt
```

Make sure the backend is running locally on `http://localhost:8000`
(the URL tested is `config.CHAT_API_URL`, editable in `config.py`).

The Groq API key and judge model are set in `config.py`
(`GROQ_API_KEY`, `GROQ_MODEL`). You can override the key without editing
the file by setting the `GROQ_API_KEY` environment variable before
running.

## Adding questions

Edit `input/questions.json` — add as many entries as you want:

```json
{
  "questions": [
    {
      "id": 1,
      "question": "How many days of annual leave do employees get?",
      "conversation_id": null,
      "expected_type": "covered",
      "relevant_document_names": ["Leave_Policy_1.pdf"]
    },
    {"id": 2, "question": "what is 2+2", "conversation_id": null, "expected_type": "off_topic"}
  ]
}
```

- `id` — any unique label, just used to identify the result.
- `question` — required.
- `conversation_id` — normally leave `null` so each question is tested
  independently. Set it only if you want a question to continue a
  specific earlier conversation (e.g. a `conversation_id` you saw in a
  previous run's output).
- `expected_type` — **optional.** One of `"off_topic"`, `"uncovered"`,
  or `"covered"`. Enables the robustness check for this question (see
  "What it does" above). Leave it out for anything genuinely ambiguous
  — the harness skips grading rather than guessing.
- `relevant_document_names` — **optional**, a list like
  `["Leave_Policy_1.pdf"]`. Enables precision@10/recall@10 for this
  question, matching against each returned citation's `document_name`.
  Coarser than exact chunk matching but much easier to author by hand.
- `relevant_chunk_ids` — **optional**, a list of exact `chunk_id` UUIDs
  (see a prior run's `citations` for real values). Takes priority over
  `relevant_document_names` when both are present — more precise, but
  you need to already know the right chunk IDs (e.g. from a previous
  run's output) to set it up.

There's no limit on how many questions you add.

## Running

```bash
cd backend/rag_chat_test/main_script
python test_runner.py
```

Progress prints to the console as each question runs. You can stop early
with `Ctrl+C` — whatever ran so far is still saved to the output report.

## Output

`output/run_<ISO-timestamp>.json`:

```json
{
  "summary": {
    "total_questions": 2,
    "success_count": 2,
    "timeout_count": 0,
    "error_count": 0,
    "avg_response_time_seconds": 3.21,
    "min_response_time_seconds": 1.05,
    "max_response_time_seconds": 5.37,
    "median_response_time_seconds": 3.21,
    "avg_relevance_score": 9.0,
    "avg_groundedness_score": 8.5,
    "avg_completeness_score": 8.0,
    "avg_clarity_score": 9.0,
    "avg_overall_score": 8.6,
    "hallucination_risk_breakdown": {"low": 2},
    "chat_pipeline_precision_at_10": 0.9,
    "chat_pipeline_recall_at_10": 1.0,
    "chat_pipeline_precision_recall_question_count": 2,
    "robustness_summary": {
      "total_checked": 2,
      "total_passed": 2,
      "total_failed": 0,
      "by_expected_type": {"covered": {"passed": 2, "failed": 0}}
    },
    "rerank_variant_comparison": {
      "chat_pipeline (/api/v1/chat)": {"reranking_method": null, "avg_overall_score": 8.6, "avg_response_time_seconds": 3.21, "times_best_for_a_question": 1, "avg_precision_at_10": null, "avg_recall_at_10": null},
      "no_rerank__hybrid_0.5": {"reranking_method": "Hybrid search (alpha=0.5), no reranking", "avg_overall_score": 7.0, "avg_response_time_seconds": 1.9, "times_best_for_a_question": 0, "avg_precision_at_10": 0.7, "avg_recall_at_10": 0.8},
      "rerank__hybrid_0.5": {"reranking_method": "Hybrid search (alpha=0.5) + BGE-Reranker-v2-M3", "avg_overall_score": 8.8, "avg_response_time_seconds": 2.6, "times_best_for_a_question": 1, "avg_precision_at_10": 0.9, "avg_recall_at_10": 1.0},
      "rerank__vector_only_1.0": {"reranking_method": "Vector-only search + BGE-Reranker-v2-M3", "avg_overall_score": 8.1, "avg_response_time_seconds": 2.5, "times_best_for_a_question": 0, "avg_precision_at_10": 0.8, "avg_recall_at_10": 0.9},
      "rerank__bm25_only_0.0": {"reranking_method": "BM25-only search + BGE-Reranker-v2-M3", "avg_overall_score": 6.5, "avg_response_time_seconds": 2.4, "times_best_for_a_question": 0, "avg_precision_at_10": 0.6, "avg_recall_at_10": 0.7}
    },
    "total_run_time_seconds": 12.4,
    "run_started_at": "...",
    "run_finished_at": "...",
    "target_api": "http://localhost:8000/api/v1/chat",
    "judge_model": "openai/gpt-oss-120b",
    "per_question_timeout_seconds": null
  },
  "results": [
    {
      "question_id": 1,
      "question": "What is this document about?",
      "status": "SUCCESS",
      "response_time_seconds": 1.05,
      "answer": "...",
      "citations": [...],
      "citation_count": 3,
      "returned_conversation_id": "...",
      "llm_evaluation": {
        "relevance_score": 9,
        "groundedness_score": 8,
        "completeness_score": 8,
        "clarity_score": 9,
        "overall_score": 8.5,
        "hallucination_risk": "low",
        "speed_assessment": "fast",
        "strengths": "...",
        "weaknesses": "...",
        "summary": "..."
      },
      "llm_evaluation_time_seconds": 0.9,
      "precision_recall_at_10": {
        "k": 10, "match_field": "document_name", "ground_truth": ["Leave_Policy_1.pdf"],
        "retrieved_at_k": ["Leave_Policy_1.pdf", "Leave_Policy_1.pdf"], "hits": 2,
        "precision_at_10": 1.0, "recall_at_10": 1.0
      },
      "robustness_check": {"expected_type": "covered", "checked": true, "passed": true, "detail": "grounded, cited answer as expected"},
      "raw_response": { "...full /api/v1/chat response..." },
      "rerank_variants": [
        {
          "variant_name": "rerank__hybrid_0.5",
          "reranking_method": "Hybrid search (alpha=0.5) + BGE-Reranker-v2-M3",
          "config": {"rerank": true, "alpha": 0.5, "limit": 30, "top_k": 10},
          "status": "SUCCESS",
          "response_time_seconds": 2.6,
          "answer": "...",
          "citations": [...],
          "citation_count": 3,
          "was_answerable": true,
          "llm_evaluation": { "...same shape as above..." },
          "llm_evaluation_time_seconds": 0.8,
          "precision_recall_at_10": { "...same shape as above..." },
          "raw_response": { "...full /api/v1/rag/ask response..." }
        }
      ],
      "best_variant": {"name": "rerank__hybrid_0.5", "overall_score": 8.8}
    }
  ]
}
```
