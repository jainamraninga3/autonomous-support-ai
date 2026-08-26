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
   vector-only, rerank + BM25-only. Each variant's answer is
   independently judged by the same LLM, so every question ends up with
   several differently-retrieved answers to compare, plus a computed
   "best variant for this question" based on the judge's `overall_score`.
   Edit or extend the list in `config.py` (`RERANK_VARIANTS`), or turn
   this off entirely with `ENABLE_RERANK_VARIANTS = False`.
5. Writes one detailed JSON report per run to `output/run_<timestamp>.json`
   (plus `output/latest.json`, always the most recent run) containing,
   per question: response time, HTTP status, the full raw API response,
   the chat answer, citations, and the LLM judge's full evaluation, PLUS
   every reranking variant's own answer/citations/timing/evaluation and
   the picked best variant — plus an aggregate summary (avg/min/max/
   median response time, success/timeout/error counts, average scores
   per dimension, hallucination-risk breakdown, and a per-variant
   comparison table: average score, average response time, and how many
   questions each variant "won").

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
    {"id": 1, "question": "What is this document about?", "conversation_id": null},
    {"id": 2, "question": "How many days of annual leave do employees get?", "conversation_id": null}
  ]
}
```

- `id` — any unique label, just used to identify the result.
- `conversation_id` — normally leave `null` so each question is tested
  independently. Set it only if you want a question to continue a
  specific earlier conversation (e.g. a `conversation_id` you saw in a
  previous run's output).

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
    "rerank_variant_comparison": {
      "chat_pipeline (/api/v1/chat)": {"avg_overall_score": 8.6, "avg_response_time_seconds": 3.21, "times_best_for_a_question": 1},
      "no_rerank__hybrid_0.5": {"avg_overall_score": 7.0, "avg_response_time_seconds": 1.9, "times_best_for_a_question": 0},
      "rerank__hybrid_0.5": {"avg_overall_score": 8.8, "avg_response_time_seconds": 2.6, "times_best_for_a_question": 1},
      "rerank__vector_only_1.0": {"avg_overall_score": 8.1, "avg_response_time_seconds": 2.5, "times_best_for_a_question": 0},
      "rerank__bm25_only_0.0": {"avg_overall_score": 6.5, "avg_response_time_seconds": 2.4, "times_best_for_a_question": 0}
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
      "raw_response": { "...full /api/v1/chat response..." },
      "rerank_variants": [
        {
          "variant_name": "rerank__hybrid_0.5",
          "config": {"rerank": true, "alpha": 0.5, "limit": 30, "top_k": 10},
          "status": "SUCCESS",
          "response_time_seconds": 2.6,
          "answer": "...",
          "citations": [...],
          "citation_count": 3,
          "was_answerable": true,
          "llm_evaluation": { "...same shape as above..." },
          "llm_evaluation_time_seconds": 0.8,
          "raw_response": { "...full /api/v1/rag/ask response..." }
        }
      ],
      "best_variant": {"name": "rerank__hybrid_0.5", "overall_score": 8.8}
    }
  ]
}
```
