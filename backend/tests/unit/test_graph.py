"""Unit tests for the full LangGraph workflow (classify -> general | RAG
-> verify), using a scripted fake LLM client and a fake Weaviate client
— no real model or Weaviate connection needed.

Replaces the old placeholder-graph test: `build_graph()` used to compile
a single echo node just to prove LangGraph worked at all; it now takes
real dependencies and runs the full plan.md pipeline.
"""

from dataclasses import dataclass

import pytest

from app.graph.state import initial_state
from app.core.config import get_settings
from app.graph.workflow import build_graph
from app.llm.base import LLMClient


@dataclass
class _FakeMetadata:
    score: float


@dataclass
class _FakeObject:
    properties: dict
    metadata: _FakeMetadata


@dataclass
class _FakeHybridResponse:
    objects: list


class _FakeQueryHandle:
    def __init__(self, objects: list) -> None:
        self._objects = objects
        # Every hybrid() call's query text, so a test can assert how many
        # retrieval passes ran and in which language.
        self.queries: list[str] = []

    def hybrid(self, **kwargs):
        self.queries.append(kwargs.get("query", ""))
        return _FakeHybridResponse(objects=self._objects)


class _FakeCollection:
    def __init__(self, objects: list) -> None:
        self.query = _FakeQueryHandle(objects)


class _FakeCollections:
    def __init__(self, exists: bool, collection: _FakeCollection) -> None:
        self._exists = exists
        self._collection = collection

    def exists(self, name: str) -> bool:
        return self._exists

    def get(self, name: str):
        return self._collection


class _FakeWeaviateClient:
    def __init__(self, exists: bool = True, objects: list | None = None) -> None:
        self.collections = _FakeCollections(exists, _FakeCollection(objects or []))


def _fake_object(index: int, score: float) -> _FakeObject:
    return _FakeObject(
        properties={
            "chunk_id": f"chunk-{index}",
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "version": 1,
            "chunk_index": index,
            "start_page": 1,
            "end_page": 1,
            "content": f"content {index}",
        },
        metadata=_FakeMetadata(score=score),
    )


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 for _ in texts]


class _ScriptedLLMClient(LLMClient):
    """Returns a canned reply based on a distinctive substring in the
    prompt — each of the four prompt types this graph sends
    (classification, rewrite, grounded-answer, verification) has one."""

    def __init__(self, responses: dict[str, str], default: str = "a general reply") -> None:
        self.responses = responses
        self.default = default
        self.calls: list[str] = []

    async def generate_reply(self, message: str) -> str:
        self.calls.append(message)
        for marker, reply in self.responses.items():
            if marker in message:
                return reply
        return self.default


@pytest.mark.asyncio
async def test_general_query_is_refused_without_calling_the_llm_to_answer_it() -> None:
    """A classified-GENERAL query (off-topic — math, coding, general
    trivia) must be refused with a fixed message, not answered — and the
    raw user message must never be handed to the LLM for free-form
    answering (a real security boundary, not just a UX choice: it closes
    off a path a user could otherwise use to make the bot answer
    anything, or attempt a prompt injection)."""
    llm = _ScriptedLLMClient({"GENERAL, SMALL_TALK, or RAG_REQUIRED": "GENERAL"})
    graph = build_graph(llm_client=llm, weaviate_client=None)

    result = await graph.ainvoke(initial_state("what is 2 + 2?"))

    assert result["classification"] == "GENERAL"
    assert "I can only help" in result["response"]
    assert result["chunks"] == []
    assert result["citations"] == []
    # Exactly one LLM call happened (classification) — no second call was
    # made asking the LLM to freely answer the raw question.
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_rag_query_with_supported_answer_returns_the_generated_answer(monkeypatch) -> None:
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.graph.workflow.get_reranker", lambda: _FakeReranker())

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten leave policy question",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nEvery claim matches the context.",
        },
        default="Employees get 20 days of leave. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["classification"] == "RAG_REQUIRED"
    assert result["rewritten_query"] == "rewritten leave policy question"
    assert result["verified"] is True
    assert result["response"] == "Employees get 20 days of leave. [Source 1]"
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_rag_query_with_unsupported_answer_returns_the_refusal(monkeypatch) -> None:
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.graph.workflow.get_reranker", lambda: _FakeReranker())

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten query",
            "SUPPORTED or UNSUPPORTED": "UNSUPPORTED\nThe answer invents a number not in the context.",
        },
        default="some ungrounded answer",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["verified"] is False
    assert "could not fully verify" in result["response"]


@pytest.mark.asyncio
async def test_rag_query_with_no_weaviate_client_falls_back_to_disclosed_general_answer() -> None:
    llm = _ScriptedLLMClient(
        {"GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED", "Rewritten query:": "rewritten query"},
        default="a general-knowledge reply",
    )
    graph = build_graph(llm_client=llm, weaviate_client=None)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["chunks"] == []
    assert result["was_answerable"] is False
    assert result["verified"] is None
    assert result["citations"] == []
    assert "isn't covered by our available documents" in result["response"]
    assert "a general-knowledge reply" in result["response"]


@pytest.mark.asyncio
async def test_rag_query_where_llm_says_not_found_in_context_falls_back_to_general(monkeypatch) -> None:
    """The real-world case this feature targets: retrieval DOES find
    chunks (e.g. a leave-policy document is ingested), but they don't
    actually cover the asked question (e.g. work-from-home leave isn't
    in that document) — the grounded-answer LLM call correctly emits the
    NOT_FOUND_IN_CONTEXT sentinel, and the graph should route to a
    disclosed general-knowledge answer rather than a bare refusal."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr("app.graph.workflow.get_reranker", lambda: _FakeReranker())

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten wfh leave question",
            "NOT_FOUND_IN_CONTEXT": "NOT_FOUND_IN_CONTEXT",
        },
        default="Most companies allow 2-3 WFH days per week.",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many work-from-home leave days can I take?"))

    assert result["was_answerable"] is False
    assert result["verified"] is None
    assert result["citations"] == []
    assert "isn't covered by our available documents" in result["response"]
    assert "Most companies allow 2-3 WFH days per week." in result["response"]


@pytest.mark.asyncio
async def test_reranking_is_skipped_and_the_model_never_loads_when_disabled(monkeypatch) -> None:
    """`RERANK_ENABLED=False` (the default — see app/core/config.py) must
    skip the cross-encoder pass entirely, not just discard its output:
    loading BGE-Reranker-v2-M3 is the expensive part, so a disabled
    reranker that still gets constructed would save nothing."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(get_settings(), "RERANK_ENABLED", False)

    def _boom():
        raise AssertionError("reranker should not be loaded when RERANK_ENABLED is False")

    monkeypatch.setattr("app.graph.workflow.get_reranker", _boom)

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten query",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nGrounded.",
        },
        default="Employees get 20 days of leave. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert result["verified"] is True
    assert len(result["chunks"]) == 1


@pytest.mark.asyncio
async def test_reranking_still_runs_when_explicitly_enabled(monkeypatch) -> None:
    """The switch is a default, not a removal — flipping RERANK_ENABLED
    back on must restore the Stage-2 pass."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(get_settings(), "RERANK_ENABLED", True)

    loaded: list[bool] = []

    def _tracked_reranker():
        loaded.append(True)
        return _FakeReranker()

    monkeypatch.setattr("app.graph.workflow.get_reranker", _tracked_reranker)

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "rewritten query",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nGrounded.",
        },
        default="Employees get 20 days of leave. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many leave days do employees get?"))

    assert loaded == [True]
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_greeting_gets_a_conversational_reply_instead_of_retrieval() -> None:
    """"Hi" is neither a document question nor something to refuse.
    Running retrieval on it costs an embedding pass and a reranker pass
    to find nothing; refusing it makes the bot feel broken. It should get
    a short conversational reply and skip retrieval entirely."""
    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "SMALL_TALK",
            "greeting or pleasantry": "Hello! Ask me anything about our company policies.",
        }
    )
    graph = build_graph(llm_client=llm, weaviate_client=None)

    result = await graph.ainvoke(initial_state("Hi"))

    assert result["classification"] == "SMALL_TALK"
    assert result["answer_source"] == "small_talk"
    assert result["response"] == "Hello! Ask me anything about our company policies."
    assert result["chunks"] == []
    assert result["citations"] == []
    # Classification + the greeting reply, and nothing else — no rewrite,
    # no generation, no verification.
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_a_greeting_that_also_asks_something_still_goes_through_rag(monkeypatch) -> None:
    """The SMALL_TALK shortcut must not swallow real questions that
    happen to open with a greeting."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(get_settings(), "RERANK_ENABLED", False)

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "sick leave entitlement",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nGrounded.",
        },
        default="Employees get 7 days of sick leave. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("hi, how many sick leaves do I get?"))

    assert result["classification"] == "RAG_REQUIRED"
    assert result["answer_source"] == "rag"
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_a_non_english_question_retrieves_twice_and_merges(monkeypatch) -> None:
    """A Hindi question embeds into the same BGE-M3 space as an English
    document, but doesn't reliably surface the same chunks an English
    query would — so retrieval runs a second time in English and the two
    ranked lists are merged (app/rag/query_translation.py)."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(get_settings(), "RERANK_ENABLED", False)

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "कितनी सिक लीव मिलती है?",
            "English query:": "How many days of sick leave per year?",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nGrounded.",
        },
        default="7 दिन। [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("कितनी सिक लीव मिलती है?"))

    assert result["english_query"] == "How many days of sick leave per year?"
    queries = weaviate_client.collections.get("x").query.queries
    assert queries == ["कितनी सिक लीव मिलती है?", "How many days of sick leave per year?"]
    # Both passes found the same single chunk — the merge must dedupe it,
    # not hand generation the same text twice.
    assert len(result["chunks"]) == 1


@pytest.mark.asyncio
async def test_an_english_question_retrieves_only_once(monkeypatch) -> None:
    """No second pass for a question already in English — it would search
    with identical text and find identical chunks."""
    monkeypatch.setattr("app.graph.workflow.get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(get_settings(), "RERANK_ENABLED", False)

    llm = _ScriptedLLMClient(
        {
            "GENERAL, SMALL_TALK, or RAG_REQUIRED": "RAG_REQUIRED",
            "Rewritten query:": "sick leave entitlement per year",
            "English query:": "ALREADY_ENGLISH",
            "SUPPORTED or UNSUPPORTED": "SUPPORTED\nGrounded.",
        },
        default="7 days. [Source 1]",
    )
    weaviate_client = _FakeWeaviateClient(objects=[_fake_object(0, 0.9)])
    graph = build_graph(llm_client=llm, weaviate_client=weaviate_client)

    result = await graph.ainvoke(initial_state("how many sick leaves per year?"))

    assert result["english_query"] is None
    assert weaviate_client.collections.get("x").query.queries == ["sick leave entitlement per year"]
