"""Context construction + grounded answer generation (plan.md sections 20-25).

Deliberately NOT implemented yet, per the project's phase-by-phase
scope: query classification (section 14), query rewriting (section
15), and answer verification (section 26). This module assumes the
caller already decided RAG is needed and already has a final set of
chunks to answer from (i.e., it's the last two boxes in plan.md's
pipeline diagram — context builder + Groq — not the whole thing).
"""

from app.rag.generation.answer_generator import RAGAnswer, generate_answer
from app.rag.generation.context_builder import Citation, build_citations, build_context

__all__ = ["RAGAnswer", "generate_answer", "Citation", "build_citations", "build_context"]
