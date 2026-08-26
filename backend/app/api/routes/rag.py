"""RAG query route.

Contains no business logic — requests are delegated to RagQueryService.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_query_service
from app.rag.generation.context_builder import Citation
from app.schemas.rag import CitationResponse, RagAskRequest, RagAskResponse
from app.services.rag_service import RagQueryService

router = APIRouter(tags=["rag"])


def _to_citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        label=citation.label,
        document_name=citation.document_name,
        start_page=citation.start_page,
        end_page=citation.end_page,
        chunk_id=citation.chunk_id,
    )


@router.post("/rag/ask", response_model=RagAskResponse)
async def rag_ask(
    request: RagAskRequest,
    rag_service: RagQueryService = Depends(get_rag_query_service),
) -> RagAskResponse:
    """Retrieve context and generate a grounded answer to `question`."""
    result = await rag_service.ask(
        request.question,
        limit=request.limit,
        alpha=request.alpha,
        do_rerank=request.rerank,
        top_k=request.top_k,
    )
    return RagAskResponse(
        answer=result.answer,
        citations=[_to_citation_response(c) for c in result.citations],
        was_answerable=result.was_answerable,
    )
