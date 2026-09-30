from fastapi import APIRouter, HTTPException, status

from enterprise_ai_knowledge_assistant.models.chat import (
    ChatRequest,
    ChatResponse,
)
from enterprise_ai_knowledge_assistant.services.llm_service import (
    LLMService,
    LLMServiceError,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


llm_service = LLMService()


@router.post(
    "/",
    response_model=ChatResponse,
    responses={status.HTTP_502_BAD_GATEWAY: {"description": "The language model failed"}},
)
async def chat(request: ChatRequest):
    try:
        answer = await llm_service.generate_response(
            request.message
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChatResponse(answer=answer)
