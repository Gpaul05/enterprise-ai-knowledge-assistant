import logging

from fastapi import FastAPI

from enterprise_ai_knowledge_assistant.api.routes.documents import router as document_router
from enterprise_ai_knowledge_assistant.core.config import settings
from enterprise_ai_knowledge_assistant.api.routes.chat import (
    router as chat_router,
)

# Show our INFO logs (e.g. the tool-calling flow) in the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)
# httpx logs every request URL at INFO; keep the console focused on our flow.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(
    document_router,
    prefix="/api/v1",
)

app.include_router(
    chat_router,
    prefix="/api/v1",
)