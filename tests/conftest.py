import os
from collections.abc import Iterator

# Unit tests must never call the real OpenRouter API. Set a dummy key before
# the app is imported (load_dotenv does not override an existing variable).
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

from enterprise_ai_knowledge_assistant.main import app
from enterprise_ai_knowledge_assistant.services.document_service import (
    DocumentService,
    get_document_service,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Give each test its own empty in-memory store so tests stay isolated.
    service = DocumentService()
    app.dependency_overrides[get_document_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
