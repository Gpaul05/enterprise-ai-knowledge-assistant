import pytest
from fastapi.testclient import TestClient

from enterprise_ai_knowledge_assistant.api.routes import chat as chat_routes
from enterprise_ai_knowledge_assistant.services.llm_service import LLMServiceError

CHAT_URL = "/api/v1/chat/"


def test_chat_returns_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate_response(prompt: str) -> str:
        return f"Echo: {prompt}"

    monkeypatch.setattr(chat_routes.llm_service, "generate_response", fake_generate_response)

    response = client.post(CHAT_URL, json={"message": "Hi"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Echo: Hi"}


def test_chat_returns_502_when_llm_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_generate_response(prompt: str) -> str:
        raise LLMServiceError("The language model request failed")

    monkeypatch.setattr(chat_routes.llm_service, "generate_response", failing_generate_response)

    response = client.post(CHAT_URL, json={"message": "Hi"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The language model request failed"}


def test_chat_with_empty_message_returns_422(client: TestClient) -> None:
    response = client.post(CHAT_URL, json={"message": ""})

    assert response.status_code == 422
