from fastapi.testclient import TestClient

DOCUMENTS_URL = "/api/v1/documents/"


def create_document(client: TestClient, name: str, department: str) -> dict:
    response = client.post(DOCUMENTS_URL, json={"name": name, "department": department})
    assert response.status_code == 201
    return response.json()


def test_create_document_returns_201(client: TestClient) -> None:
    response = client.post(
        DOCUMENTS_URL,
        json={"name": "Employee Handbook", "department": "HR"},
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": 1,
        "name": "Employee Handbook",
        "department": "HR",
        "status": "available",
    }


def test_create_document_generates_sequential_ids(client: TestClient) -> None:
    first = create_document(client, "Employee Handbook", "HR")
    second = create_document(client, "Security Policy", "IT")

    assert first["id"] == 1
    assert second["id"] == 2


def test_create_document_with_short_name_returns_422(client: TestClient) -> None:
    response = client.post(DOCUMENTS_URL, json={"name": "ab", "department": "HR"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "name"]


def test_create_document_with_missing_field_returns_422(client: TestClient) -> None:
    response = client.post(DOCUMENTS_URL, json={"name": "Employee Handbook"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "department"]


def test_get_documents_when_empty(client: TestClient) -> None:
    response = client.get(DOCUMENTS_URL)

    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_get_documents_returns_created_documents(client: TestClient) -> None:
    create_document(client, "Employee Handbook", "HR")
    create_document(client, "Security Policy", "IT")

    response = client.get(DOCUMENTS_URL)

    assert response.status_code == 200
    names = [doc["name"] for doc in response.json()["documents"]]
    assert names == ["Employee Handbook", "Security Policy"]


def test_get_documents_filtered_by_department(client: TestClient) -> None:
    create_document(client, "Employee Handbook", "HR")
    create_document(client, "Security Policy", "IT")
    create_document(client, "Leave Policy", "HR")

    response = client.get(DOCUMENTS_URL, params={"department": "hr"})

    assert response.status_code == 200
    names = [doc["name"] for doc in response.json()["documents"]]
    assert names == ["Employee Handbook", "Leave Policy"]


def test_get_document_by_id(client: TestClient) -> None:
    created = create_document(client, "Employee Handbook", "HR")

    response = client.get(f"{DOCUMENTS_URL}{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_nonexistent_document_returns_404(client: TestClient) -> None:
    response = client.get(f"{DOCUMENTS_URL}999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document with id 999 not found"}


def test_get_document_with_invalid_id_returns_422(client: TestClient) -> None:
    response = client.get(f"{DOCUMENTS_URL}abc")

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["path", "document_id"]
