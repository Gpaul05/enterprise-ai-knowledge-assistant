from enterprise_ai_knowledge_assistant.models.document import (
    DocumentCreate,
    DocumentResponse,
)


class DocumentService:
    """Business logic for documents, using in-memory storage for now."""

    def __init__(self) -> None:
        self._documents: dict[int, DocumentResponse] = {}
        self._next_id: int = 1

    def create_document(self, document: DocumentCreate) -> DocumentResponse:
        new_document = DocumentResponse(
            id=self._next_id,
            name=document.name,
            department=document.department,
            status="available",
        )
        self._documents[new_document.id] = new_document
        self._next_id += 1
        return new_document

    def get_documents(self, department: str | None = None) -> list[DocumentResponse]:
        documents = list(self._documents.values())
        if department is not None:
            # Case-insensitive, so ?department=hr matches "HR".
            documents = [
                doc for doc in documents if doc.department.lower() == department.lower()
            ]
        return documents

    def get_document(self, document_id: int) -> DocumentResponse | None:
        return self._documents.get(document_id)


# One shared instance: in-memory data must survive between requests.
document_service = DocumentService()


def get_document_service() -> DocumentService:
    """FastAPI dependency that provides the DocumentService."""
    return document_service
