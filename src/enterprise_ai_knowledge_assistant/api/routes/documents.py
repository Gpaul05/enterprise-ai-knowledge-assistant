from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from enterprise_ai_knowledge_assistant.models.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
)
from enterprise_ai_knowledge_assistant.services.document_service import (
    DocumentService,
    get_document_service,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

# Reusable type: "a DocumentService, provided by FastAPI's dependency injection".
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    payload: DocumentCreate,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """
    Create a new document.

    - **name**: 3–100 characters (surrounding spaces are trimmed)
    - **department**: 2–50 characters

    The server assigns the `id` and sets `status` to `available`.
    Returns **201 Created** with the new document, or **422** if validation fails.
    """
    return service.create_document(payload)


@router.get("/", response_model=DocumentListResponse)
async def get_documents(
    service: DocumentServiceDep,
    department: str | None = None,
) -> DocumentListResponse:
    """
    List all documents.

    - **department** (optional): return only documents from this department.
      Matching is case-insensitive, so `hr` matches `HR`.

    Returns an empty list if no documents match.
    """
    return DocumentListResponse(documents=service.get_documents(department))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Document not found"}},
)
async def get_document(
    document_id: int,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """
    Get a single document by its ID.

    - **document_id**: must be an integer, otherwise **422** is returned.

    Returns **404 Not Found** if no document has this ID.
    """
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {document_id} not found",
        )
    return document
