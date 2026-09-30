from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Request body for creating a document (what the client sends).""" # ---- Python Docstring: It is a multi-line string used to document code ------

    # Trim surrounding whitespace before validating, so "   " fails min_length.  # ----standard code comment, Visible only for human eyes -------
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=3, max_length=100, examples=["Employee Handbook"])
    department: str = Field(min_length=2, max_length=50, examples=["HR"])


class DocumentResponse(BaseModel):
    """A document as returned by the API (what the server sends back)."""

    id: int
    name: str
    department: str
    status: str


class DocumentListResponse(BaseModel):
    """A collection of documents."""

    documents: list[DocumentResponse]
