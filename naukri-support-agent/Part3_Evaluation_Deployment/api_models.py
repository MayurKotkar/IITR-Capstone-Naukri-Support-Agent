"""
Task 11: Pydantic Request and Response Models for FastAPI Deployment.

This module defines the strict Pydantic schemas for the Naukri.com Support Agent API:
1. AskRequest / AskResponse: Query submission, multi-turn session ID, RAG/lookup answers,
   source documents, grounding flags, escalation scores, and trace metadata.
2. AddDocRequest / AddDocResponse: Dynamic document ingestion into ChromaDB knowledge base.
3. HealthResponse: System status and offline MOCK_LLM verification.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    HTTP Request payload for POST /ask.
    Supports single-turn and multi-turn (via session_id) inquiries.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="User question, policy inquiry, or job application status lookup",
        examples=["What is the notice period policy?", "What is the status of application APP-005?"]
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier for multi-turn conversation memory",
        examples=["session_user_42", "ws_sess_abc123"]
    )


class AskResponse(BaseModel):
    """
    HTTP Response payload for POST /ask.
    Conforms to the multi-agent AgentResponse schema with observability extensions.
    """
    query: str = Field(..., description="Echoed input query (sanitized if PII detected)")
    answer: str = Field(..., description="Grounded final answer composed by the multi-agent team")
    source_documents: list[str] = Field(
        default_factory=list,
        description="List of knowledge base doc IDs or dataset record IDs cited"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Retrieval confidence or cosine similarity score"
    )
    is_grounded: bool = Field(
        ...,
        description="Whether the answer is strictly grounded in retrieved evidence"
    )
    escalation_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Calculated escalation score for job application records (Task 6)"
    )
    trace_id: str = Field(..., description="Unique UUID trace ID for ELK-style logging")
    response_time_ms: float = Field(..., ge=0.0, description="End-to-end request latency in milliseconds")
    session_id: Optional[str] = Field(None, description="Session ID if conversational memory was utilized")
    guardrail_flags: Optional[dict[str, Any]] = Field(
        None,
        description="Status flags from input/output guardrail checks"
    )


class AddDocRequest(BaseModel):
    """
    HTTP Request payload for POST /add-document.
    Submits a new policy document for dynamic chunking, embedding, and ChromaDB upsert.
    """
    doc_id: str = Field(
        ...,
        min_length=1,
        description="Unique document identifier (e.g., 'kb_013', 'DOC-SABBATICAL-01')",
        examples=["kb_013", "DOC-LEAVE-02"]
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="Policy topic or category name",
        examples=["sabbatical-leave policy", "workplace safety guidelines"]
    )
    content: str = Field(
        ...,
        min_length=10,
        description="Full text content of the policy document (minimum 10 characters)",
        examples=["Employees with at least five years of service may apply for unpaid sabbatical leave of up to 12 months."]
    )


class AddDocResponse(BaseModel):
    """
    HTTP Response payload for POST /add-document.
    Confirms dynamic vector database indexing and chunk counts.
    """
    status: str = Field("success", description="Indexing status ('success' or 'error')")
    doc_id: str = Field(..., description="ID of the ingested document")
    chunks_indexed: int = Field(..., ge=0, description="Number of sentence chunks embedded and upserted")
    topic: str = Field(..., description="Document topic category")
    trace_id: str = Field(..., description="Unique UUID trace ID for audit logging")
    response_time_ms: float = Field(..., ge=0.0, description="Indexing latency in milliseconds")
    message: str = Field(..., description="Human-readable confirmation message")


class HealthResponse(BaseModel):
    """
    HTTP Response payload for GET /health.
    """
    status: str = Field("ok", description="Service health status")
    mode: str = Field("MOCK_LLM", description="Execution mode (MOCK_LLM: zero paid API keys, zero network)")
    version: str = Field("1.0.0", description="API version")
    timestamp: str = Field(..., description="Current ISO-8601 UTC timestamp")

