from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.enums import SemanticType, RecommendedAction


class UploadResponse(BaseModel):
    session_id: str
    diagnosis: dict


class ColumnSemantic(BaseModel):
    column: str
    semantic_type: SemanticType
    is_identifier: bool
    notes: str = ""


class ClassifyResponse(BaseModel):
    columns: list[ColumnSemantic]


class Recommendation(BaseModel):
    column: str
    issue: str
    severity: str
    recommended_action: RecommendedAction
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]


class ActionRequest(BaseModel):
    session_id: str
    column: str
    action: RecommendedAction
    justification: str = ""


class ActionResponse(BaseModel):
    success: bool
    column: str
    action: str
    before: dict
    after: dict
    full_diagnosis: dict


class AnalyzeRequest(BaseModel):
    session_id: str
    target_column: str | None = None
    target_purpose: str | None = None


class AnalyzeResponse(BaseModel):
    semantic_types: list[ColumnSemantic]
    recommendations: list[Recommendation]
    target_analysis: dict | None = None

class ActionSummary(BaseModel):
    column: str
    action: str
    justification: str
    before: dict
    after: dict

class SummaryResponse(BaseModel):
    original_shape: dict
    final_shape: dict
    actions_applied: list[ActionSummary]
    export_ready: bool
