from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services import session_store, stats_engine
from app.services.llm_pipeline import run_analysis_pipeline

router = APIRouter(tags=["Analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_data(request: AnalyzeRequest):
    df = session_store.get_session(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    diagnosis = stats_engine.full_diagnosis(df)
    result = await run_analysis_pipeline(
        df, 
        diagnosis, 
        target_column=request.target_column, 
        target_purpose=request.target_purpose
    )

    return AnalyzeResponse(
        semantic_types=result.get("semantic_types", []),
        recommendations=result.get("recommendations", []),
        target_analysis=result.get("target_analysis")
    )
