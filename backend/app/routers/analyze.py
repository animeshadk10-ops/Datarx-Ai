from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services import session_store, stats_engine
from app.services.llm_pipeline import run_analysis_pipeline, _check_flagged

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

    errors = result.get("errors", [])
    
    has_flagged = _check_flagged(diagnosis)
        
    # Check if the pipeline failed to generate expected recommendations due to an LLM error (like invalid API key)
    # OR if it failed completely to get any semantic types.
    if errors:
        if (has_flagged and not result.get("recommendations")) or (not result.get("semantic_types")):
            # If there are errors (e.g. 403 Forbidden, API key invalid, Rate Limit) and it returned empty when we expected data, 
            # it's a silent failure. We should raise an exception so the frontend displays the error.
            error_msg = errors[-1] # The last error is usually the fatal one
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise HTTPException(status_code=502, detail="AI Analysis failed: Gemini Free Tier API rate limit exceeded. Please wait a minute and try again.")
            raise HTTPException(status_code=502, detail=f"AI Analysis failed: {error_msg}. Please check your API key and network connection.")

    return AnalyzeResponse(
        semantic_types=result.get("semantic_types", []),
        recommendations=result.get("recommendations", []),
        target_analysis=result.get("target_analysis")
    )
