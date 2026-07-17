from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import ActionRequest, ActionResponse
from app.services import session_store
from app.services.executor import execute_action

router = APIRouter(tags=["Actions"])


@router.post("/apply-action", response_model=ActionResponse)
async def apply_action(request: ActionRequest):
    df = session_store.get_session(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.action.value != "drop_column" and request.column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.column}' not found in dataset",
        )
    if request.action.value == "drop_column" and request.column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.column}' not found in dataset",
        )

    # Validate that numeric actions are only applied to numeric columns
    numeric_actions = ["impute_median", "clip_outliers", "log_transform"]
    if request.action.value in numeric_actions:
        import pandas as pd
        if not pd.api.types.is_numeric_dtype(df[request.column]):
            raise HTTPException(
                status_code=400,
                detail=f"Action '{request.action.value}' cannot be applied to non-numeric column '{request.column}'."
            )

    modified_df, before_stats, after_stats = execute_action(
        df, request.column, request.action
    )
    session_store.update_session(request.session_id, modified_df)

    from app.services import stats_engine
    full_diagnosis = stats_engine.full_diagnosis(modified_df)

    return ActionResponse(
        success=True,
        column=request.column,
        action=request.action.value,
        before=before_stats,
        after=after_stats,
        full_diagnosis=full_diagnosis,
    )
