from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.enums import RecommendedAction


def _column_stats(df: pd.DataFrame, column: str) -> dict:
    """Compute summary stats for a single column (handles both numeric and non-numeric)."""
    if column not in df.columns:
        return {"missing_count": None, "missing_pct": None}

    series = df[column]
    n = len(series)
    missing = int(series.isna().sum())
    stats: dict = {
        "missing_count": missing,
        "missing_pct": round(100 * missing / n, 2) if n > 0 else 0.0,
    }
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        stats.update({
            "mean": round(float(desc.get("mean", 0)), 4),
            "std": round(float(desc.get("std", 0)), 4),
            "min": round(float(desc.get("min", 0)), 4),
            "max": round(float(desc.get("max", 0)), 4),
        })
    return stats


def execute_action(
    df: pd.DataFrame, column: str, action: RecommendedAction
) -> tuple[pd.DataFrame, dict, dict]:
    """Execute a cleaning action on *column* and return (modified_df, before_stats, after_stats)."""

    before = _column_stats(df, column)
    result_df = df.copy()

    if action == RecommendedAction.impute_median:
        if column in result_df.columns and pd.api.types.is_numeric_dtype(result_df[column]):
            median_val = result_df[column].median()
            result_df[column] = result_df[column].fillna(median_val)

    elif action == RecommendedAction.impute_mode:
        if column in result_df.columns:
            mode_vals = result_df[column].mode()
            if not mode_vals.empty:
                result_df[column] = result_df[column].fillna(mode_vals.iloc[0])

    elif action == RecommendedAction.drop_column:
        if column in result_df.columns:
            result_df = result_df.drop(columns=[column])
            after: dict = {"missing_count": None, "missing_pct": None}
            return result_df, before, after

    elif action == RecommendedAction.clip_outliers:
        if column in result_df.columns and pd.api.types.is_numeric_dtype(result_df[column]):
            lower = result_df[column].quantile(0.01)
            upper = result_df[column].quantile(0.99)
            result_df[column] = result_df[column].clip(lower=lower, upper=upper)

    elif action == RecommendedAction.log_transform:
        if column in result_df.columns and pd.api.types.is_numeric_dtype(result_df[column]):
            # Handle non-positive values by shifting the column so the minimum is 1
            min_val = result_df[column].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                result_df[column] = np.log1p(result_df[column] + shift)
            else:
                result_df[column] = np.log1p(result_df[column])

    elif action == RecommendedAction.merge_categories:
        if column in result_df.columns:
            # 1. Case-normalization (lowercase + strip whitespace)
            if pd.api.types.is_string_dtype(result_df[column]):
                result_df[column] = result_df[column].str.lower().str.strip()
            
            # 2. Rare-category consolidation into "Other"
            if not pd.api.types.is_numeric_dtype(result_df[column]):
                vc = result_df[column].value_counts(normalize=True) * 100
                rare_categories = vc[vc < 1.0].index
                if len(rare_categories) > 0:
                    result_df.loc[result_df[column].isin(rare_categories), column] = "Other"

    elif action == RecommendedAction.none:
        pass  # no-op

    after = _column_stats(result_df, column)
    return result_df, before, after
