import pandas as pd
import numpy as np

from app.services.dataset_models import detect_multivariate_outliers

def basic_shape(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def dtype_summary(df: pd.DataFrame) -> list:
    rows = []
    for col in df.columns:
        series = df[col]
        raw_dtype = str(series.dtype)
        if pd.api.types.is_bool_dtype(series):
            inferred = "Boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred = "Datetime"
        elif pd.api.types.is_numeric_dtype(series):
            inferred = "Numerical"
        else:
            coerced_numeric = pd.to_numeric(series, errors="coerce")
            pct_numeric = coerced_numeric.notna().mean()
            if pct_numeric > 0.9:
                inferred = "Numerical (hidden in text)"
            else:
                coerced_dt = pd.to_datetime(series, errors="coerce")
                pct_dt = coerced_dt.notna().mean()
                inferred = "Datetime (hidden in text)" if pct_dt > 0.9 else "Categorical"
        rows.append({"column": col, "raw_dtype": raw_dtype, "inferred_type": inferred})
    return rows


def missingness(df: pd.DataFrame, threshold: float = 40.0) -> list:
    pct = (df.isna().mean() * 100).round(2)
    out = pct.reset_index()
    out.columns = ["column", "missing_pct"]
    out["flagged"] = out["missing_pct"] > threshold
    out = out.sort_values("missing_pct", ascending=False)
    return out.to_dict(orient="records")


def cardinality(df: pd.DataFrame, high_card_threshold: float = 0.9) -> list:
    rows = []
    n = len(df)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique(dropna=True)
            ratio = nunique / n if n else 0
            rows.append({
                "column": col,
                "unique_values": int(nunique),
                "unique_ratio": round(ratio, 3),
                "flagged_high_cardinality": bool(ratio > high_card_threshold),
            })
    return rows


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method=method).round(3)


def flagged_correlation_pairs(corr: pd.DataFrame, r_threshold: float = 0.85) -> list:
    pairs = []
    if corr.empty:
        return pairs
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > r_threshold:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "r": round(float(r), 3)})
    return pairs


def outlier_report(df: pd.DataFrame) -> list:
    rows = []
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append({
            "column": col,
            "outlier_count": int(outliers.shape[0]),
            "outlier_pct": round(100 * outliers.shape[0] / len(series), 2),
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
        })
    return rows


def full_diagnosis(df: pd.DataFrame) -> dict:
    corr = correlation_matrix(df)
    return {
        "shape": basic_shape(df),
        "dtypes": dtype_summary(df),
        "missingness": missingness(df),
        "cardinality": cardinality(df),
        "correlated_pairs": flagged_correlation_pairs(corr),
        "outliers": outlier_report(df),
        "multivariate_outliers": detect_multivariate_outliers(df),
    }
