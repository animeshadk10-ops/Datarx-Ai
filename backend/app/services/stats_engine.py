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
        # Exclude infinite values from being counted as standard outliers
        series = numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()
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


def duplicate_row_report(df: pd.DataFrame, id_columns: list = None) -> dict:
    subset = df.columns.difference(id_columns) if id_columns else df.columns
    if len(subset) == 0 or df.empty:
        return {"duplicate_row_count": 0, "duplicate_row_pct": 0.0, "duplicate_row_indices": []}
    dup_mask = df.duplicated(subset=subset, keep=False)
    dup_indices = df.index[dup_mask].tolist()
    count = len(dup_indices)
    pct = round(100 * count / len(df), 2)
    return {
        "duplicate_row_count": count,
        "duplicate_row_pct": pct,
        "duplicate_row_indices": dup_indices
    }


def duplicate_column_report(df: pd.DataFrame) -> dict:
    if df.empty or df.shape[1] < 2:
        return {"duplicate_column_pairs": []}
        
    pairs = []
    hashes = {col: pd.util.hash_pandas_object(df[col], index=False).sum() for col in df.columns}
    cols = list(df.columns)
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            if hashes[col_a] == hashes[col_b]:
                if df[col_a].equals(df[col_b]):
                    pairs.append({"col_a": col_a, "col_b": col_b})
                    
    return {"duplicate_column_pairs": pairs}


def constant_feature_report(df: pd.DataFrame) -> list:
    rows = []
    if df.empty:
        return rows
    
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique == 1:
            has_missing = df[col].isna().any()
            val = df[col].dropna().iloc[0]
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            elif isinstance(val, (pd.Timestamp, pd.Timedelta)):
                val = str(val)
            rows.append({
                "column": col,
                "fully_constant": bool(not has_missing),
                "constant_excluding_missing": bool(has_missing),
                "constant_value": val
            })
    return rows


def infinite_value_report(df: pd.DataFrame) -> list:
    rows = []
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return rows
        
    for col in numeric_df.columns:
        inf_count = int(np.isinf(numeric_df[col]).sum())
        if inf_count > 0:
            rows.append({
                "column": col,
                "infinite_count": inf_count
            })
    return rows


def rare_category_report(df: pd.DataFrame, rare_threshold_pct: float = 1.0) -> list:
    rows = []
    if df.empty:
        return rows
        
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            vc = df[col].value_counts(normalize=True) * 100
            rare = vc[vc < rare_threshold_pct]
            if not rare.empty:
                rare_list = [{"value": str(k), "pct": round(float(v), 2)} for k, v in rare.items()]
                rows.append({
                    "column": col,
                    "rare_categories": rare_list,
                    "rare_category_count": len(rare_list)
                })
    return rows

def generate_chart_data(df: pd.DataFrame, corr: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_charts = []
    
    for col in numeric_df.columns:
        series = numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Calculate Histogram (20 bins)
        try:
            n_unique = max(1, series.nunique())
            bins = min(20, n_unique) if n_unique < 20 else 20
            counts, bin_edges = np.histogram(series, bins=bins)
            histogram = [
                {
                    "bin_start": float(bin_edges[i]), 
                    "bin_end": float(bin_edges[i+1]), 
                    "count": int(counts[i])
                } 
                for i in range(len(counts))
            ]
        except Exception:
            histogram = []
        
        numeric_charts.append({
            "column": col,
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "histogram": histogram
        })
        
    categorical_charts = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            vc = df[col].value_counts().head(10)
            value_counts = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
            if value_counts:
                categorical_charts.append({
                    "column": col,
                    "value_counts": value_counts
                })
                
    corr_matrix_data = []
    if not corr.empty:
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r):
                    corr_matrix_data.append({"col_a": cols[i], "col_b": cols[j], "r": round(float(r), 3)})
                    
    scatter_samples = []
    if not corr.empty:
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) > 0.6:
                    col_a = cols[i]
                    col_b = cols[j]
                    pair_df = df[[col_a, col_b]].dropna()
                    if len(pair_df) > 300:
                        pair_df = pair_df.sample(300, random_state=42)
                    sample_data = [{"x": float(row[col_a]), "y": float(row[col_b])} for _, row in pair_df.iterrows()]
                    scatter_samples.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "r": round(float(r), 3),
                        "sample": sample_data
                    })
    
    return {
        "numeric": numeric_charts,
        "categorical": categorical_charts,
        "correlation_matrix": corr_matrix_data,
        "scatter_samples": scatter_samples
    }


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
        "duplicate_rows": duplicate_row_report(df),
        "duplicate_columns": duplicate_column_report(df),
        "constant_features": constant_feature_report(df),
        "infinite_values": infinite_value_report(df),
        "rare_categories": rare_category_report(df),
        "chart_data": generate_chart_data(df, corr)
    }
