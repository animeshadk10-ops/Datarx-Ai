from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from app.core.config import GOOGLE_API_KEY, MODEL_NAME
from app.models.enums import RecommendedAction, SemanticType
from app.models.schemas import ColumnSemantic, Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load prompt templates once at module level
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CLASSIFY_SYSTEM_PROMPT = (_PROMPTS_DIR / "classify_system.txt").read_text(encoding="utf-8")
REASON_SYSTEM_PROMPT = (_PROMPTS_DIR / "reason_system.txt").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# LLM instance (re-used across calls)
# ---------------------------------------------------------------------------
_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True,
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    diagnosis: dict
    df_sample_payload: str  # JSON string sent to classify node
    has_flagged_columns: bool
    semantic_types: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    target_column: str | None
    target_purpose: str | None
    target_analysis: dict | None
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_classify_payload(df: pd.DataFrame, diagnosis: dict) -> str:
    """Build the user-message payload for the classification node."""
    dtype_map: dict[str, str] = {
        entry["column"]: entry["inferred_type"] for entry in diagnosis.get("dtypes", [])
    }
    items = []
    for col in df.columns:
        sample = df[col].dropna()
        sample_values = sample.head(8).tolist() if len(sample) >= 5 else sample.tolist()
        # Convert numpy types to native Python for JSON serialisation
        sample_values = [
            v.item() if hasattr(v, "item") else v for v in sample_values
        ]
        items.append({
            "column": col,
            "dtype": dtype_map.get(col, str(df[col].dtype)),
            "sample_values": sample_values,
        })
    return json.dumps(items, indent=2, default=str)


def _extract_json(text: str) -> Any:
    """Strip markdown fences and parse JSON from LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` or ``` ... ```
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _check_flagged(diagnosis: dict) -> bool:
    """Return True if any column has a data-quality flag."""
    for m in diagnosis.get("missingness", []):
        if m.get("missing_pct", 0) > 0:
            return True
    for o in diagnosis.get("outliers", []):
        if o.get("outlier_count", 0) > 0:
            return True
    if diagnosis.get("correlated_pairs"):
        return True
    return False


def _build_reason_payload(diagnosis: dict, semantic_types: list[dict]) -> str:
    """Build the user-message payload for the reasoning node."""
    # Filter missingness / outlier info to only flagged columns
    flagged_missing = [
        m for m in diagnosis.get("missingness", []) if m.get("missing_pct", 0) > 0
    ]
    flagged_outliers = [
        o for o in diagnosis.get("outliers", []) if o.get("outlier_count", 0) > 0
    ]
    payload = {
        "missingness": flagged_missing,
        "outliers": flagged_outliers,
        "correlated_pairs": diagnosis.get("correlated_pairs", []),
        "semantic_types": semantic_types,
    }
    target_purpose = diagnosis.get("target_purpose")
    if target_purpose:
        payload["target_purpose"] = target_purpose
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------
async def classify_node(state: PipelineState) -> dict:
    """Stage 2: semantic classification.
    
    Integration point for the local semantic classifier:
    1. Run classify_column_local() for every column first  (zero LLM cost)
    2. Only send columns where needs_escalation() is True to Gemini
    3. Merge local + escalated results into one final per-column semantic_type list
    """
    errors: list[str] = list(state.get("errors", []))

    # ── Step 1: Try local classifier first ──────────────────────────────
    local_results: list[dict] = []
    escalation_columns: list[str] = []
    df = state.get("df_raw")

    try:
        from app.services.semantic_classifier import classify_column_local, needs_escalation
        if df is not None:
            for col in df.columns:
                result = classify_column_local(df[col], col)
                if needs_escalation(result):
                    escalation_columns.append(col)
                else:
                    # Local model is confident — use its prediction directly
                    local_results.append({
                        "column": col,
                        "semantic_type": result["semantic_type"],
                        "is_identifier": result["semantic_type"] == "id",
                        "notes": f"local_model (raw_conf={result.get('raw_confidence', 0):.2f}, reported={result.get('reported_confidence', 0):.2f})",
                        "raw_confidence": result.get("raw_confidence"),
                        "reported_confidence": result.get("reported_confidence")
                    })
            logger.info(
                "Local classifier: %d confident, %d escalated to Gemini",
                len(local_results), len(escalation_columns),
            )
    except Exception as exc:
        # If the local model isn't trained yet or fails, fall back entirely to Gemini
        logger.warning("Local classifier unavailable, falling back to Gemini: %s", exc)
        escalation_columns = list(df.columns) if df is not None else []

    # ── Step 2: Escalate low-confidence columns to Gemini ───────────────
    gemini_results: list[dict] = []
    if escalation_columns:
        payload = state["df_sample_payload"]
        
        from app.services.embedding_retrieval import get_similar_examples
        examples_str = "\n\nHere are some similar examples from our knowledge base to help you:\n"
        examples_injected = False
        
        for col_name in escalation_columns:
            similar = get_similar_examples(col_name)
            for ex in similar:
                examples_str += f"- If you see a column like '{ex['column_name']}', it is usually semantic_type='{ex['semantic_type']}'. Reason: {ex['reasoning']}\n"
                examples_injected = True
                
        if examples_injected:
            payload += examples_str
            
        for attempt in range(2):
            try:
                messages = [
                    SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
                    HumanMessage(content=payload),
                ]
                if attempt == 1 and errors:
                    messages.append(
                        HumanMessage(
                            content=(
                                "Your previous response had a validation error:\n"
                                f"{errors[-1]}\n"
                                "Please fix it and return valid JSON."
                            )
                        )
                    )
                response = await _llm.ainvoke(messages)
                raw = _extract_json(response.content)
                print(f"DEBUG: Gemini raw response: {raw}")
                for item in raw:
                    cs = ColumnSemantic(**item)
                    d = cs.model_dump()
                    # Only keep escalated columns from Gemini results
                    if d["column"] in escalation_columns:
                        gemini_results.append(d)
                print(f"DEBUG: gemini_results after filtering: {gemini_results}")
                break
            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                errors.append(f"classify attempt {attempt + 1}: {exc!s}")
                logger.warning("classify_node attempt %d failed: %s", attempt + 1, exc)

    # ── Step 3: Merge local + Gemini results ────────────────────────────
    # Local results take priority for confident columns
    seen_cols = {r["column"] for r in local_results}
    for gr in gemini_results:
        if gr["column"] not in seen_cols:
            local_results.append(gr)
            seen_cols.add(gr["column"])

    print(f"DEBUG: classify_node returning {len(local_results)} semantic types. Errors: {errors}")
    return {"semantic_types": local_results, "errors": errors}



async def target_analysis_node(state: PipelineState) -> dict:
    """Stage 2.5: Deterministic Target Checks (if target provided)."""
    target_col = state.get("target_column")
    if not target_col:
        return {"target_analysis": None}

    from app.services.target_aware_checks import run_target_checks
    
    # We pass the original df (which isn't in state directly, but we only have a sample payload in state)
    # Actually, we need the full dataframe for deterministic checks!
    # Wait, llm_pipeline currently doesn't store the full df in state.
    # Let's add it to initial state.
    df = state.get("df_raw")
    if df is None:
        # If it's not in state, we can't do the checks.
        return {"target_analysis": None}

    analysis = run_target_checks(df, target_col, state.get("semantic_types", []))
    return {"target_analysis": analysis}


async def reason_node(state: PipelineState) -> dict:
    """Stage 3: reasoning / recommendations via Gemini (Self-Consistency)."""
    errors: list[str] = list(state.get("errors", []))
    payload = _build_reason_payload(state["diagnosis"], state.get("semantic_types", []))

    messages = [
        SystemMessage(content=REASON_SYSTEM_PROMPT),
        HumanMessage(content=payload),
    ]

    import asyncio
    from collections import defaultdict, Counter

    # We need to explicitly import ChatGoogleGenerativeAI and create a slightly higher temp LLM 
    # for the consistency check to be meaningful, but the user explicitly requested Temp 0.
    # We will use the existing _llm (Temp 0). If it returns identical results, the logic 
    # perfectly handles it. If Google's backend has slight jitter, we catch it.
    
    async def fetch_reasoning(attempt_idx: int):
        for retry in range(2):
            try:
                msgs = list(messages)
                if retry == 1 and errors:
                    msgs.append(
                        HumanMessage(content=f"Validation error previously:\n{errors[-1]}\nPlease fix it and return valid JSON.")
                    )
                response = await _llm.ainvoke(msgs)
                raw = _extract_json(response.content)
                validated = []
                for item in raw:
                    rec = Recommendation(**item)
                    validated.append(rec.model_dump())
                return validated
            except Exception as exc:
                errors.append(f"reason attempt {attempt_idx}-{retry}: {exc!s}")
        return []

    results = await asyncio.gather(*[fetch_reasoning(i) for i in range(3)])
    
    col_recs = defaultdict(list)
    for res_list in results:
        for rec in res_list:
            col_recs[rec["column"]].append(rec)
            
    final_recs = []
    for col, recs in col_recs.items():
        if not recs:
            continue
        
        actions = [r["recommended_action"] for r in recs]
        counter = Counter(actions)
        most_common_action, count = counter.most_common(1)[0]
        
        rep = next(r for r in recs if r["recommended_action"] == most_common_action)
        
        matching_conf = [r["confidence"] for r in recs if r["recommended_action"] == most_common_action]
        avg_conf = sum(matching_conf) / len(matching_conf) if matching_conf else rep["confidence"]
        rep["confidence"] = avg_conf
        
        # If less than 2 models agreed, it needs human review
        rep["needs_review"] = count < 2 or len(recs) < 2
        
        final_recs.append(rep)

    return {"recommendations": final_recs, "errors": errors}


def _should_reason(state: PipelineState) -> str:
    """Conditional edge: skip reasoning when no columns are flagged."""
    if state.get("has_flagged_columns"):
        return "reason"
    return "end"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def _build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("classify", classify_node)
    graph.add_node("target_analysis", target_analysis_node)
    graph.add_node("reason", reason_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "target_analysis")
    
    # After target analysis, we decide if we reason
    graph.add_conditional_edges(
        "target_analysis",
        _should_reason,
        {"reason": "reason", "end": END},
    )
    graph.add_edge("reason", END)
    return graph


_compiled_graph = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def run_analysis_pipeline(
    df: pd.DataFrame, 
    diagnosis: dict,
    target_column: str | None = None,
    target_purpose: str | None = None
) -> dict[str, Any]:
    """Run the full Stage 2 + Stage 2.5 + Stage 3 pipeline and return results."""
    # inject target_purpose into diagnosis so _build_reason_payload finds it
    diagnosis_for_payload = dict(diagnosis)
    if target_purpose:
        diagnosis_for_payload["target_purpose"] = target_purpose

    initial_state: PipelineState = {
        "df_raw": df, # Keep df in state for target_analysis_node
        "diagnosis": diagnosis_for_payload,
        "df_sample_payload": _build_classify_payload(df, diagnosis),
        "has_flagged_columns": _check_flagged(diagnosis),
        "semantic_types": [],
        "recommendations": [],
        "target_column": target_column,
        "target_purpose": target_purpose,
        "target_analysis": None,
        "errors": [],
    }
    result = await _compiled_graph.ainvoke(initial_state)
    return {
        "semantic_types": result.get("semantic_types", []),
        "recommendations": result.get("recommendations", []),
        "target_analysis": result.get("target_analysis"),
        "errors": result.get("errors", []),
    }
