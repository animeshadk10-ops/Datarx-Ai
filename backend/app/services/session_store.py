import pandas as pd

_sessions: dict[str, pd.DataFrame] = {}
_action_history: dict[str, list[dict]] = {}
_original_shapes: dict[str, dict] = {}


def create_session(session_id: str, df: pd.DataFrame) -> None:
    _sessions[session_id] = df
    _action_history[session_id] = []
    _original_shapes[session_id] = {"rows": len(df), "columns": len(df.columns)}


def get_session(session_id: str) -> pd.DataFrame | None:
    return _sessions.get(session_id)


def update_session(session_id: str, df: pd.DataFrame) -> None:
    _sessions[session_id] = df


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    _action_history.pop(session_id, None)
    _original_shapes.pop(session_id, None)

def log_action(session_id: str, column: str, action: str, justification: str, before: dict, after: dict) -> None:
    if session_id not in _action_history:
        _action_history[session_id] = []
    
    _action_history[session_id].append({
        "column": column,
        "action": action,
        "justification": justification,
        "before": before,
        "after": after
    })

def get_summary(session_id: str) -> dict:
    df = _sessions.get(session_id)
    if df is None:
        return {}
    
    return {
        "original_shape": _original_shapes.get(session_id, {"rows": 0, "columns": 0}),
        "final_shape": {"rows": len(df), "columns": len(df.columns)},
        "actions_applied": _action_history.get(session_id, []),
        "export_ready": True
    }
