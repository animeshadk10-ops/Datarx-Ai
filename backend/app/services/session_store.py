import pandas as pd

_sessions: dict[str, pd.DataFrame] = {}


def create_session(session_id: str, df: pd.DataFrame) -> None:
    _sessions[session_id] = df


def get_session(session_id: str) -> pd.DataFrame | None:
    return _sessions.get(session_id)


def update_session(session_id: str, df: pd.DataFrame) -> None:
    _sessions[session_id] = df


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
