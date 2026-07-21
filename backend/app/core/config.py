from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2: str | None = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT: str | None = os.getenv("LANGCHAIN_PROJECT")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3.5-flash")
