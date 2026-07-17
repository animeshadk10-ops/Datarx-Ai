from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables early (side-effect import)
import app.core.config  # noqa: F401

from app.routers import actions, analyze, export, upload

app = FastAPI(title="DataRx AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(actions.router)
app.include_router(export.router)


@app.get("/")
def health_check():
    return {"status": "ok"}
