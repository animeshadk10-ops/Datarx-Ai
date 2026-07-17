# Setup and Troubleshooting Guide

This document outlines common setup and runtime issues for DataRx AI and their solutions.

## Troubleshooting

### "Session not found" errors
DataRx AI uses an in-memory session store (it has no database by design). If the FastAPI backend restarts or crashes, all active sessions are lost. If you encounter a `404 Session not found` error in the UI:
- **Solution:** Return to the upload step and re-upload your `.csv` file to start a new session.

### CORS errors
If the frontend cannot communicate with the backend and you see CORS errors in your browser console:
- **Check Backend Host/Port:** Ensure the backend is running at `http://localhost:8000`. The frontend is hardcoded to call this exact endpoint.
- **Check Frontend Port:** Ensure the Next.js frontend is running at `http://localhost:3000`. The FastAPI backend's `CORSMiddleware` only allows origins strictly matching `http://localhost:3000`.

### Gemini API errors
If the "Analyze with AI" step fails or returns `500 Internal Server Error`:
- **Verify Key:** Ensure `GOOGLE_API_KEY` in `backend/.env` is correct and does not contain extra spaces.
- **Check Quota/Access:** If the key is valid, you may have hit an API rate limit or billing quota. The application expects access to the `gemini-2.5-flash` model.

### joblib loading failures
If the backend crashes on startup with an error pointing to `semantic_classifier.joblib`:
- **Scikit-learn version mismatch:** The local ML models were trained with a specific version of scikit-learn. If you installed a newer or older version, `joblib` may fail to unpickle the pipeline. 
- **Solution:** Ensure your environment strictly matches the versions defined in `backend/requirements.txt` (or re-run the training script `backend/scripts/train_classifier.py` locally to generate new joblib files).

### LangSmith traces not appearing
If you want to observe the LLM orchestration pipeline but aren't seeing traces in LangSmith:
- **Verify Configuration:** Check `backend/.env` for `LANGCHAIN_TRACING_V2=true` and that `LANGCHAIN_API_KEY` is a valid key.
- **Note:** LangSmith tracing is entirely optional. If you do not provide a LangSmith API key, the application will still function perfectly fine (tracing will simply be disabled).
