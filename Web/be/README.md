# StrokeGuard AI Backend

FastAPI skeleton for the StrokeGuard AI web app.

Run locally:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /api/chat/message`
- `POST /api/predict/stroke`
