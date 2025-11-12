# Voter Intentions API

FastAPI + scikit-learn inference service for the KNN voter intention model.

## Estructura
- `ml_service/app.py`: FastAPI entrypoint + Pydantic schemas.
- `src/`: custom transformer definitions.
- `artifacts/knn_voter_intentions.joblib`: trained pipeline bundle.
- `requirements.txt`: runtime dependencies.

## Ejecución local
```bash
pip install -r requirements.txt
set PYTHONPATH=src
uvicorn ml_service.app:app --reload --port 8000
```

## Variables
- `ALLOWED_ORIGINS`: lista separada por comas (ej. "https://voterintentions.vercel.app").
- `PYTHONPATH`: debe apuntar a `src` para que FastAPI encuentre los transformadores personalizados.

## Deploy en Vercel
1. Importa este repo en Vercel como proyecto nuevo.
2. Framework preset: **FastAPI** (o **Other** con `pip install -r requirements.txt`).
3. Root directory: `/` (este folder).
4. Variables de entorno: `PYTHONPATH=src`, `ALLOWED_ORIGINS=https://voterintentions.vercel.app`.
