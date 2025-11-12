FROM python:3.11-slim

WORKDIR /app

COPY ml_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY artifacts ./artifacts
COPY src ./src
COPY ml_service ./ml_service

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "ml_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
