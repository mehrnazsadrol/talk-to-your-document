FROM python:3.13-slim AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/frontend-requirements.txt

FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --home-dir /app --shell /bin/bash app
WORKDIR /app
RUN chown -R app:app /app \
 && mkdir -p /data \
 && chown -R app:app /data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod \
    API_BASE_URL=http://localhost:8000 \
    CHROMA_PERSIST_DIR=/data/chroma_db \
    PORT=7860

ENV MLFLOW_ENABLED=false \
    DRIFT_LOGGING_ENABLED=false

COPY --chown=app:app app/ ./app/
COPY --chown=app:app frontend/ ./frontend/
COPY hfspace/start.sh /start.sh
RUN chmod +x /start.sh

USER app

EXPOSE 7860
VOLUME ["/data"]

CMD ["/start.sh"]
