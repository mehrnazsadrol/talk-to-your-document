# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd --create-home --home-dir /app --shell /bin/bash app
WORKDIR /app
RUN chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_BASE_URL=http://api:8000

EXPOSE 8501

COPY --chown=app:app frontend/ ./frontend/

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
