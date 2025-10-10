# Dockerfile — Hugging Face Space (Flask via waitress)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc bash && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md

ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV ALLOWED_ORIGINS="https://viniciuskanh.github.io,http://localhost:3000,http://localhost:5173"
ENV SECRET_KEY="cleansight-secret"

EXPOSE 7860

# Usa a porta dinâmica do Space ($PORT). Fallback para 7860 local.
CMD ["bash","-lc","python -c 'import os;print(\"PORT=\",os.environ.get(\"PORT\"));from waitress import serve;from src.main import app;serve(app, host=\"0.0.0.0\", port=int(os.environ.get(\"PORT\",7860)))'"]
