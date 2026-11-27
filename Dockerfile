FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SE_EPHE_PATH=/app/ephe \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bundle Swiss Ephemeris asteroid, planetary & lunar files (seas_18.se1, sepl_18.se1, semo_18.se1)
COPY data/ephe/ /app/ephe/
COPY data/ephe/ /app/data/ephe/

COPY src/ /app/src/
COPY static/ /app/static/
COPY shards/ /app/shards/
COPY data/*.csv data/*.json data/*.pkl /app/data/
COPY server.py main.py /app/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
