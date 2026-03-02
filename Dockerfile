FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (ffmpeg for whisper; build-essential helps compile any wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 --access-logfile - --error-logfile -
