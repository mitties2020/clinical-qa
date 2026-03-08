FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTESPACE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apk add --no-cache \
    ffmpeg \
    pkg-config \
    gcc \
    musl-dev \
    libavformat \
    libavcodec \
    libavdevice \
    libavfilter \
    libavutil \
    libswresample \
    libswscale

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-"]
