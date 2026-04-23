# Multi-stage Dockerfile with Tailwind CSS build
FROM node:20-alpine AS css-builder
WORKDIR /app
COPY package.json tailwind.config.js ./
COPY static/input.css ./static/
RUN npm install && npm run build:css

# Stage 2: Python build
FROM python:3.11-slim as python-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg pkg-config gcc musl-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Runtime
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg nginx curl && rm -rf /var/lib/apt/lists/*
RUN rm -f /etc/nginx/nginx.conf /etc/nginx/sites-enabled/* /etc/nginx/sites-available/*

# Copy built assets
COPY --from=css-builder /app/static/style.css ./static/
COPY --from=python-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Copy application
COPY . .
COPY nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /var/cache/nginx /var/log/nginx && \
    chown -R www-data:www-data /var/cache/nginx /var/log/nginx /app

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1
CMD sh -c 'nginx -g "daemon off;" & exec gunicorn app:app --bind 127.0.0.1:5000 --workers 4 --threads 2 --worker-class gthread --timeout 120 --access-logfile - --error-logfile -'
