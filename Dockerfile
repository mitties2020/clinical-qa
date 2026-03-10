# Multi-stage Dockerfile for VividMedi
# Stage 1: Build Python dependencies
FROM python:3.11-slim as python-builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    pkg-config \
    gcc \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Remove default nginx config
RUN rm -f /etc/nginx/nginx.conf /etc/nginx/sites-enabled/* /etc/nginx/sites-available/*

# Copy Python packages from builder
COPY --from=python-builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application
COPY . .

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Create nginx cache directory and set permissions
RUN mkdir -p /var/cache/nginx /var/log/nginx && \
    chown -R www-data:www-data /var/cache/nginx /var/log/nginx /app

# Expose ports
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start Nginx and Gunicorn
CMD sh -c 'nginx -g "daemon off;" & exec gunicorn app:app --bind 127.0.0.1:5000 --workers 4 --threads 2 --worker-class gthread --timeout 120 --access-logfile - --error-logfile -'
