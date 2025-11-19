# Use a lightweight Python base
FROM python:3.9-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies often needed for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your application code
COPY backend.py .

# Expose the port Render/Railway expects (usually provided via $PORT env var)
# But we default to 8000
CMD uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000}