# Build stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY APP/ ./APP/

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose microservice port
EXPOSE 8000

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application with Uvicorn
CMD ["uvicorn", "APP.app:app", "--host", "0.0.0.0", "--port", "8000"]
