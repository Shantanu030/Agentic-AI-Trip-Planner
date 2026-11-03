
# 1. Select Python base image
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy dependency file
COPY requirements.txt .

# 5. Install dependencies
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    -r requirements.txt && \
    pip install --no-cache-dir \
    --default-timeout=300 \
    --retries=10 \
    certifi
# 6. Copy application code
COPY . .

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Start FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
