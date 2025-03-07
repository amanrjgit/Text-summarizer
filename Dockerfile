# Use a stable Python version
FROM python:3.12  

# Set working directory
WORKDIR /app  

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_NAME="sshleifer/distilbart-cnn-12-6" \
    LOG_LEVEL="INFO" \
    MAX_WORKERS=3  

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  

# Install Python dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt  

# Copy application code
COPY . .  

# Create a non-root user and set ownership
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app  
USER appuser  

# Expose the FastAPI application port
EXPOSE ${PORT}  

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  
