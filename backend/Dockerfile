FROM python:3.11-slim

# Install system dependencies for librosa/soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Start the API server on port 7860 (HuggingFace default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
