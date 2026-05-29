FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by chromadb + unstructured
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and data files
COPY app.py .
COPY data/ ./data/
COPY static/ ./static/

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start Flask
CMD ["python", "app.py"]
