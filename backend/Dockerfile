FROM python:3.11-slim

# Install FFmpeg (required for video extraction)
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Render dynamically assigns the PORT environment variable
ENV PORT=8000
EXPOSE $PORT

# Start the FastAPI server using run.py or directly via uvicorn
CMD ["sh", "-c", "uvicorn api:api --host 0.0.0.0 --port $PORT"]
