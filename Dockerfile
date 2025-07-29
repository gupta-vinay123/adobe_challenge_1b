# Use official Python base image (CPU, AMD64)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Create input/output folders
RUN mkdir -p /app/input /app/output

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy model and code files
COPY challenge1b.py ./
COPY embedding_model ./embedding_model
COPY summarizer_model ./summarizer_model
COPY yolo_model.pt ./

# (Optional) Copy any other needed files (e.g., tokenizer, config)
# COPY ...

# Set environment variables for offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Entrypoint
ENTRYPOINT ["python", "challenge1b.py"] 