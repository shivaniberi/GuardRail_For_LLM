#!/bin/bash
set -e

echo ">>> Starting Ollama service..."
ollama serve &
sleep 8

echo ">>> Pulling LLM models (skips if already downloaded)..."
ollama pull qwen:0.5b

echo ">>> Downloading latest artifacts from S3..."
aws s3 cp s3://$S3_BUCKET/embeddings/ ./core/ --recursive
aws s3 cp s3://$S3_BUCKET/models/ ./core/ --recursive

echo ">>> Starting GuardRail FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000
