#!/bin/bash
set -e

echo ">>> Downloading latest artifacts from S3..."
aws s3 cp s3://$S3_BUCKET/embeddings/ ./core/ --recursive 2>/dev/null || echo "S3 embeddings not found, skipping..."
aws s3 cp s3://$S3_BUCKET/models/ ./core/ --recursive 2>/dev/null || echo "S3 models not found, skipping..."

echo ">>> Starting GuardRail FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000
