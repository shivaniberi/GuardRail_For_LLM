FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    curl wget git build-essential zstd unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" \
    && unzip /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Copy and permission startup script
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 11434

CMD ["./start.sh"]
