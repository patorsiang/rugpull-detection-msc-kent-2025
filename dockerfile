# Dockerfile
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Install OS-level dependencies (libgomp for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install submodule as a package
RUN pip install -e ./backend/utils/feature_extraction/evm_cfg_builder

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
# CMD ["uvicorn", "backend.main:app", "--proxy-headers", "--reload", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.yml"]
