FROM osgeo/gdal:ubuntu-small-3.8.4

# Install pip and OpenCV's headless display dependency
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (GDAL is already in the base image — see requirements.txt)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.maxUploadSize=2048"]
