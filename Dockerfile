FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4

# Install dependencies and create non-root user in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    curl \
    && useradd -m -u 1000 streamlit \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown -R streamlit:streamlit /app

# Copy requirements and install with pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=streamlit:streamlit app/ ./app/

# Switch to non-root user
USER streamlit

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.maxUploadSize=2048"]
