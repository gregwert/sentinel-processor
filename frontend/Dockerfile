FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sentinel_frontend/ ./sentinel_frontend/
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "sentinel_frontend/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=2048"]
