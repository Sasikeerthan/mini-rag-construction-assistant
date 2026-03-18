FROM python:3.11-slim

WORKDIR /app

COPY pip-requirements.txt .
RUN pip install --no-cache-dir -r pip-requirements.txt

# Pre-download the embedding model at build time (not at every startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
