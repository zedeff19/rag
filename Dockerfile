FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-bake model weights into the image so a scale-to-zero cold start doesn't
# have to re-download them from the HF Hub on every wake-up.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app/ app/

EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
