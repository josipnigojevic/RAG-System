FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python - <<EOF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
AutoTokenizer.from_pretrained('google/flan-t5-base')
EOF
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
EOF
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
