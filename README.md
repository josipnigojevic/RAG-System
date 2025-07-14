# RAG Document Retrieval System — Quickstart & Usage Guide

This guide walks you through building, running, and interacting with your Retrieval-Augmented Generation (RAG) service powered by LangChain, FAISS, and a Flan-T5 LLM.

---

## 📦 Prerequisites

- **Docker** & **Docker Compose**  
- **(Optional)** Python 3.10+ & pip (for running the bulk-ingest script locally)  
- **(Optional)** A CLI HTTP client (e.g. `curl`) or Postman (since you have swagger docs and can use the system there)

---

## 🚀 Build & Start

1. **First-time build** (downloads dependencies & models):
   ```bash
   docker compose up --build
   ```
   - The image layer cache will save this work for future runs.

2. **Subsequent starts** (fast!):
   ```bash
   docker compose up
   ```

3. The API will be live at:  
   ```
   http://localhost:8000
   ```

---

## 🔍 Verify the Service

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "RAG Document Retrieval System is running. Use /docs for API docs."
}
```

---

## 📑 API Endpoints

### 1. Ingest Documents  
**`POST /ingest`**  
Adds new text(s) into the FAISS index.

- **Request Body** (JSON):
  ```json
  {
    "documents": [
      "Text of document #1 ...",
      "Text of document #2 ..."
    ],
    "metadata": [
      { "source": "doc1" },
      { "source": "doc2" }
    ]
  }
  ```
  - `metadata` is optional. If provided, it must align one-to-one with `documents`.

- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8000/ingest     -H "Content-Type: application/json"     -d '{
      "documents": [
        "Hubble Space Telescope launched in 1990 ...",
        "It orbits at ~547 km and captures high-res images."
      ],
      "metadata": [
        { "source": "hubble-1" },
        { "source": "hubble-2" }
      ]
    }'
  ```

- **Success Response**:
  ```json
  {
    "ingested": 2,
    "total_documents": 5
  }
  ```

---

### 2. Query the System  
**`POST /query`**  
Retrieves relevant chunks and generates an answer.

- **Request Body** (JSON):
  ```json
  {
    "query": "What can you tell me about the Hubble telescope?"
  }
  ```

- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8000/query     -H "Content-Type: application/json"     -d '{"query":"What can you tell me about the Hubble telescope?"}'
  ```

- **Sample Response**:
  ```json
  {
    "query": "What can you tell me about the Hubble telescope?",
    "answer": "The Hubble Space Telescope is a large telescope in low Earth orbit, launched by NASA in 1990. ..."
  }
  ```

---

## ⚙️ Configuration & Persistence

1. **Persistent Index**  
   Ensure your FAISS index directory stays mounted so it isn’t rebuilt every time:
   ```yaml
   services:
     rag_app:
       volumes:
         - ./data:/app/data
   ```

2. **Custom Models**  
   Override defaults via environment variables in `docker-compose.yml`:
   ```yaml
   environment:
     EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
     GENERATOR_MODEL: google/flan-t5-base
   ```

---

## 🛠️ Bulk Ingestion Script

To ingest the entire 20 Newsgroups corpus asynchronously (avoid startup blocking), run a small Python script **outside** Docker:

```python
# bulk_ingest.py
import requests
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="all")
docs = data.data
metas = [{"category": data.target_names[t]} for t in data.target]

BATCH = 100
for i in range(0, len(docs), BATCH):
    r = requests.post(
        "http://localhost:8000/ingest",
        json={
            "documents": docs[i : i + BATCH],
            "metadata": metas[i : i + BATCH]
        }
    )
    print(f"Ingested {i}–{i+BATCH}: {r.status_code}")
```

Run:
```bash
python bulk_ingest.py
```

---

## 🚨 Troubleshooting

- **No logs / hangs on startup**  
  ⇨ Likely blocked by import-time ingestion.  
  **Solution**: Disable the auto-ingest block in `app/main.py`, restart, then POST to `/ingest` manually.
  Right now the auto bulk ingest is commented out because it takes really long to finnish.

- **Irrelevant answers**  
  ⇨ Your index only contains unrelated docs.  
  **Solution**: Ingest relevant documents first, then re-run `/query`.

---

## 📚 Further Reading

- [LangChain RAG Overview](https://python.langchain.com/docs/modules/data_connection/retrieval)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
