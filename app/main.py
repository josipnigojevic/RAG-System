import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from sklearn.datasets import fetch_20newsgroups    # za auto ingestanje znanja, to je dolje zakomentirano

from app.document_store import DocumentStore
from app.rag_pipeline import RAGPipeline

class IngestRequest(BaseModel):
    documents: list[str]
    metadata: list[dict] = None

class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="Document RAG Retrieval System", version="1.0")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GENERATOR_MODEL_NAME = os.getenv(
    "GENERATOR_MODEL", "google/flan-t5-base"
)

doc_store = DocumentStore(embedding_model=EMBEDDING_MODEL_NAME)
rag_pipeline = RAGPipeline(doc_store, generator_model=GENERATOR_MODEL_NAME)

# if doc_store.index is None:
#     print("Auto-ingesting 20 Newsgroups (this can take 10+ minutes)…")
#     newsgroups = fetch_20newsgroups(subset="all")
#     texts = newsgroups.data
#     metas = [{"category": newsgroups.target_names[t]} for t in newsgroups.target]
#     doc_store.add_documents(texts, metadatas=metas)
#     rag_pipeline.update_retriever()
#     print(f"Indexed {len(texts)} documents from 20 Newsgroups.")

@app.post("/ingest", summary="Ingest new documents into the vector store")
def ingest_documents(request: IngestRequest):

    docs = request.documents
    metas = request.metadata if request.metadata is not None else None
    if not docs:
        raise HTTPException(status_code=400, detail="No documents provided for ingestion.")
    try:
        doc_store.add_documents(docs, metadatas=metas)
        rag_pipeline.update_retriever()
        return {"ingested": len(docs), "total_documents": len(doc_store.index.docstore._dict)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


@app.post("/query", summary="Query the system and get an answer")
def query_documents(request: QueryRequest):

    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    answer = rag_pipeline.answer_query(query)
    return {"query": query, "answer": answer}

@app.get("/", include_in_schema=False)
def index():
    
    return {"message": "RAG Document Retrieval System is running. Use /docs for API docs."}
