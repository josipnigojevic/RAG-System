import os
from typing import List, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter

class DocumentStore:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/index",
        index_name: str = "documents_index"
    ):
    
        self.index_path = index_path
        self.index_name = index_name
        os.makedirs(self.index_path, exist_ok=True)

        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"}
        )

        index_file = os.path.join(self.index_path, f"{self.index_name}.faiss")
        if os.path.exists(index_file):
            self.index = FAISS.load_local(
                folder_path=self.index_path,
                embeddings=self.embedding,
                index_name=self.index_name
            )
        else:
            self.index = None

    def add_documents(self, docs: List[str], metadatas: Optional[List[dict]] = None):

        splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        texts: List[str] = []
        metas: List[dict] = []
        for i, doc in enumerate(docs):
            chunks = splitter.split_text(doc)
            texts.extend(chunks)
            if metadatas:
                metas.extend([metadatas[i]] * len(chunks))

        if self.index is None:
            self.index = FAISS.from_texts(
                texts,
                self.embedding,
                metadatas=metas or None
            )
        else:
            self.index.add_texts(
                texts,
                metadatas=metas or None
            )
        self.index.save_local(self.index_path, self.index_name)

    def search(self, query: str, k: int = 5) -> List[str]:

        if self.index is None:
            return []
        docs = self.index.similarity_search(query, k=k)
        return [d.page_content for d in docs]
