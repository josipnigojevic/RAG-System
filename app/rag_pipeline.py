from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.document_store import DocumentStore

class RAGPipeline:
    def __init__(
        self,
        doc_store: DocumentStore,
        generator_model: str = "google/flan-t5-base",
        max_new_tokens: int = 200
    ):
    
        self.doc_store = doc_store
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        self.textgen_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_new_tokens,
            do_sample=False
        )
        self.llm = HuggingFacePipeline(pipeline=self.textgen_pipeline)

        if self.doc_store.index:
            self.retriever = self.doc_store.index.as_retriever(search_kwargs={"k": 5})
        else:
            self.retriever = None

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                "{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

        if self.retriever:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
        else:
            self.qa_chain = None

    def answer_query(self, query: str) -> str:

        if not self.qa_chain:
            return "No documents available to answer the query."
        return self.qa_chain.run(query)

    def update_retriever(self):

        if self.doc_store.index:
            self.retriever = self.doc_store.index.as_retriever(search_kwargs={"k": 5})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
