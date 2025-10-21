from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

# use 0.6b for faster results
embeddings = OllamaEmbeddings(model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"))

vectordb = Chroma(
    collection_name=os.getenv("VECTORDB_COLLECTION_NAME", "dqa_doc_embeddings"),
    embedding_function=embeddings,
    persist_directory=os.getenv("VECTORDB_DIR","./dqa_vectordb")
)



