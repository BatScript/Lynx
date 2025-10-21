# main.py  -- RAG, NO AGENT, WITH CHUNKING
import os
import traceback
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from embedding import vectordb
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import parse_file_by_type

# CONFIG
INPUT_DIR = os.getenv("INPUT_DIR", "./input_directory")
VECTORDB_DIR = os.getenv("VECTORDB_DIR", "./vectordb")
TOP_K = int(os.getenv("TOP_K", "5"))

# Chunking configuration (tunable)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters (approx)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # characters

# System prompt (strict RAG)
SYSTEM_PROMPT = """
You are a retrieval-based assistant. RULES (MANDATORY):
1) Use ONLY the text explicitly provided in the 'CONTEXT' sections below to answer the user's question.
2) Do NOT hallucinate or invent facts. If the answer is not present in the provided contexts, reply exactly: Data not found!
3) Do not return full documents — synthesize a short, direct answer (1-3 sentences) and cite the source filenames in parentheses.
4) If multiple contexts disagree, state that the data is inconsistent and cite the filenames.
"""


def build_or_load_vectorstore(input_dir: str):
    """
    Build or load a persisted Chroma vectorstore.
    - Reads parsed md from ingestion.parse_file_by_type(...)
    - Splits the md into chunks
    - Adds chunks to Chroma with metadata: source, file_hash, chunk_id
    """

    # quick check: if no data present, ingest from input_dir
    try:
        has_content = len(vectordb.get()["metadatas"]) > 0
    except Exception:
        has_content = False

    if not has_content:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = []
        metadatas = []

        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.startswith("."):
                    continue
                path = os.path.join(root, fname)
                try:
                    md, meta = parse_file_by_type(path, use_cache=True)
                except Exception:
                    # skip files that fail parsing
                    continue

                # split text into chunks and create metadata per chunk
                chunks = splitter.split_text(md)
                for i, c in enumerate(chunks):
                    texts.append(c)
                    metadatas.append({"source": fname, "file_hash": meta.get("file_hash"), "chunk_id": i})

        if texts:
            vectordb.add_texts(texts=texts, metadatas=metadatas)
    return vectordb


def build_context_snippets(retrieved_documents, char_limit_per_chunk: int = 2000):
    """
    Format retrieved chunk Documents into a single context string.
    Each chunk is limited to `char_limit_per_chunk`. The function returns a concatenated string.
    """
    parts = []
    for doc in retrieved_documents:
        src = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id")
        text = (doc.page_content or "").strip()
        if len(text) > char_limit_per_chunk:
            text = text[:char_limit_per_chunk] + "\n\n...[truncated]"
        parts.append(f"--- CONTEXT (file: {src}, chunk: {chunk_id}) ---\n{text}\n")
    return "\n".join(parts)


def main():
    load_dotenv()
    print("[start] building/loading vectorstore...")
    try:
        vectordb = build_or_load_vectorstore(INPUT_DIR, VECTORDB_DIR)
    except Exception as e:
        print("Vectorstore init failed:", e)
        return

    # create LLM (adjust model name as needed)
    llm = ChatOllama(model=os.getenv("CHAT_MODEL", "qwen3:8b"))
    
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT
    )

    try:
        query = input("Please Enter the user query below...\n> ").strip()
        
        if not query:
            print("No query provided. Exiting.")
            return

        # retrieve top-k chunks
        results = vectordb.similarity_search(query, k=TOP_K)

        if not results:
            print("Data not found!")
            return

        # build the context for the LLM from retrieved chunks
        context = build_context_snippets(results, char_limit_per_chunk=2000)
        
        print("context for the query : ", query)

        agent_response = agent.invoke({
           "messages" : {
                "role": "user",
                "content": f"CONTEXT:\n\n{context}\n\nUser question: {query}\n\nAnswer concisely and cite sources."
            }
       })
        
        agent_response["messages"][-1].pretty_print()

    except Exception:
        print("Error during query:")
        traceback.print_exc()
        
        
if __name__ == "__main__":
    main()