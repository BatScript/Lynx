# 🦉 Lynx — A Directory-Based Contextual Chatbot

Lynx is a local Retrieval-Augmented Generation (RAG) system that reads, parses, and chats with your files — all from your local directory.
It’s privacy-friendly, modular, and built for developers who want fast, contextual insights from their data.


## ⚙️ Features
- 📂 Directory-based ingestion — reads and processes all supported files automatically.
- 🧩 Supports multiple formats — JSON, PDF, CSV, and HTML.
- 🧠 Contextual Q&A — chats with your documents using vector-based retrieval.
- ⚡ Caching system — powered by SHA-256 hashes to avoid redundant processing.
- 💾 MongoDB + Chroma integration — for metadata and embeddings storage.
- 🔍 Similarity search — efficient context fetching using Chroma’s built-in methods.


## 🧱 Requirements
	•	Python ≥ 3.12
	•	Ollama (for local LLMs)
	•	LlamaIndex API key
	•	uv (recommended for dependency management)


## 🚀 How to Run the App
1.	Pull the models from Ollama
```
ollama pull qwen3:8b
ollama pull qwen3-embedding:4b
```
2.	Set up the environment
```
cp .env-example .env
```

then fill in your values

3.	Run the app
```
uv run main.py
```

(or use python main.py if uv isn’t installed)



## 💡 Tip:
- Make sure INPUT_DIR and VECTORDB_DIR exist before running ingestion.
- Tune CHUNK_SIZE and CHUNK_OVERLAP for optimal accuracy vs. performance.



## 🧩 Architecture Overview
1.	File Scanning & Parsing
Lynx scans the directory and parses supported files (.json, .pdf, .csv, .html) into Markdown using custom parsing functions.
2.	Caching & Storage
Each file is hashed (SHA-256). If the hash exists in MongoDB or Chroma, it’s skipped to save time.
Metadata (like file name, hash, and timestamp) is stored in MongoDB.
3.	Chunking & Embeddings
Parsed Markdown is split into chunks and embedded via the embedding model.
Chunks, along with source metadata, are stored in the Chroma vector database.
4.	Chat Retrieval Flow
	•	User asks a question.
	•	Similar chunks are fetched via similarity_search.
	•	Lynx builds a context snippet containing:
	•	File name
	•	Chunk ID
	•	A preview of the chunk text
	•	This context is sent to the chat model along with the user’s query.
	•	The model generates a contextual, file-aware response.



## 📈 Performance Notes
- Ingestion time scales with file size — optimization is ongoing.
- Chroma’s persistent vector store allows incremental updates.
- Cached files drastically reduce subsequent processing runs.



## 🧪 Tech Stack
- LLMs: Qwen (via Ollama)
- Vector DB: Chroma
- Database: MongoDB
- Framework: LangChain
- Runtime: Python 3.12 / uv
- Env: .env-based configuration



## 📚 Roadmap
- Add async file ingestion
- Add support for .txt, .docx, .md
- UI for querying files
- REST API endpoints for ingestion & query
- Performance metrics dashboard



## 🧠 Example Usage

> User : What are the main points discussed in data.pdf?

> Lynx : The document discusses data ingestion strategies, highlighting schema detection,
data normalization, and the use of streaming pipelines for real-time updates.




## 🧑‍💻 Author

Mohit Ranjan
Built with ❤️ and LangChain.