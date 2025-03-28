# OpenAI RAG System

A Retrieval-Augmented Generation (RAG) system built with OpenAI's models and LangChain. This project allows you to create a knowledge base from various sources, including Confluence pages, and query it using natural language.

## Features

- Basic RAG implementation with OpenAI models
- Direct integration with Confluence to load pages and spaces
- Support for document chunking and vector embeddings
- Interactive query interface
- Web UI using Streamlit

## Documentation

📚 Check out our comprehensive documentation:

- [RAG Overview](docs/RAG_Overview.md) - High-level explanation of the RAG implementation
- [RAG Deep Dive](docs/RAG_Deep_Dive.md) - Detailed explanation of each component and enhancement opportunities

## Prerequisites

- Python 3.8+
- OpenAI API key
- Confluence credentials (if using Confluence integration)

## Installation

```bash
# Clone the repository
git clone https://github.com/Nikhil9989/openai-rag.git
cd openai-rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key

# Only needed if using Confluence integration
CONFLUENCE_URL=https://your-instance.atlassian.net
CONFLUENCE_USERNAME=your_email@example.com
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_SPACE_KEY=TEAM
```

The Confluence Space Key is a unique identifier for a specific space in your Confluence instance. It's typically a short, all-uppercase code that you can find in the URL when viewing the space (e.g., `/spaces/TEAM/`).

## Usage

### SQLite Issue Solution

If you encounter an error related to SQLite version, use the FAISS implementations instead:

```bash
# Basic RAG with FAISS
python faiss_rag.py

# Confluence integration with FAISS
python faiss_confluence_rag.py

# Web UI with FAISS
streamlit run faiss_app.py
```

These files use FAISS vector store which doesn't have SQLite dependencies.

### Basic RAG System (ChromaDB)

```bash
python rag.py
```

### Confluence Integration (ChromaDB)

```bash
python confluence_rag.py
```

### Web Interface (ChromaDB)

```bash
streamlit run app.py
```

## Project Structure

- `rag.py`: Basic RAG implementation with ChromaDB
- `confluence_loader.py`: Utility for loading Confluence content
- `confluence_rag.py`: RAG system with Confluence integration (ChromaDB)
- `app.py`: Streamlit web interface with ChromaDB

- `faiss_rag.py`: Basic RAG implementation with FAISS
- `faiss_confluence_rag.py`: RAG system with Confluence integration (FAISS)
- `faiss_app.py`: Streamlit web interface with FAISS

- `requirements.txt`: Project dependencies
- `.env.example`: Template for environment variables

## Troubleshooting

If you get errors about SQLite version:

1. Switch to the FAISS implementations that don't require SQLite
2. Or try to use ChromaDB with an in-memory storage by modifying the code:

```python
from chromadb.config import Settings

# Create custom ChromaDB settings to use in-memory storage
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db",
    anonymized_telemetry=False
)

# Use this in Chroma initialization
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name=collection_name,
    persist_directory="./chroma_db",
    client_settings=chroma_settings  # Add this line
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
