# OpenAI RAG System

A Retrieval-Augmented Generation (RAG) system built with OpenAI's models and LangChain. This project allows you to create a knowledge base from various sources, including Confluence pages, and query it using natural language.

## Features

- Basic RAG implementation with OpenAI models
- Direct integration with Confluence to load pages and spaces
- Support for document chunking and vector embeddings
- Interactive query interface
- Web UI using Streamlit (optional)

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

## Usage

### Basic RAG System

```bash
python rag.py
```

### Confluence Integration

```bash
python confluence_rag.py
```

### Web Interface (Optional)

```bash
streamlit run app.py
```
