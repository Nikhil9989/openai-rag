---
marp: true
theme: default
paginate: true
backgroundColor: #FFFFFF
---

# Retrieval-Augmented Generation (RAG) System
## With OpenAI and Confluence Integration

---

# Agenda

1. RAG Overview
2. System Architecture
3. Core Components
4. Confluence Integration
5. Implementation & Code Walkthrough
6. Demo & Use Cases
7. Next Steps & Enhancements

---

# What is RAG?

Retrieval-Augmented Generation (RAG) combines:

* **Retrieval**: Finding relevant information from a knowledge base
* **Generation**: Using an LLM to create responses based on retrieved information

![RAG Process](https://i.imgur.com/JORrn6o.png)

---

# Why RAG?

* **Reduces hallucinations** by grounding responses in factual information
* **Overcomes knowledge cutoff** of LLMs
* **Domain-specific knowledge** without fine-tuning
* **Transparency** through source attribution
* **Reduced costs** compared to fine-tuning

---

# System Architecture

![System Architecture](https://i.imgur.com/AfV9LJh.png)

---

# Document Processing Pipeline

1. **Text Splitting**: Break documents into manageable chunks
2. **Embedding Generation**: Convert text to vectors
3. **Vector Storage**: Store embeddings in a vector database

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
```

---

# Retrieval System

When user asks a question:
1. Convert question to embedding
2. Find most similar chunks in the vector database
3. Return top k results

```python
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

---

# LLM Integration

* Using OpenAI's models (default: GPT-4o)
* Combines retrieved context with the query
* Generates relevant, accurate responses

```python
llm = ChatOpenAI(model="gpt-4o")
```

---

# Prompting Strategy

Clear instructions help the model use retrieved content effectively:

```python
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")
```

---

# Retrieval-Augmented Chain

Connects all components into a seamless pipeline:

```python
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Usage
response = retrieval_chain.invoke({"input": query})
```

---

# Confluence Integration

Directly access your team's Confluence knowledge base:

```python
confluence_loader = ConfluenceLoader(
    url=os.getenv("CONFLUENCE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"),
    api_token=os.getenv("CONFLUENCE_API_TOKEN")
)

# Load all pages from a space
documents = confluence_loader.load_space(space_key, limit=100)
```

---

# Confluence Features

* **HTML Processing**: Handles tables, macros, and formatting
* **Attachment Support**: Processes PDFs, Word docs, etc.
* **Space Navigation**: Load entire spaces or specific pages
* **Metadata Preservation**: Retains page titles, authors, URLs

---

# Implementation Options

**Two Vector Database Options:**

1. **ChromaDB**: Feature-rich, but requires SQLite 3.35+
2. **FAISS**: High performance, no SQLite dependency

**User Interface Options:**

1. **Command Line**: Simple interactive experience
2. **Streamlit Web UI**: User-friendly interface

---

# Project Structure

* **rag.py/faiss_rag.py**: Basic RAG systems
* **confluence_loader.py**: Utility for Confluence
* **confluence_rag.py/faiss_confluence_rag.py**: Confluence integration
* **app.py/faiss_app.py**: Streamlit web interfaces
* **docs/**: Comprehensive documentation

---

# The Query Flow

![Query Flow](https://i.imgur.com/yq5n3hc.png)

---

# Performance Considerations

* **Chunk Size**: 1000 characters (balance between context and precision)
* **Overlap**: 200 characters (maintains context across boundaries)
* **Retrieval k-value**: 3-5 documents (more isn't always better)
* **Vector Store Choice**: ChromaDB (features) vs FAISS (performance)

---

# Enhancement Opportunities

* **Semantic Chunking**: Split based on meaning, not just character count
* **Maximum Marginal Relevance (MMR)**: Balance relevance with diversity
* **Query Transformation**: Expand/refine queries for better retrieval
* **Structured Output**: Format responses consistently

---

# Getting Started

1. Clone the repository
2. Install dependencies
3. Configure API keys
4. Run the appropriate script:

```bash
# Basic RAG
python faiss_rag.py

# Confluence integration
python faiss_confluence_rag.py

# Web UI
streamlit run faiss_app.py
```

---

# Demo

Let's see it in action:

1. Loading content from Confluence
2. Building a vector database
3. Asking questions through the UI
4. Examining the retrieved context

---

# Next Steps

* **Add additional data sources**: SharePoint, Jira, etc.
* **Implement conversational memory**: Support follow-up questions
* **Improve embedding strategies**: Experiment with other models
* **Add evaluation framework**: Measure response quality

---

# Questions?

Thank you!

GitHub: [github.com/Nikhil9989/openai-rag](https://github.com/Nikhil9989/openai-rag)
