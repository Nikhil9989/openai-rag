# OpenAI RAG Implementation Overview

This document provides a high-level overview of the Retrieval-Augmented Generation (RAG) system implementation in this repository.

## Fundamental Concept of RAG

RAG combines large language models (LLMs) with a retrieval mechanism to enhance responses with specific knowledge. The process works like this:

1. **Document Ingestion**: Text documents are processed, split into chunks, and stored in a vector database
2. **Query Processing**: When a user asks a question, the system retrieves relevant information from the database
3. **Augmented Generation**: The LLM generates a response using both the retrieved information and its own knowledge

## Core Components

### 1. Document Processing Pipeline

#### Text Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
```
This splits documents into manageable chunks of 1000 characters with 200-character overlaps to maintain context across chunks.

#### Embedding Generation
```python
embeddings = OpenAIEmbeddings()
```
This uses OpenAI's embedding model to convert text chunks into numerical vectors that capture semantic meaning.

#### Vector Storage
Two options are implemented:
- **ChromaDB**: A vector database with persistence (has SQLite dependency)
- **FAISS**: A high-performance vector similarity search library (no SQLite dependency)

### 2. Retrieval System

```python
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)
```

This creates a retriever that, when queried, returns the top k most similar document chunks to the query.

### 3. LLM Integration

```python
llm = ChatOpenAI(model="gpt-4o")
```

This initializes the OpenAI model (GPT-4o) that will generate the final responses.

### 4. Prompting Strategy

```python
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")
```

This template provides clear instructions to the model, ensuring it focuses on the retrieved context.

### 5. Retrieval-Augmented Chain

```python
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

This connects the retriever to the LLM, creating a pipeline that:
1. Takes a user query
2. Retrieves relevant documents
3. "Stuffs" them into the prompt template
4. Sends the populated prompt to the LLM
5. Returns the LLM's response

## Confluence Integration

The system can pull content directly from Confluence using the Atlassian API:

```python
confluence_loader = ConfluenceLoader(
    url=os.getenv("CONFLUENCE_URL"),
    username=os.getenv("CONFLUENCE_USERNAME"),
    api_token=os.getenv("CONFLUENCE_API_TOKEN")
)
```

### Confluence-Specific Features

1. **Space Loading**: Loads all pages from a Confluence space
   ```python
   documents = confluence_loader.load_space(space_key, limit=100)
   ```

2. **HTML Processing**: Handles Confluence's HTML content, tables, and macros
   ```python
   def _clean_confluence_content(self, html_content):
       # Converts HTML to clean text with structure preserved
   ```

3. **Attachment Processing**: Processes attached documents like PDFs and DOCXs
   ```python
   def load_page_with_attachments(self, page_id):
       # Loads attachments and processes them
   ```

## File Structure and Implementations

### Basic RAG

- **rag.py/faiss_rag.py**: Simple RAG systems that use text files
  - Creates a vector database from documents
  - Provides an interactive query loop for users

### Confluence RAG

- **confluence_rag.py/faiss_confluence_rag.py**: Enhanced RAG systems with Confluence integration
  - Fetches content from Confluence pages
  - Creates vector embeddings from Confluence content
  - Allows for updates when Confluence content changes

### Web Interface

- **app.py/faiss_app.py**: Streamlit web interfaces
  - Provides a user-friendly web UI
  - Allows switching between basic and Confluence modes
  - Displays query results in a formatted way

## The Query Flow in Detail

1. **User Input**: The user submits a query like "What are the requirements for project X?"

2. **Query Embedding**: The system converts the query to a vector using the same embedding model

3. **Similarity Search**: The system finds the most similar document chunks in the vector database
   ```python
   # Behind the scenes in the retriever
   similar_docs = vectordb.similarity_search(query, k=3)
   ```

4. **Context Assembly**: The retrieved chunks are assembled into the prompt's context
   ```python
   # Inside the chain execution
   context = "\n\n".join([doc.page_content for doc in similar_docs])
   filled_prompt = prompt.format(context=context, input=query)
   ```

5. **LLM Response**: The LLM generates a response based on the query and retrieved context
   ```python
   # Inside the chain execution
   response = llm(filled_prompt)
   ```

6. **Delivery**: The response is returned to the user with relevant information from the documents

## Practical Usage Examples

### Basic Document RAG

```bash
python rag.py  # Or python faiss_rag.py for non-SQLite version
```
- Create a `data.txt` file with your content
- The system will create a vector database and allow you to query it

### Confluence RAG

```bash
python confluence_rag.py  # Or python faiss_confluence_rag.py
```
- Configure your Confluence credentials in the `.env` file
- The system will fetch content from your specified Confluence space
- You can query the Confluence knowledge base directly

### Web Interface

```bash
streamlit run app.py  # Or streamlit run faiss_app.py
```
- Provides a user-friendly interface for your RAG system
- Allows switching between document sources
- Displays answers with formatting

## Performance Considerations

- **Chunk Size**: Larger chunks (>1000 characters) provide more context but reduce precision
- **Overlap**: Higher overlap (>200 characters) maintains context but increases storage
- **Retrieval k-value**: Higher k values (>3) provide more context but may introduce noise
- **Vector Store Choice**: ChromaDB offers more features while FAISS offers better performance

This implementation uses a combination of embedding, retrieval, and generation techniques to create an effective RAG system that can be applied to both simple document collections and complex knowledge bases like Confluence.

For a more detailed examination of each component, see the [RAG Deep Dive](RAG_Deep_Dive.md) document.
