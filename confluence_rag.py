import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from confluence_loader import ConfluenceLoader

# Load environment variables
load_dotenv()

# Initialize models
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")

def setup_confluence_rag(update=False):
    """Set up or update the RAG system with Confluence content."""
    collection_name = "confluence_docs"
    
    if update and os.path.exists("./chroma_db"):
        # Update existing vector store
        vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Update with new content
        confluence_loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_token=os.getenv("CONFLUENCE_API_TOKEN")
        )
        
        space_key = os.getenv("CONFLUENCE_SPACE_KEY")
        documents = confluence_loader.load_space(space_key, limit=20)
        
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        vectordb.add_documents(splits)
        vectordb.persist()
        
        print(f"Updated vector store with {len(splits)} chunks from {len(documents)} documents")
    else:
        # Create new vector store
        confluence_loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_token=os.getenv("CONFLUENCE_API_TOKEN")
        )
        
        space_key = os.getenv("CONFLUENCE_SPACE_KEY")
        documents = confluence_loader.load_space(space_key, limit=100)
        
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        print(f"Created new vector store with {len(splits)} chunks from {len(documents)} documents")
    
    return vectordb

def query_rag(query, vectordb):
    """Query the RAG system."""
    # Create a retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more documents for Confluence content
    )
    
    # Create a prompt that handles Confluence-specific content
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant with access to Confluence documentation. 
    Answer the following question based only on the provided context from Confluence pages.
    
    <context>
    {context}
    </context>
    
    When referencing information, mention the page title if available in the metadata.
    If the information isn't in the context, say so rather than making up an answer.
    
    Question: {input}
    """)
    
    # Create the chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Execute the chain
    response = retrieval_chain.invoke({"input": query})
    
    return response["answer"]

def update_confluence_vector_db(vectordb, space_key=None, page_id=None):
    """Update the vector store with new or changed Confluence content."""
    confluence_loader = ConfluenceLoader(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_token=os.getenv("CONFLUENCE_API_TOKEN")
    )
    
    new_documents = []
    
    if page_id:
        # Update a specific page
        new_documents = [confluence_loader.load_page(page_id)]
    elif space_key:
        # Get recently updated pages from the space
        new_documents = confluence_loader.load_space(space_key, limit=20)
    
    if new_documents:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(new_documents)
        
        # Add to the existing vector store
        vectordb.add_documents(splits)
        vectordb.persist()
        
        print(f"Added {len(splits)} new chunks to the vector store")

if __name__ == "__main__":
    # Check if Confluence credentials are provided
    if not os.getenv("CONFLUENCE_URL") or not os.getenv("CONFLUENCE_USERNAME") or not os.getenv("CONFLUENCE_API_TOKEN"):
        print("Error: Confluence credentials not found in .env file")
        print("Please add CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN, and CONFLUENCE_SPACE_KEY to your .env file")
        exit(1)
    
    # Set up the RAG system
    print("Setting up Confluence RAG system...")
    
    try:
        vectordb = setup_confluence_rag(update=False)
        
        # Interactive query loop
        while True:
            user_query = input("\nEnter your question about Confluence content (or 'exit' to quit): ")
            if user_query.lower() == "exit":
                break
            
            print("\nSearching Confluence knowledge base...")
            answer = query_rag(user_query, vectordb)
            print("\nAnswer:", answer)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Confluence credentials are correct and you have access to the specified space.")
