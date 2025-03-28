import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rag import query_rag as query_basic_rag
from confluence_rag import query_rag as query_confluence_rag, setup_confluence_rag

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG Q&A System", page_icon="🔍", layout="wide")
st.title("RAG Q&A System")

# Sidebar for configuration
st.sidebar.title("Configuration")
database_type = st.sidebar.radio(
    "Select Knowledge Base Type",
    options=["Basic Documents", "Confluence"],
    index=0
)

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

embeddings = get_embeddings()

# Load the appropriate vector database
@st.cache_resource
def load_vectordb(db_type):
    if db_type == "Basic Documents":
        if os.path.exists("./chroma_db"):
            return Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="my_documents"
            )
        else:
            st.error("No basic document database found. Please run rag.py first to create one.")
            return None
    elif db_type == "Confluence":
        # Check if Confluence credentials are provided
        if not os.getenv("CONFLUENCE_URL") or not os.getenv("CONFLUENCE_USERNAME") or not os.getenv("CONFLUENCE_API_TOKEN"):
            st.error("Confluence credentials not found in .env file. Please add them to use Confluence integration.")
            return None
        
        if os.path.exists("./chroma_db"):
            return Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name="confluence_docs"
            )
        else:
            with st.spinner("Setting up Confluence database for the first time..."):
                try:
                    return setup_confluence_rag(update=False)
                except Exception as e:
                    st.error(f"Error setting up Confluence database: {e}")
                    return None

# Main app logic
try:
    vectordb = load_vectordb(database_type)
    
    if vectordb:
        # Create the input field
        user_query = st.text_input("Ask a question about your documents:")
        
        if user_query:
            with st.spinner("Generating answer..."):
                if database_type == "Basic Documents":
                    answer = query_basic_rag(user_query, vectordb)
                else:  # Confluence
                    answer = query_confluence_rag(user_query, vectordb)
                
                st.markdown("### Answer")
                st.markdown(answer)
        
        # Additional features
        with st.expander("Database Info"):
            collection_name = "my_documents" if database_type == "Basic Documents" else "confluence_docs"
            try:
                collection_count = vectordb._collection.count()
                st.write(f"Collection: {collection_name}")
                st.write(f"Number of documents: {collection_count}")
            except:
                st.write("Could not retrieve collection information.")
    
    else:
        if database_type == "Basic Documents":
            st.info("Please run 'python rag.py' first to create a vector database from your documents.")
        else:
            st.info("Please check your Confluence credentials in the .env file.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Make sure you've set up the environment correctly and have the necessary credentials.")
