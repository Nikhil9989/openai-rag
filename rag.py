import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Initialize the embedding model
embeddings = OpenAIEmbeddings()

# Function to process documents and create a vector store
def create_vector_db(file_path, collection_name="my_documents"):
    # Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    # Persist the database
    vectordb.persist()
    
    return vectordb

# Function to query the RAG system
def query_rag(query, vectordb):
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o")
    
    # Create a retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    # Create a document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Execute the chain
    response = retrieval_chain.invoke({"input": query})
    
    return response["answer"]

# Example usage
if __name__ == "__main__":
    # Load or create vector database
    try:
        print("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="my_documents"
        )
    except:
        print("Creating new vector database...")
        # Check if data.txt exists, if not create a sample file
        if not os.path.exists("data.txt"):
            with open("data.txt", "w") as f:
                f.write("This is a sample document for testing the RAG system.\n")
                f.write("RAG stands for Retrieval-Augmented Generation.\n")
                f.write("It combines the power of language models with a knowledge base.\n")
            print("Created sample data.txt file")
        
        vectordb = create_vector_db("data.txt")
    
    # Interactive query loop
    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        
        answer = query_rag(user_query, vectordb)
        print("\nAnswer:", answer)
