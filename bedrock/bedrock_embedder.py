import streamlit as st
import numpy as np
import os
import errno
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import boto3

# AWS Configuration
def initialize_aws():
    """Initialize AWS configuration with environment variables or UI inputs"""
    if not (os.getenv('AWS_ACCESS_KEY_ID') and 
            os.getenv('AWS_SECRET_ACCESS_KEY') and 
            os.getenv('AWS_REGION')):
        # Use streamlit secrets if available
        if 'aws' in st.secrets:
            os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['aws']['aws_access_key_id']
            os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['aws']['aws_secret_access_key']
            os.environ['AWS_REGION'] = st.secrets['aws']['aws_region']
        else:
            # If not in session state, get from user input
            if 'aws_configured' not in st.session_state:
                st.sidebar.header("AWS Configuration")
                aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password")
                aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
                aws_region = st.sidebar.text_input("AWS Region (e.g., us-east-1)")
                
                if st.sidebar.button("Save AWS Configuration"):
                    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
                    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
                    os.environ['AWS_REGION'] = aws_region
                    st.session_state.aws_configured = True
                    st.sidebar.success("AWS credentials saved!")
                return False
            
    return True

# Create embeddings with proper AWS configuration
def create_embeddings():
    """Create BedrockEmbeddings instance with proper configuration"""
    try:
        # Create a boto3 session with explicit credentials
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION')
        )
        
        # Create Bedrock runtime client
        bedrock_client = session.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION')
        )
        
        # Initialize BedrockEmbeddings with the client
        return BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            client=bedrock_client
        )
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def read_file(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        document = ""
        for page in range(len(pdf_reader.pages)):
            document += pdf_reader.pages[page].extract_text()
    else:
        document = file.getvalue().decode("utf-8")
    return document

def rag_search(prompt: str, index_path) -> list:
    allow_dangerous = True
    index_file_path = os.path.join(index_path, "index.faiss")
    if not os.path.exists(index_file_path):
        return [f"Index file {index_file_path} does not exist. Please follow these steps to create it: \n"
                "1. Upload the text or PDF files you want to index. \n"
                "2. Look for the 'Index' button at the bottom of the sidebar. \n"
                "3. Click the 'Index' button to index the files. \n"
                "The indexed files will be saved in the created folder and will be used as your local index."]
    
    embeddings = create_embeddings()
    if embeddings is None:
        return ["Error: Could not create embeddings. Please check AWS configuration."]
        
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)
    docs = db.similarity_search(prompt, k=5)
    return docs

def search_index(prompt: str, index_path: str):
    if prompt:
        matching_docs = rag_search(prompt, index_path)
        st.session_state['matching_docs'] = matching_docs
    else:
        st.session_state['matching_docs'] = []

def save_index(vectorstore, index_path: str = "faiss_index"):
    vectorstore.save_local(index_path)

def index_file(uploaded_files=None, index_path=None):
    if uploaded_files and index_path:
        embeddings = create_embeddings()
        if embeddings is None:
            return None, None, None
            
        documents = [read_file(file) for file in uploaded_files]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents(documents)
        document_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
        combined_embeddings = np.array(document_embeddings)
        
        if len(combined_embeddings.shape) == 1:
            combined_embeddings = combined_embeddings.reshape(-1, 1)

        # Create or load the vectorstore
        if os.path.exists(index_path):
            if not os.listdir(index_path):
                print(f"Directory at {index_path} is empty. Creating new vectorstore.")
                vectorstore = FAISS.from_documents(docs, embeddings)
                save_index(vectorstore, index_path)
            else:
                print(f"Loading existing vectorstore from {index_path}")
                vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(docs)
                save_index(vectorstore, index_path)
        else:
            if not os.path.isdir(index_path):
                try:
                    print(f"Trying to create directory: {index_path}")
                    os.makedirs(index_path, exist_ok=True)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    print(f"Directory {index_path} could not be created. Please check the path and your permissions. Error: {e}")

            print(f"Directory {index_path} does not exist. Creating new vectorstore.")
            vectorstore = FAISS.from_documents(docs, embeddings)
            save_index(vectorstore, index_path)

        return vectorstore, docs, combined_embeddings
    else:
        print("No uploaded files or index path provided.")
        return None, None, None

def main():
    st.title("Document Search with AWS Bedrock")
    
    # Initialize AWS configuration first
    if not initialize_aws():
        st.warning("Please configure AWS credentials in the sidebar to continue.")
        return
        
    index_path = "faiss_index"

    # Initialize session state
    if "files_indexed" not in st.session_state:
        st.session_state["files_indexed"] = False

    # Always display file uploader
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

    if st.button("Index Files"):
        if uploaded_files:
            vectorstore, docs, combined_embeddings = index_file(uploaded_files, index_path)
            if vectorstore is None or docs is None or combined_embeddings is None:
                st.error("Failed to index files. Please check AWS configuration and try again.")
                return
            st.success(f"{len(uploaded_files)} files indexed. Total documents in index: {vectorstore.index.ntotal}")
            st.session_state["files_indexed"] = True
        else:
            st.error("Please upload files before indexing.")

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    st.text_input("Enter your search query", key="search_query", on_change=search_index, args=(st.session_state["search_query"], index_path))
    
    placeholder = st.empty()
    if "matching_docs" in st.session_state:
        with placeholder.container():
            formatted_docs = "\n\n---\n\n".join(str(doc) for doc in st.session_state['matching_docs'])
            st.markdown(f"## Search Results\n\n{formatted_docs}")

if __name__ == "__main__":
    main()
