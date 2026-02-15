import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_area_name(file_path: str) -> str:
    """
    Extracts the clean protected area name from a file path.
    
    Assumes file names are like 'data/AreaName_assessment.pdf'
    """
    # 1. Get the base filename (e.g., 'Vechtplassen_assessment.pdf')
    filename = os.path.basename(file_path)
    
    # 2. Remove the extension (.pdf)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 3. Remove the suffix '_assessment'
    clean_name = name_without_ext.replace('_assessment', '').replace('_ASSESSMENT', '')
    
    # 4. Replace underscores with spaces for a clean display name
    display_name = clean_name.replace('_', ' ')
    
    return display_name

# --- CONFIGURATION ---
PDF_DIRECTORY = "data"  # Create a folder named 'data' and put your 162 PDFs inside
VECTOR_STORE_DIRECTORY = "vector_store" # The folder where the index will be saved
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A popular, fast, and good open-source model

def main():
    """
    This script builds the vector store index from all PDFs in the PDF_DIRECTORY
    and saves it to the VECTOR_STORE_DIRECTORY.
    
    This script is intended to be run only ONCE.
    """
    
    # Clean up the old vector store directory if it exists
    if os.path.exists(VECTOR_STORE_DIRECTORY):
        print(f"Removing existing vector store at: {VECTOR_STORE_DIRECTORY}")
        shutil.rmtree(VECTOR_STORE_DIRECTORY)
        
    print(f"Loading documents from: {PDF_DIRECTORY}")
    
    # 1. Load Documents
    loader = DirectoryLoader(
        PDF_DIRECTORY,
        glob="*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    
    if not documents:
        print(f"No PDF documents found in {PDF_DIRECTORY}. Please check the path.")
        return

    print(f"Loaded {len(documents)} PDF documents.")
    
    # --- CRITICAL CHANGE: ENRICH METADATA FOR FILTERING ---
    # We add a clean 'area_name' field to the metadata of each document.
    for doc in documents:
        # Calls the local get_area_name function
        doc.metadata['area_name'] = get_area_name(doc.metadata['source'])
        
    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} text chunks.")

    # 3. Create Embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model_kwargs = {'device': 'cpu'} 
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs
    )
    print("Embedding model loaded.")

    # 4. Create and Save Vector Store
    print(f"Creating vector store at: {VECTOR_STORE_DIRECTORY}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIRECTORY
    )
    
    vector_store.persist()
    
    print("\n--- SUCCESS! ---")
    print(f"Vector store has been created and saved to: {VECTOR_STORE_DIRECTORY}")
    print(f"You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()