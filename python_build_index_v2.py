import os
import re
import json
import time
import sys
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils import get_area_name

PDF_DIRECTORY = "data"
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROCESSED_FILES_LOG = "processed_files.json"


def load_processed_files() -> set:
    """Load the set of already-processed PDF relative paths."""
    if os.path.exists(PROCESSED_FILES_LOG):
        try:
            with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data)
        except Exception:
            return set()
    return set()


def save_processed_files(processed_files: set) -> None:
    """Save the set of processed PDF relative paths."""
    with open(PROCESSED_FILES_LOG, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(processed_files)), f, ensure_ascii=False, indent=2)


def get_all_pdf_files() -> set:
    """
    Recursively find all PDFs under PDF_DIRECTORY.
    Returns a set of relative paths (relative to PDF_DIRECTORY), e.g. "provx/file.pdf" or "sub/a/b.pdf".
    """
    all_pdfs = set()
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        return all_pdfs

    for root, _, files in os.walk(PDF_DIRECTORY):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, PDF_DIRECTORY).replace("\\", "/")
                all_pdfs.add(rel_path)
    return all_pdfs


def main(rebuild=False):
    """
    This script builds the vector store index incrementally from PDFs organized in subfolders.
    Only NEW PDFs (not previously indexed) are processed and added, unless --rebuild flag is used.
    
    Args:
        rebuild (bool): If True, delete and rebuild the entire vector store from scratch
    """
    
    if rebuild:
        print("🔄 REBUILD MODE: Deleting existing vector store and processing all PDFs...")
        if os.path.exists(VECTOR_STORE_DIRECTORY):
            import shutil
            shutil.rmtree(VECTOR_STORE_DIRECTORY)
            print(f"  ✓ Deleted {VECTOR_STORE_DIRECTORY}/")
        if os.path.exists(PROCESSED_FILES_LOG):
            os.remove(PROCESSED_FILES_LOG)
            print(f"  ✓ Deleted {PROCESSED_FILES_LOG}")
        processed_files = set()
    else:
        processed_files = load_processed_files()
    
    all_pdfs = get_all_pdf_files()
    new_pdfs = all_pdfs - processed_files

    print(f"\nTotal PDFs under '{PDF_DIRECTORY}/': {len(all_pdfs)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"New PDFs to add: {len(new_pdfs)}")

    if not new_pdfs:
        print("✓ Vector store is up to date. No new PDFs to process.")
        if not rebuild:
            print("  Tip: Use 'python python_build_index_v2.py --rebuild' to force a full rebuild.")
        return

    print(f"\nLoading {len(new_pdfs)} new PDF documents...")
    start = time.time()
    new_documents = []
    for rel_path in sorted(new_pdfs):
        pdf_path = os.path.join(PDF_DIRECTORY, *rel_path.split("/"))
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['relative_path'] = rel_path
                doc.metadata['folder'] = os.path.dirname(rel_path).replace("\\", "/")
                doc.metadata['area_name'] = get_area_name(pdf_path)
            new_documents.extend(docs)
            print(f"  ✓ Loaded: {rel_path}")
        except Exception as e:
            print(f"  ✗ Error loading {rel_path}: {e}")

    elapsed = time.time() - start
    print(f"Document loading took: {elapsed:.1f}s")

    if not new_documents:
        print("No documents were successfully loaded.")
        return

    print(f"\nSplitting documents...")
    start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = text_splitter.split_documents(new_documents)
    elapsed = time.time() - start
    print(f"Splitting took: {elapsed:.1f}s")
    print(f"Split into {len(new_chunks)} chunks.")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    start = time.time()
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)
    elapsed = time.time() - start
    print(f"Model loading took: {elapsed:.1f}s")
    print("Embedding model loaded.")

    print("\nConnecting to vector store...")
    if os.path.exists(VECTOR_STORE_DIRECTORY):
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIRECTORY, embedding_function=embeddings)
        print(f"Adding {len(new_chunks)} new chunks to existing vector store...")
        
        # Chroma has a max batch size of 5461, so split into batches
        batch_size = 5000
        start = time.time()
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(new_chunks) + batch_size - 1) // batch_size
            batch_start = time.time()
            print(f"  Adding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            vector_store.add_documents(batch)
            batch_elapsed = time.time() - batch_start
            print(f"    Batch took: {batch_elapsed:.1f}s")
        
        total_elapsed = time.time() - start
        print(f"Total adding chunks took: {total_elapsed:.1f}s")
            
    else:
        print(f"Creating new vector store at: {VECTOR_STORE_DIRECTORY}")
        # For initial creation, also use batches to be safe
        batch_size = 5000
        vector_store = None
        start = time.time()
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(new_chunks) + batch_size - 1) // batch_size
            batch_start = time.time()
            print(f"  Creating batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=VECTOR_STORE_DIRECTORY
                )
            else:
                vector_store.add_documents(batch)
            
            batch_elapsed = time.time() - batch_start
            print(f"    Batch took: {batch_elapsed:.1f}s")
        
        total_elapsed = time.time() - start
        print(f"Total creation took: {total_elapsed:.1f}s")

    processed_files.update(new_pdfs)
    save_processed_files(processed_files)

    print("\n--- SUCCESS! ---")
    print(f"Vector store updated. Total files processed: {len(processed_files)}")
    print("U kunt nu de Streamlit app starten met: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector store from PDFs")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild: delete existing vector store and reprocess all PDFs"
    )
    args = parser.parse_args()
    
    main(rebuild=args.rebuild)