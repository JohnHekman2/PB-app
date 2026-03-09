import os
import streamlit as st
from openai import OpenAI

# Configuration
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

@st.cache_resource
def get_vector_store():
    from langchain_chroma import Chroma
    print("Loading vector store...")
    return Chroma(persist_directory=VECTOR_STORE_DIRECTORY, embedding_function=get_embedding_model())

@st.cache_resource
def get_custom_llm(provider):
    from langchain_openai import ChatOpenAI
    """LangChain wrapper voor RAG taken"""
    print(f"Initializing custom LLM connection for {provider}...")
    
    # Read secrets directly in the service
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    if provider == "Mijn Gemini":
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY niet gevonden in secrets.toml.")
            st.stop()
        return ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            temperature=1
        )
    else:
        # Default: Interne OpenAI
        return ChatOpenAI(
            model="gpt-5-mini",
            api_key=YOUR_API_KEY,
            base_url=YOUR_API_BASE_URL,
            temperature=1
        )

@st.cache_resource
def get_openai_client(provider):
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    if provider == "Mijn Gemini":
        if not GEMINI_API_KEY:
            return None
        return OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    else:
        # Default: Interne OpenAI
        base_url = YOUR_API_BASE_URL
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip('/') + '/v1' 
        return OpenAI(api_key=YOUR_API_KEY, base_url=base_url)
