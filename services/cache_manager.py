"""
Cache manager for expensive resources (embedding models, vector stores, LLM clients).
Uses Python's functools.lru_cache for deterministic, reusable caching.
No Streamlit dependencies.

This module manages the lifecycle of heavyweight objects that should be initialized once
and reused across function calls. Unlike Streamlit's @st.cache_resource (which resets on
redeployment), these caches persist for the lifetime of the Python process, making them
suitable for both Streamlit and non-Streamlit applications.
"""

import os
import json
from functools import lru_cache
from typing import TYPE_CHECKING, List

from services.config import AppConfig

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from openai import OpenAI


@lru_cache(maxsize=1)
def get_embedding_model(embedding_model_name: str = "all-MiniLM-L6-v2") -> "HuggingFaceEmbeddings":
    """
    Get or initialize the HuggingFace embedding model.
    Cached on first call; subsequent calls return cached instance.
    
    Args:
        embedding_model_name: Name of the embedding model (default: all-MiniLM-L6-v2)
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"}
    )


@lru_cache(maxsize=1)
def get_vector_store(vector_store_directory: str = "vector_store", 
                     embedding_model_name: str = "all-MiniLM-L6-v2") -> "Chroma":
    """
    Get or initialize the Chroma vector store.
    Cached on first call; subsequent calls return cached instance.
    
    Args:
        vector_store_directory: Path to vector store directory
        embedding_model_name: Name of embedding model to use
        
    Returns:
        Chroma vector store instance
    """
    from langchain_chroma import Chroma
    
    print("Loading vector store...")
    embeddings = get_embedding_model(embedding_model_name)
    return Chroma(
        persist_directory=vector_store_directory,
        embedding_function=embeddings
    )


@lru_cache(maxsize=1)
def get_all_area_names(vector_store_directory: str = "vector_store") -> List[str]:
    """
    Get all unique area names from vector store or cache file.
    Cached on first call; subsequent calls return cached instance.
    
    Tries to load from area_names.json first; falls back to querying vector store.
    
    Args:
        vector_store_directory: Path to vector store directory
        
    Returns:
        Sorted list of area names
    """
    # Try to load from JSON cache first
    area_names_path = os.path.join(vector_store_directory, "area_names.json")
    if os.path.exists(area_names_path):
        try:
            with open(area_names_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading area_names.json: {e}")

    # Fallback: Load from vector store (slow)
    print("Fallback: Loading unique area names from vector store...")
    vector_store = get_vector_store(vector_store_directory)
    documents = vector_store.get(include=["metadatas"])
    unique_area_names = set()
    for metadata in documents["metadatas"]:
        if "area_name" in metadata:
            unique_area_names.add(metadata["area_name"])
    return sorted(list(unique_area_names))


@lru_cache(maxsize=1)
def load_gemeenten() -> List[str]:
    """
    Load unique municipality names from geodata file.
    Cached on first call; subsequent calls return cached instance.
    
    Returns:
        Sorted list of municipality names
    """
    gemeenten_path = "supportdata/Gemeentegrenzen.gml"
    
    try:
        import geopandas as gpd
        
        if not os.path.exists(gemeenten_path):
            print(f"Gemeenten file not found at {gemeenten_path}")
            return []
        
        gemeenten_gdf = gpd.read_file(gemeenten_path)
        return sorted(gemeenten_gdf["Gemeentenaam"].unique().tolist())
    except Exception as e:
        print(f"Error loading gemeenten from {gemeenten_path}: {e}")
        return []


def get_custom_llm(config: AppConfig) -> "ChatOpenAI":
    """
    Get or initialize the LangChain LLM client based on AppConfig.
    Uses internal caching via _custom_llm_cache to handle provider switching.
    
    Args:
        config: AppConfig instance with API credentials and provider selection
        
    Returns:
        ChatOpenAI instance
        
    Raises:
        ValueError: If required credentials are missing for the selected provider
    """
    config.validate()
    return _get_custom_llm_cached(
        provider=config.ai_provider,
        base_url=config.base_url,
        api_key=config.api_key,
        gemini_api_key=config.gemini_api_key or "",
        gemini_base_url=config.gemini_base_url,
    )


@lru_cache(maxsize=2)
def _get_custom_llm_cached(
    provider: str,
    base_url: str,
    api_key: str,
    gemini_api_key: str,
    gemini_base_url: str,
) -> "ChatOpenAI":
    """
    Internal cached function for LLM initialization.
    LRU cache with maxsize=2 allows caching both providers.
    
    Note: All parameters must be hashable (strings), so we pass individual values
    rather than the AppConfig object.
    """
    from langchain_openai import ChatOpenAI
    
    print(f"Initializing custom LLM connection for {provider}...")
    
    if provider == "Mijn Gemini":
        if not gemini_api_key:
            raise ValueError("gemini_api_key is required for 'Mijn Gemini' provider")
        return ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
            base_url=gemini_base_url,
            temperature=1,
        )
    else:
        # Default: Interne OpenAI
        return ChatOpenAI(
            model="gpt-5-mini",
            api_key=api_key,
            base_url=base_url,
            temperature=1,
        )


def get_openai_client(config: AppConfig) -> "OpenAI":
    """
    Get or initialize the OpenAI client based on AppConfig.
    Uses internal caching via _openai_client_cache to handle provider switching.
    
    Args:
        config: AppConfig instance with API credentials and provider selection
        
    Returns:
        OpenAI client instance or None if credentials are missing
    """
    config.validate()
    return _get_openai_client_cached(
        provider=config.ai_provider,
        base_url=config.base_url,
        api_key=config.api_key,
        gemini_api_key=config.gemini_api_key or "",
        gemini_base_url=config.gemini_base_url,
    )


@lru_cache(maxsize=2)
def _get_openai_client_cached(
    provider: str,
    base_url: str,
    api_key: str,
    gemini_api_key: str,
    gemini_base_url: str,
) -> "OpenAI":
    """
    Internal cached function for OpenAI client initialization.
    LRU cache with maxsize=2 allows caching both providers.
    """
    from openai import OpenAI
    
    if provider == "Mijn Gemini":
        if not gemini_api_key:
            return None
        return OpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
    else:
        # Default: Interne OpenAI
        normalized_base_url = base_url
        if normalized_base_url and not normalized_base_url.endswith("/v1"):
            normalized_base_url = normalized_base_url.rstrip("/") + "/v1"
        return OpenAI(api_key=api_key, base_url=normalized_base_url)
