"""
LLM service module - wrapper around cache_manager.
Delegates all operations to cache_manager for backwards compatibility.
No Streamlit dependencies.

Note: All LLM and embedding model initialization is now managed by cache_manager.py
which handles caching through Python's functools.lru_cache instead of Streamlit's @st.cache_resource.
"""

from services.cache_manager import (
    get_embedding_model,
    get_vector_store,
    get_custom_llm,
    get_openai_client,
)

# Configuration constants (legacy, kept for reference)
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

__all__ = [
    "get_embedding_model",
    "get_vector_store",
    "get_custom_llm",
    "get_openai_client",
    "VECTOR_STORE_DIRECTORY",
    "EMBEDDING_MODEL",
]
