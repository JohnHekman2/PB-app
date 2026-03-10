"""
Configuration module for the application.
Holds all runtime settings (API keys, model paths, user preferences).
Populated once at app startup; services receive this as a parameter.
No Streamlit dependencies.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """
    Centralized application configuration.
    All services receive this as a parameter instead of reading from st.secrets or st.session_state.
    
    Attributes:
        base_url: OpenAI API base URL (from secrets)
        api_key: OpenAI API key (from secrets)
        gemini_api_key: Gemini API key (optional, from secrets)
        gemini_base_url: Gemini API base URL (constant)
        ai_provider: Currently selected AI provider ("Interne OpenAI" or "Mijn Gemini")
        vector_store_directory: Path to Chroma vector store
        embedding_model: Name of embedding model to use
    """
    base_url: str
    api_key: str
    ai_provider: str = "Interne OpenAI"
    gemini_api_key: Optional[str] = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    vector_store_directory: str = "vector_store"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    @classmethod
    def from_streamlit_secrets(cls, st_secrets: dict, ai_provider: str = "Interne OpenAI"):
        """
        Factory method to create AppConfig from Streamlit secrets.
        Called once in app8.py at startup.
        
        Args:
            st_secrets: Streamlit secrets dictionary
            ai_provider: Current AI provider selection
            
        Returns:
            AppConfig instance
            
        Raises:
            KeyError: If required secrets are missing
        """
        return cls(
            base_url=st_secrets["BASE_URL"],
            api_key=st_secrets["API_KEY"],
            ai_provider=ai_provider,
            gemini_api_key=st_secrets.get("GEMINI_API_KEY"),
        )
    
    def validate(self) -> bool:
        """
        Validate that all required fields are properly set.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.base_url or not self.api_key:
            raise ValueError("base_url and api_key are required in AppConfig")
        
        if self.ai_provider == "Mijn Gemini" and not self.gemini_api_key:
            raise ValueError("gemini_api_key is required when ai_provider is 'Mijn Gemini'")
        
        return True
