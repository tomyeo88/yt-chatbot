"""
Configuration management for the YouTube Video Intelligence Chatbot.
"""
import os
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Application Settings
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    STREAMLIT_SERVER_PORT: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")
    
    # API Keys
    YOUTUBE_API_KEY: str = Field(..., env="YOUTUBE_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    # Rate Limiting & Caching
    MAX_REQUESTS_PER_MINUTE: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    CACHE_TTL_HOURS: int = Field(default=24, env="CACHE_TTL_HOURS")
    
    # Optional Settings
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    GOOGLE_ANALYTICS_ID: Optional[str] = Field(default=None, env="GOOGLE_ANALYTICS_ID")
    
    # Security
    ALLOWED_HOSTS: list[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields like DATABASE_URL
    
    @validator('ALLOWED_HOSTS', pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse ALLOWED_HOSTS from comma-separated string to list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v or ["localhost", "127.0.0.1"]
    
    @property
    def is_production(self) -> bool:
        """Check if the application is running in production."""
        return not self.DEBUG

# Create settings instance
settings = Settings()

# Export settings for easy access
__all__ = ["settings"]
