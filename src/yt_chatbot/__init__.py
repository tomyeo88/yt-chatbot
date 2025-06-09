"""YouTube Video Intelligence Chatbot.

A powerful AI-powered chatbot that provides in-depth analysis and optimization
recommendations for YouTube videos using Google's Gemini AI.
"""

__version__ = "0.1.0"

# Import key components for easier access
from yt_chatbot.api.youtube_client import YouTubeClient
from yt_chatbot.api.gemini_client import GeminiClient
from yt_chatbot.database.supabase_client import DatabaseClient

# Re-export commonly used components
__all__ = [
    "YouTubeClient",
    "GeminiClient",
    "DatabaseClient",
]
