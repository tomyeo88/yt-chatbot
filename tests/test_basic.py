"""Basic tests for the YouTube Video Intelligence Chatbot package."""

import pytest
from yt_chatbot import __version__


def test_version():
    """Test that the package version is defined and has the correct format."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    # Version should be in format x.y.z (semantic versioning)
    parts = __version__.split('.')
    assert len(parts) >= 2  # At least major.minor
    assert all(part.isdigit() for part in parts)  # All parts should be numbers


def test_imports():
    """Test that key modules can be imported."""
    # Test that we can import the main package
    import yt_chatbot  # noqa: F401
    
    # Test that we can import key components
    from yt_chatbot.api import YouTubeClient, GeminiClient  # noqa: F401
    from yt_chatbot.database import DatabaseClient  # noqa: F401
    from yt_chatbot.cli import main  # noqa: F401


class TestYouTubeClient:
    """Tests for the YouTubeClient class."""
    
    def test_extract_video_id(self):
        """Test video ID extraction from various URL formats."""
        from yt_chatbot.api.youtube_client import YouTubeClient
        
        client = YouTubeClient()
        
        # Test standard URL
        assert client.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        
        # Test short URL
        assert client.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        
        # Test URL with timestamp
        assert client.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
        
        # Test invalid URL
        assert client.extract_video_id("not a url") is None


class TestGeminiClient:
    """Tests for the GeminiClient class."""
    
    def test_generate_content(self, mocker):
        """Test content generation with the Gemini API."""
        from yt_chatbot.api.gemini_client import GeminiClient
        
        # Mock the Gemini API response
        mock_response = mocker.MagicMock()
        mock_response.text = "Generated content"
        
        # Create a mock for the Gemini client
        mock_client = mocker.MagicMock()
        mock_client.generate_content.return_value = mock_response
        
        # Patch the Gemini client to return our mock
        mocker.patch("yt_chatbot.api.gemini_client.genai.GenerativeModel", return_value=mock_client)
        
        # Test the method
        client = GeminiClient()
        result = client.generate_content("Test prompt")
        
        assert result == "Generated content"
        mock_client.generate_content.assert_called_once_with("Test prompt")
