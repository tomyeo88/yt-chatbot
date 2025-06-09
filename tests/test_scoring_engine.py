"""Tests for the ScoringEngine class with Gemini-based scoring."""

import pytest
import json
from unittest.mock import patch, MagicMock
from src.analysis.scoring_engine import ScoringEngine
from src.api.gemini_client import GeminiClient

# Sample test data
SAMPLE_VIDEO_DATA = {
    "id": "test_video_id",
    "video_id": "test_video_id",  # Added video_id for the refactored code
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Standard test video URL
    "title": "How to Optimize Your YouTube Videos for Better Performance",
    "description": "Learn the best practices for optimizing your YouTube videos to increase views, engagement, and subscriber growth. This comprehensive guide covers everything from thumbnails to SEO.",
    "tags": ["youtube optimization", "video seo", "youtube growth"],
    "thumbnails": {
        "default": {"url": "https://example.com/thumbnail.jpg"},
        "maxres": {"url": "https://example.com/thumbnail_maxres.jpg"}
    },
    "statistics": {
        "viewCount": "10000",
        "likeCount": "500",
        "commentCount": "100"
    },
    "contentDetails": {
        "duration": "PT15M30S"  # 15 minutes 30 seconds
    },
    "definition": "hd",
    "transcript": "This is a sample transcript for testing purposes. It contains keywords like engagement, optimization, and quality content."
}

SAMPLE_ANALYSIS_RESULT = {
    "raw_response": "The video has good content quality and production value.",
    "content_analysis": "The content is well-structured and informative.",
    "title_analysis": {
        "effectiveness": "The title clearly communicates the value proposition and includes relevant keywords.",
        "seo_score": 4.2,
        "recommendations": ["Consider adding a number for more impact", "Test with a question format"]
    },
    "thumbnail_analysis": {
        "effectiveness": "The thumbnail has good contrast and clear text overlay.",
        "visual_appeal": 4.0,
        "recommendations": ["Add a human face for better CTR", "Increase text size for mobile viewers"]
    },
    "seo_analysis": {
        "keyword_optimization": "Good use of primary and secondary keywords.",
        "metadata_completeness": 4.1,
        "recommendations": ["Expand description with more specific keywords", "Add more tags related to the topic"]
    },
    "technical_analysis": {
        "video_quality": "HD resolution with good lighting and clear audio.",
        "production_value": 4.3,
        "recommendations": ["Improve audio consistency", "Consider adding captions"]
    }
}

# Sample Gemini response for testing
SAMPLE_GEMINI_RESPONSE = {
    "score": 4.2,
    "analysis": "This video demonstrates strong hook quality with an effective title and thumbnail.",
    "strengths": ["Clear title that includes keywords", "High-quality thumbnail with good contrast"],
    "weaknesses": ["Title could be more emotionally compelling"],
    "recommendations": ["Add a number or question to the title for better CTR"]
}


class TestScoringEngine:
    """Test cases for the ScoringEngine class."""
    
    @pytest.fixture
    def scoring_engine(self):
        """Create a ScoringEngine instance for testing."""
        with patch('src.analysis.scoring_engine.load_guidelines') as mock_load:
            # Mock the guidelines loading
            mock_load.return_value = {
                "hook": "Test hook guidelines",
                "content": "Test content guidelines",
                "seo": "Test SEO guidelines",
                "technical": "Test technical guidelines",
                "general_guidelines": "Test general guidelines"
            }
            
            # Mock the GeminiClient
            with patch('src.analysis.scoring_engine.GeminiClient') as mock_gemini:
                instance = mock_gemini.return_value
                instance.generate_score_with_guidelines.return_value = SAMPLE_GEMINI_RESPONSE
                
                engine = ScoringEngine(gemini_api_key="fake_key")
                return engine
    
    def test_score_video_integration(self, scoring_engine):
        """Test the full video scoring process."""
        # Score the video
        result = scoring_engine.score_video(SAMPLE_VIDEO_DATA, SAMPLE_ANALYSIS_RESULT.copy())
        
        # Verify the overall structure
        assert "overall_score" in result
        assert "factors" in result
        assert all(factor in result["factors"] for factor in [
            "hook_quality", "content_quality", "seo_optimization", 
            "engagement_metrics", "technical_quality"
        ])
        
        # Verify score ranges
        assert 1 <= result["overall_score"] <= 5
        for factor in result["factors"]:
            assert 1 <= result["factors"][factor]["score"] <= 5
    
    def test_score_hook_quality(self, scoring_engine):
        """Test the hook quality scoring with Gemini."""
        # Mock the Gemini client response
        scoring_engine.gemini_client.generate_score_with_guidelines.return_value = {
            "score": 4.5,
            "analysis": "Excellent hook with compelling title and thumbnail",
            "strengths": ["Clear value proposition", "Eye-catching thumbnail"],
            "weaknesses": [],
            "recommendations": ["Consider A/B testing variations"]
        }
        
        # Test the hook quality scoring
        analysis_result = {}
        score = scoring_engine._score_hook_quality(SAMPLE_VIDEO_DATA, analysis_result)
        
        # Verify the score and analysis
        assert 4.5 == score
        assert "hook_analysis" in analysis_result
        assert "analysis" in analysis_result["hook_analysis"]
        assert "strengths" in analysis_result["hook_analysis"]
        assert "recommendations" in analysis_result["hook_analysis"]
    
    def test_score_hook_quality_fallback(self, scoring_engine):
        """Test the hook quality scoring fallback when Gemini fails."""
        # Make Gemini client raise an exception
        scoring_engine.gemini_client.generate_score_with_guidelines.side_effect = Exception("API error")
        
        # Test the hook quality scoring with fallback
        analysis_result = {}
        score = scoring_engine._score_hook_quality(SAMPLE_VIDEO_DATA, analysis_result)
        
        # Verify a valid score is returned despite the error
        assert 1 <= score <= 5
    
    def test_score_content_quality(self, scoring_engine):
        """Test the content quality scoring with Gemini."""
        # Mock the Gemini client response
        scoring_engine.gemini_client.generate_score_with_guidelines.return_value = {
            "score": 4.2,
            "analysis": "Well-structured content with good information density",
            "strengths": ["Clear explanations", "Good pacing"],
            "weaknesses": ["Could include more examples"],
            "recommendations": ["Add timestamps for key sections"]
        }
        
        # Test the content quality scoring
        analysis_result = {}
        score = scoring_engine._score_content_quality(SAMPLE_VIDEO_DATA, analysis_result)
        
        # Verify the score and analysis
        assert 4.2 == score
        assert "content_analysis" in analysis_result
    
    def test_score_seo_optimization(self, scoring_engine):
        """Test the SEO optimization scoring with Gemini."""
        # Mock the Gemini client response
        scoring_engine.gemini_client.generate_score_with_guidelines.return_value = {
            "score": 3.8,
            "analysis": "Good SEO practices with room for improvement",
            "strengths": ["Relevant tags", "Good keyword density"],
            "weaknesses": ["Description could be longer"],
            "recommendations": ["Add more specific keywords to description"]
        }
        
        # Test the SEO optimization scoring
        analysis_result = {}
        score = scoring_engine._score_seo_optimization(SAMPLE_VIDEO_DATA, analysis_result)
        
        # Verify the score and analysis
        assert 3.8 == score
        assert "seo_analysis" in analysis_result
    
    def test_score_technical_quality(self, scoring_engine):
        """Test the technical quality scoring with Gemini."""
        # Mock the Gemini client response
        scoring_engine.gemini_client.generate_score_with_guidelines.return_value = {
            "score": 4.0,
            "analysis": "Good technical quality with HD resolution",
            "strengths": ["HD resolution", "Clear audio"],
            "weaknesses": ["Some background noise"],
            "recommendations": ["Consider using a noise reduction filter"]
        }
        
        # Test the technical quality scoring
        analysis_result = {}
        score = scoring_engine._score_technical_quality(SAMPLE_VIDEO_DATA, analysis_result)
        
        # Verify the score and analysis
        assert 4.0 == score
        assert "technical_analysis" in analysis_result
    
    def test_score_engagement_metrics(self, scoring_engine):
        """Test the engagement metrics scoring (rule-based)."""
        # Test the engagement metrics scoring
        score = scoring_engine._score_engagement_metrics(SAMPLE_VIDEO_DATA)
        
        # Verify a valid score is returned
        assert 1 <= score <= 5
    
    def test_get_score_description(self, scoring_engine):
        """Test the score description generation."""
        # Test various score ranges
        assert "Excellent" in scoring_engine._get_score_description(4.8)
        assert "Very Good" in scoring_engine._get_score_description(4.2)
        assert "Good" in scoring_engine._get_score_description(3.7)
        assert "Average" in scoring_engine._get_score_description(2.6)
        assert "Poor" in scoring_engine._get_score_description(1.6)
