"""
Test direct validation of Pydantic schemas used in the YouTube video analysis.

This test validates that each schema correctly validates proper JSON data
and rejects invalid data appropriately.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.schemas import (
    ContentQualityAnalysisSchema,
    TitleAnalysisSchema, 
    ThumbnailAnalysisSchema,
    SEOAnalysisSchema,
    AudienceEngagementSchema,
    TechnicalPerformanceSchema
)
from pydantic import ValidationError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_title_analysis_schema():
    """Test validation using the TitleAnalysisSchema."""
    print("\n=== Testing TitleAnalysisSchema ===")
    
    # Valid data
    valid_data = {
        "score": 4.0,
        "effectiveness": "The title is compelling and clearly communicates the video topic",
        "keywords": "Good use of primary keywords",
        "length": "Appropriate length under 60 characters",
        "clarity": "Clear and easy to understand",
        "emotional_appeal": "Moderate emotional appeal",
        "recommendations": [
            "Add more emotional keywords",
            "Consider including numbers for better CTR"
        ]
    }
    
    try:
        validated = TitleAnalysisSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
        print(f"   First recommendation: {validated.recommendations[0]}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")
    
    # Invalid data - missing required field
    invalid_data = {
        "effectiveness": "The title is compelling",
        "recommendations": ["Add keywords"]
    }
    
    try:
        validated = TitleAnalysisSchema.parse_obj(invalid_data)
        print("❌ Invalid data unexpectedly passed validation")
    except ValidationError as e:
        print(f"✅ Invalid data correctly rejected: missing required fields")

def test_thumbnail_analysis_schema():
    """Test validation using the ThumbnailAnalysisSchema."""
    print("\n=== Testing ThumbnailAnalysisSchema ===")
    
    # Valid data
    valid_data = {
        "score": 4.0,
        "design_effectiveness": "Strong visual hierarchy",
        "clarity": "Text is readable",
        "branding": "Consistent with channel identity",
        "visual_appeal": "Good use of contrasting colors",
        "visual_elements": {
            "people": "Presenter shown with engaged expression",
            "objects": "Relevant props",
            "text": "Key phrase that supports title",
            "colors": "High contrast",
            "composition": "Z-pattern reading flow"
        },
        "thumbnail_optimization": "Well-optimized",
        "clickability": "High clickability",
        "relevance_to_title": "Directly supports title's promise",
        "emotional_appeal": "Creates curiosity",
        "recommendations": [
            "Increase text size slightly",
            "Test more saturated colors"
        ]
    }
    
    try:
        validated = ThumbnailAnalysisSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
        print(f"   First recommendation: {validated.recommendations[0]}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")
    
    # Test partial data (only required fields)
    partial_data = {
        "score": 3.5,
        "recommendations": ["Use brighter colors"]
    }
    
    try:
        validated = ThumbnailAnalysisSchema.parse_obj(partial_data)
        print("✅ Partial data with only required fields passes validation")
    except ValidationError as e:
        print(f"❌ Partial data failed validation: {e}")

def test_seo_analysis_schema():
    """Test validation using the SEOAnalysisSchema."""
    print("\n=== Testing SEOAnalysisSchema ===")
    
    # Valid data
    valid_data = {
        "score": 3.0,
        "title_effectiveness": "Good keyword inclusion",
        "description": "Contains relevant keywords",
        "tags": "Good variety but missing some terms",
        "thumbnail_clickability": "Thumbnail reinforces key topics",
        "recommendations": [
            "Reorganize title",
            "Add timestamps in description"
        ]
    }
    
    try:
        validated = SEOAnalysisSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")

def test_content_quality_schema():
    """Test validation using the ContentQualityAnalysisSchema."""
    print("\n=== Testing ContentQualityAnalysisSchema ===")
    
    # Valid data with all fields
    valid_data = {
        "score": 4.0,
        "summary": "High-quality content with good depth and presentation",
        "clarity": "Concepts are explained clearly with good examples",
        "depth_of_information": "Comprehensive coverage of the topic",
        "structure_and_flow": "Logical progression with smooth transitions",
        "value_proposition": "Strong practical value for the audience",
        "engagement_factors": "Good use of storytelling",
        "originality": "Presents unique perspective",
        "accuracy": "Information is factual and well-researched",
        "call_to_action_effectiveness": "Clear and compelling CTA",
        "script_quality": "Well-written script",
        "presentation_style": "Confident delivery",
        "editing_and_pacing": "Tight editing",
        "key_topics": [
            "Main subject introduction",
            "Key concepts explained"
        ],
        "strengths": [
            "In-depth explanation of complex concepts",
            "Effective use of visual aids"
        ],
        "weaknesses": [
            "Introduction could be more concise",
            "Some technical terms not fully explained"
        ],
        "recommendations": [
            "Tighten the introduction",
            "Add brief definitions for technical terms"
        ]
    }
    
    try:
        validated = ContentQualityAnalysisSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
        print(f"   Summary: {validated.summary}")
        print(f"   Strengths count: {len(validated.strengths)}")
        print(f"   Weaknesses count: {len(validated.weaknesses)}")
        print(f"   Recommendations count: {len(validated.recommendations)}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")

def test_audience_engagement_schema():
    """Test validation using the AudienceEngagementSchema."""
    print("\n=== Testing AudienceEngagementSchema ===")
    
    # Valid data
    valid_data = {
        "score": 3.0,
        "hook_strength": "Opening grabs attention",
        "storytelling": "Good narrative structure",
        "ctas": "Clear calls to action",
        "community_potential": "Some comments show interaction",
        "recommendations": [
            "Add a stronger hook",
            "Include personal stories"
        ]
    }
    
    try:
        validated = AudienceEngagementSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")

def test_technical_performance_schema():
    """Test validation using the TechnicalPerformanceSchema."""
    print("\n=== Testing TechnicalPerformanceSchema ===")
    
    # Valid data
    valid_data = {
        "score": 4.0,
        "video_quality": "Clear 1080p resolution",
        "audio_quality": "Clear audio with minimal noise",
        "length_appropriateness": "Good length for the topic",
        "accessibility": "Missing captions",
        "recommendations": [
            "Add closed captions",
            "Improve text contrast"
        ]
    }
    
    try:
        validated = TechnicalPerformanceSchema.parse_obj(valid_data)
        print("✅ Valid data passes validation")
        print(f"   Score: {validated.score}")
    except ValidationError as e:
        print(f"❌ Valid data failed validation: {e}")

def direct_schema_validation_test():
    """Run all schema validation tests directly."""
    print("\nRunning direct Pydantic schema validation tests...")
    
    test_title_analysis_schema()
    test_thumbnail_analysis_schema()
    test_seo_analysis_schema()
    test_content_quality_schema()
    test_audience_engagement_schema()
    test_technical_performance_schema()
    
    print("\nDirect schema validation tests completed.")

if __name__ == "__main__":
    print("Starting Pydantic Schema Validation Tests...")
    direct_schema_validation_test()
    print("\nAll tests completed.")
