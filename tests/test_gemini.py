#!/usr/bin/env python3
"""
Test script for the Gemini API and generate_contextual_response function.
This script tests the function directly without the Streamlit interface.
"""

import os
from datetime import datetime
from typing import Dict, Any
import json

# Import the GeminiClient directly
from src.api.gemini_client import GeminiClient

def generate_contextual_response(prompt: str, analysis: Dict[str, Any], gemini_client) -> str:
    """
    Generate a contextual response based on the user's query and video analysis data using Gemini AI.
    
    Args:
        prompt: The user's query
        analysis: Dictionary containing video analysis data
        gemini_client: Instance of GeminiClient
        
    Returns:
        A markdown-formatted response to the user's query
    """
    # Debug output to console
    print(f"Generating response for prompt: {prompt}")
    print(f"Analysis data available: {list(analysis.keys()) if analysis else 'None'}")
    
    # Ensure we have valid analysis data
    if not analysis or not isinstance(analysis, dict):
        return "I don't have enough information about the video to answer that question. Please try analyzing a video first."

    # Extract metadata and analysis data for easier access
    metadata = analysis.get("metadata", {})
    analysis_data = analysis.get("analysis", {})
    scores = analysis.get("scores", {})
    recommendations = analysis.get("recommendations", {})

    # Use Gemini to generate a comprehensive response
    try:
        timestamp = datetime.now().timestamp()
        comprehensive_prompt = f"""You are a YouTube video intelligence analyst assistant. The user has asked: "{prompt}".  
        
        Answer based on the following video analysis data:
        
        ## FACTUAL METADATA (from YouTube API - use this for all factual information):
        Title: {metadata.get('title', 'Unknown')}
        Channel: {metadata.get('channel_title', 'Unknown')}
        Views: {metadata.get('view_count', 'Unknown')}
        Likes: {metadata.get('like_count', 'Unknown')}
        Published: {metadata.get('published_at', 'Unknown')}
        Duration: {metadata.get('duration', 'Unknown')}
        
        ## CONTENT SUMMARY:
        {analysis_data.get('summary', 'No summary available.')}
        
        ## PERFORMANCE SCORES (out of 5):
        Content Quality: {scores.get('content_quality', 'N/A')}/5
        SEO Optimization: {scores.get('seo_optimization', 'N/A')}/5
        Audience Engagement: {scores.get('audience_engagement', 'N/A')}/5
        Technical Performance: {scores.get('technical_performance', 'N/A')}/5
        Market Positioning: {scores.get('market_positioning', 'N/A')}/5
        Overall Score: {scores.get('overall_score', 'N/A')}/10

        ## TOP RECOMMENDATIONS:
        {', '.join(recommendations.get('overall_recommendations', ['No recommendations available.'])[:5])}

        IMPORTANT GUIDELINES:
        1. ONLY use the factual metadata from the YouTube API for any factual claims
        2. DO NOT hallucinate or make up any metadata not present in the data
        3. Format your response in clean, readable markdown
        4. Be comprehensive and directly address the user's query
        5. If the query is about a full analysis report, provide a detailed report with all sections
        6. If the query is about specific aspects (metadata, scores, recommendations), focus on those aspects

        Response ID: {timestamp}
        """
        
        # Set generation config for high quality, deterministic responses
        generation_config = {
            "temperature": 0.2,  # Low temperature for more deterministic responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,  # Allow for detailed responses
        }
        
        # Generate the response with the comprehensive prompt
        print(f"Sending comprehensive prompt to Gemini with response ID: {timestamp}")
        response_obj = gemini_client.generate_content(
            comprehensive_prompt,
            generation_config=generation_config
        )
        
        # Extract text content from the Gemini response object
        response_text = response_obj.text if hasattr(response_obj, 'text') else str(response_obj)
        
        # Debug the response
        print(f"Received response from Gemini with length: {len(response_text)}")
        print(f"Response preview: {response_text[:100]}...")
        
        return response_text
    except Exception as e:
        # If Gemini fails, provide a helpful error message
        print(f"Error generating response with Gemini: {str(e)}")
        error_response = f"I encountered an error processing your request about this video. Please try asking in a different way or try another question.\n\nError details: {str(e)}"
        return error_response


def main():
    """Main function to test the Gemini API and generate_contextual_response function."""
    # Check if GEMINI_API_KEY is set
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
    
    # Initialize GeminiClient
    gemini_client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
    
    # Load a sample analysis from a JSON file if available, otherwise create a mock one
    analysis = {}
    try:
        with open("sample_analysis.json", "r") as f:
            analysis = json.load(f)
        print("Loaded sample analysis from sample_analysis.json")
    except FileNotFoundError:
        # Create a mock analysis
        print("Creating mock analysis data")
        analysis = {
            "metadata": {
                "title": "Sample YouTube Video",
                "channel_title": "Sample Channel",
                "view_count": "10000",
                "like_count": "500",
                "published_at": "2023-01-01",
                "duration": "10:00"
            },
            "analysis": {
                "summary": "This is a sample video about testing the Gemini API."
            },
            "scores": {
                "content_quality": 4.5,
                "seo_optimization": 3.8,
                "audience_engagement": 4.2,
                "technical_performance": 4.0,
                "market_positioning": 3.5,
                "overall_score": 8.0
            },
            "recommendations": {
                "overall_recommendations": [
                    "Add better thumbnails",
                    "Improve video description",
                    "Add more calls to action",
                    "Optimize for keywords",
                    "Improve audio quality"
                ]
            }
        }
    
    # Test prompts
    test_prompts = [
        "get the metadata for this video. https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
        "Tell me about this video",
        "What are the main recommendations?",
        "What's the overall score?",
        "How can I improve this video?",
        "Give me a full analysis report"
    ]
    
    # Test each prompt
    for prompt in test_prompts:
        print("\n" + "="*80)
        print(f"Testing prompt: '{prompt}'")
        print("="*80)
        
        response = generate_contextual_response(prompt, analysis, gemini_client)
        
        print("\nFull response:")
        print("-"*80)
        print(response)
        print("-"*80)
        print("\n")


if __name__ == "__main__":
    main()
