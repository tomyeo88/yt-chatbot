"""
Video analyzer module for YouTube video intelligence.

This module provides functionality for analyzing YouTube videos using the YouTube API
and Google's Gemini AI model. It integrates video metadata, thumbnails, and related
videos to generate comprehensive analysis and recommendations.
"""

import os

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from src.api.youtube_client import YouTubeClient
from src.api.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    Analyzes YouTube videos using metadata, thumbnails, and AI.

    This class integrates the YouTube API client and Gemini AI client to provide
    comprehensive video analysis, including content evaluation, optimization
    recommendations, and performance scoring.
    """

    def __init__(
        self,
        youtube_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        Initialize the VideoAnalyzer.

        Args:
            youtube_api_key: YouTube API key. If not provided, will try to get from
                           YOUTUBE_API_KEY environment variable.
            gemini_api_key: Google AI API key. If not provided, will try to get from
                          GEMINI_API_KEY environment variable.
        """
        self.youtube_client = YouTubeClient(api_key=youtube_api_key)
        self.gemini_client = GeminiClient(api_key=gemini_api_key)

    def analyze_video(
        self,
        video_url: str,
        include_related: bool = True,
        include_thumbnail: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a YouTube video.

        Args:
            video_url: YouTube video URL or ID
            include_related: Whether to include related videos in the analysis
            include_thumbnail: Whether to include thumbnail image in the analysis

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract video ID from URL if needed
            video_id = self._extract_video_id(video_url)
            if not video_id:
                return {"error": "Invalid YouTube URL or video ID"}

            # Get video metadata
            logger.info(f"Fetching metadata for video ID: {video_id}")
            video_metadata = self.youtube_client.get_video_metadata(video_id)

            if "error" in video_metadata:
                return {"error": video_metadata["error"]}

            # Process metadata into a more usable format with URL and ID
            processed_metadata = self._process_metadata(
                video_metadata, video_url=video_url, video_id=video_id
            )

            # Get thumbnail if requested
            thumbnail_data = None
            thumbnail_analysis = (
                None  # Ensure always defined to prevent UnboundLocalError
            )
            if include_thumbnail:
                logger.info(f"Fetching thumbnail for video ID: {video_id}")
                thumbnail_result = self.youtube_client.get_video_thumbnail(
                    video_id,
                    download=True,
                    format="jpg",
                )
                thumbnail_data = None
                if "error" not in thumbnail_result:
                    # Always pick the best valid image URL (never a video URL)
                    valid_exts = [".jpg", ".jpeg", ".png", ".webp"]

                    def is_image_url(url):
                        return url and any(
                            url.lower().endswith(ext) for ext in valid_exts
                        )

                    high_url = thumbnail_result.get("high_resolution_thumbnail_url")
                    default_url = thumbnail_result.get("default_thumbnail_url")

                    selected_url = None
                    if is_image_url(high_url):
                        selected_url = high_url
                        logger.info(
                            f"Selected high resolution thumbnail: {selected_url}"
                        )
                    elif is_image_url(default_url):
                        selected_url = default_url
                        logger.info(
                            f"Falling back to default thumbnail image: {selected_url}"
                        )
                    else:
                        logger.error(
                            f"No valid image URL found for thumbnail. Got: high={high_url}, default={default_url}"
                        )
                        selected_url = None

                    # Never use a YouTube video URL
                    if selected_url and (
                        "youtube.com/watch" in selected_url
                        or "youtu.be" in selected_url
                    ):
                        logger.error(
                            f"Thumbnail URL points to a video page, not an image: {selected_url}. Skipping thumbnail analysis."
                        )
                        selected_url = None

                    # Only set thumbnail_data if we have a valid image URL
                    if selected_url:
                        try:
                            import requests

                            resp = requests.get(selected_url, timeout=10)
                            if resp.status_code == 200:
                                from PIL import Image
                                from io import BytesIO

                                img = Image.open(BytesIO(resp.content))
                                img_format = img.format if img.format else "JPEG"
                                buffered = BytesIO()
                                img.save(buffered, format=img_format)
                                image_bytes = buffered.getvalue()
                                thumbnail_data = {
                                    "url": selected_url,
                                    "image_bytes": image_bytes,
                                    "local_path": thumbnail_result.get("local_path"),
                                }
                                logger.info(
                                    f"Thumbnail data prepared: dict_keys(['url', 'image_bytes', 'local_path'])"
                                )
                            else:
                                logger.error(
                                    f"Failed to download selected thumbnail image: {selected_url} (HTTP {resp.status_code})"
                                )
                                thumbnail_data = None
                        except Exception as e:
                            logger.error(
                                f"Error downloading or decoding selected thumbnail image: {selected_url} | {str(e)}"
                            )
                            thumbnail_data = None
                    else:
                        logger.error(
                            "No valid image URL available for thumbnail analysis."
                        )
                        thumbnail_data = None
                else:
                    logger.error(
                        f"Error fetching thumbnail: {thumbnail_result.get('error', 'Unknown error')}"
                    )
                    thumbnail_data = None
            else:
                logger.info("Thumbnail fetching not requested")

            # Get related videos if requested
            related_videos = None
            if include_related:
                logger.info(f"Fetching related videos for video ID: {video_id}")
                related_result = self.youtube_client.get_related_videos(
                    video_id, max_results=5, fetch_details=True
                )
                if "error" not in related_result:
                    related_videos = related_result.get("related_videos", [])

            # We'll analyze the video in a structured way and combine all analyses later
            logger.info(f"Preparing to analyze video with Gemini AI: {video_id}")

            # Initialize analysis dictionary
            analysis = {}

            # Generate title analysis
            title_analysis = self._analyze_title(processed_metadata)
            if title_analysis and "error" not in title_analysis:
                analysis["title_analysis"] = title_analysis
                logger.info(
                    f"Title analysis generated successfully with sections: {title_analysis.keys()}"
                )
            else:
                error = (
                    title_analysis.get("error", "Unknown error")
                    if title_analysis
                    else "No analysis returned"
                )
                logger.error(f"Error generating title analysis: {error}")

            # Generate thumbnail analysis if we have thumbnail data
            if thumbnail_data and thumbnail_data.get("image_bytes"):
                logger.info(
                    f"[THUMBNAIL ANALYSIS] Attempting analysis for URL: {thumbnail_data.get('url')}"
                )
                thumbnail_analysis = self._analyze_thumbnail(
                    thumbnail_data, processed_metadata
                )
                if (
                    isinstance(thumbnail_analysis, dict)
                    and "error" in thumbnail_analysis
                ):
                    logger.error(
                        f"[THUMBNAIL ANALYSIS] FAILURE for URL: {thumbnail_data.get('url')} | Reason: {thumbnail_analysis['error']}"
                    )
                    # Propagate error for top-level result
                    analysis["thumbnail_analysis_error"] = thumbnail_analysis
                elif thumbnail_analysis:
                    analysis["thumbnail_analysis"] = thumbnail_analysis
                    logger.info(
                        f"[THUMBNAIL ANALYSIS] SUCCESS for URL: {thumbnail_data.get('url')} | Sections: {getattr(thumbnail_analysis, 'keys', lambda: [])()}"
                    )

            # Analyze video with Gemini and add to analysis
            gemini_full_response = self._analyze_with_gemini( # Renamed for clarity
                processed_metadata, thumbnail_data, related_videos
            )
            if "error" not in gemini_full_response:
                # Merge specific top-level keys from Gemini's response if they exist
                for key_to_merge in ["summary", "scores", "recommendations"]:
                    if key_to_merge in gemini_full_response:
                        analysis[key_to_merge] = gemini_full_response[key_to_merge]
                
                # Merge the 'analysis' sub-dictionary from Gemini's response
                # This sub-dictionary contains 'content_quality', 'content_quality_analysis', etc.
                if "analysis" in gemini_full_response and isinstance(gemini_full_response["analysis"], dict):
                    analysis.update(gemini_full_response["analysis"])
                # Log if the expected 'analysis' sub-dict from Gemini is missing, for debugging.
                elif "analysis" not in gemini_full_response: # Check if 'analysis' key is missing from gemini_full_response
                    logger.warning(f"Gemini response in _analyze_with_gemini did not contain an 'analysis' sub-dictionary as expected. Keys found: {list(gemini_full_response.keys())}")

            # Combine all results into a single dictionary
            result = {
                "video_id": video_id,
                "metadata": processed_metadata,
                "analysis": analysis,
            }

            # Add title analysis directly to the top level for easier access
            if title_analysis and isinstance(title_analysis, dict):
                result["title_analysis"] = title_analysis
                # Also include in analysis for consistency
                result["analysis"]["title_analysis"] = title_analysis
                logger.info("Added title analysis to result")

            # Add thumbnail analysis directly to the top level for easier access
            if thumbnail_analysis and isinstance(thumbnail_analysis, dict):
                result["thumbnail_analysis"] = thumbnail_analysis
                # Also include in analysis for consistency
                result["analysis"]["thumbnail_analysis"] = thumbnail_analysis
                logger.info("Added thumbnail analysis to result")

            # Add thumbnail data if available
            if thumbnail_data:
                result["thumbnail"] = {
                    "url": thumbnail_data.get("url", ""),
                    "local_path": thumbnail_data.get("local_path", ""),
                }

            # Add related videos if requested and available
            if include_related and related_videos:
                result["related_videos"] = related_videos

            # Log the structure of the result for debugging
            logger.info(f"Analysis result structure: {list(result.keys())}")
            if "title_analysis" in result:
                logger.info("Title analysis included in result")
            if "thumbnail_analysis" in result:
                logger.info("Thumbnail analysis included in result")

            return result

        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
            return {"error": f"Error analyzing video: {str(e)}"}

    def _extract_video_id(self, video_url: str) -> Optional[str]:
        """
        Extract video ID from URL or return the ID if already provided.

        Args:
            video_url: YouTube video URL or ID

        Returns:
            Video ID or None if invalid
        """
        # If it's already just an ID (no slashes or dots)
        if "/" not in video_url and "." not in video_url:
            return video_url

        # Otherwise use the YouTube client's extraction method
        return self.youtube_client.extract_video_id(video_url)

    def _process_metadata(
        self, video_data: Dict[str, Any], video_url: str = "", video_id: str = ""
    ) -> Dict[str, Any]:
        """
        Process raw video metadata into a more usable format.

        Args:
            video_data: Raw video metadata from YouTube API
            video_url: URL of the video
            video_id: ID of the video

        Returns:
            Processed metadata dictionary
        """
        snippet = video_data.get("snippet", {})
        statistics = video_data.get("statistics", {})
        content_details = video_data.get("contentDetails", {})

        # Format duration
        duration = content_details.get("duration", "")
        formatted_duration = self._format_duration(duration)

        return {
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "published_at": snippet.get("publishedAt", ""),
            "tags": snippet.get("tags", []),
            "category_id": snippet.get("categoryId", ""),
            "view_count": statistics.get("viewCount", "0"),
            "like_count": statistics.get("likeCount", "0"),
            "comment_count": statistics.get("commentCount", "0"),
            "duration": duration,
            "formatted_duration": formatted_duration,
            "definition": content_details.get("definition", ""),  # hd or sd
            "caption": content_details.get("caption", "false")
            == "true",  # has captions
            "thumbnails": snippet.get("thumbnails", {}),
            "url": video_url,  # Add video URL
            "video_id": video_id,  # Add video ID
        }

    def _format_duration(self, iso_duration: str) -> str:
        """
        Format ISO 8601 duration string into human-readable format.

        Args:
            iso_duration: ISO 8601 duration string (e.g., PT1H30M15S)

        Returns:
            Formatted duration string (e.g., 1:30:15)
        """
        # Delegate to YouTube client's formatting method if available
        if hasattr(self.youtube_client, "_format_duration"):
            return self.youtube_client._format_duration(iso_duration)

        # Simple fallback implementation
        hours, minutes, seconds = 0, 0, 0

        # Remove PT prefix
        duration = iso_duration.replace("PT", "")

        # Extract hours, minutes, seconds
        if "H" in duration:
            hours_part = duration.split("H")[0]
            hours = int(hours_part)
            duration = duration.replace(f"{hours_part}H", "")

        if "M" in duration:
            minutes_part = duration.split("M")[0]
            minutes = int(minutes_part)
            duration = duration.replace(f"{minutes_part}M", "")

        if "S" in duration:
            seconds_part = duration.split("S")[0]
            seconds = int(seconds_part)

        # Format the duration
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def _analyze_with_gemini(
        self,
        metadata: Dict[str, Any],
        thumbnail_data: Optional[Dict[str, Any]] = None,
        related_videos: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze video using Gemini AI.

        Args:
            metadata: Processed video metadata
            thumbnail_data: Optional thumbnail data with base64 encoding
            related_videos: Optional list of related videos

        Returns:
            Analysis results dictionary
        """
        try:
            # Get the video URL from metadata
            video_url = metadata.get("url", "")

            # Prepare thumbnail image for multimodal analysis if available
            thumbnail_image = None
            if thumbnail_data and thumbnail_data.get("image_bytes"):
                import PIL.Image
                import io

                # Try to create image from image_bytes
                try:
                    image_data = thumbnail_data["image_bytes"]
                    thumbnail_image = PIL.Image.open(io.BytesIO(image_data))
                    logger.info(
                        f"Successfully prepared thumbnail image for analysis: {thumbnail_image.format} {thumbnail_image.size}"
                    )
                except Exception as img_error:
                    logger.error(
                        f"Error preparing thumbnail image: {str(img_error)}",
                        exc_info=True,
                    )
                    thumbnail_image = None

            # Use our enhanced analyze_video_content method from GeminiClient
            analysis_result = self.gemini_client.analyze_video_content(
                video_metadata=metadata,
                thumbnail_image=thumbnail_image,
                video_url=video_url,
            )

            # Check for errors
            if "error" in analysis_result:
                logger.error(f"Error in Gemini analysis: {analysis_result['error']}")
                return {"error": analysis_result["error"]}

            return analysis_result

        except Exception as e:
            logger.error(
                f"Error generating analysis with Gemini: {str(e)}", exc_info=True
            )
            return {"error": f"Error generating analysis with Gemini: {str(e)}"}

    def _analyze_title(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the video title using Gemini.

        Args:
            metadata: Video metadata containing the title

        Returns:
            Dictionary with title analysis results including score and recommendations
        """
        try:
            title = metadata.get("title", "")
            if not title:
                return {"error": "No title available for analysis"}

            # Build prompt for title analysis with structured recommendations format
            prompt = f"""
            Analyze this YouTube video title in detail: "{title}"
            
            Please provide a comprehensive analysis of the title covering:
            1. Effectiveness (clarity, conciseness, appeal)
            2. Keywords (presence of relevant keywords and their placement)
            3. Length analysis (optimal length for YouTube)
            4. Clickability (emotional triggers, curiosity gap, value proposition)
            5. Estimated CTR impact (how likely this title is to generate clicks)
            
            Also provide 2-3 specific recommendations for improving the title.
            
            Format your response as a JSON with these keys:
            - effectiveness: detailed analysis of title effectiveness
            - keywords: keyword analysis
            - length: length analysis
            - clickability: clickability analysis
            - ctr_impact: estimated impact on CTR
            - score: a score from 1-5 (with one decimal precision) rating the overall title quality
            - recommendations: array of objects, where each object has these exact keys:
                - action: a single word describing the action (e.g., "Add", "Emphasize", "Clarify")
                - suggestion: the specific suggestion text without any prefixes
                - justification: explanation of why this suggestion would improve the title
            
            IMPORTANT: For the recommendations, follow this exact format for each item:
            {{
              "action": "Add",  // Single word like Add, Emphasize, Clarify, etc.
              "suggestion": "The specific suggestion text without any prefixes",
              "justification": "Explanation of why this would improve the title"
            }}
            
            Do not include prefixes like "type:", "suggestion:", or "justification:" in the actual values.
            """

            # Get analysis from Gemini
            logger.info("Sending title analysis prompt to Gemini")
            response = self.gemini_client.generate_content(prompt)

            # Check if response is valid and has text content
            if (
                response
                and isinstance(response, dict)
                and "text" in response
                and "error" not in response
            ):
                # Parse the response - expecting JSON format
                try:
                    import json

                    # Extract JSON from the response text
                    response_text = response["text"]
                    logger.info(
                        f"Received title analysis response of length: {len(response_text)}"
                    )

                    # Try to find JSON in markdown code blocks
                    if "```json" in response_text:
                        json_content = (
                            response_text.split("```json")[1].split("```")[0].strip()
                        )
                        logger.info("Extracted JSON from code block with json tag")
                    elif "```" in response_text:
                        json_content = (
                            response_text.split("```")[1].split("```")[0].strip()
                        )
                        logger.info("Extracted JSON from generic code block")
                    else:
                        # Try to find JSON-like content in the text
                        if response_text.strip().startswith(
                            "{"
                        ) and response_text.strip().endswith("}"):
                            json_content = response_text.strip()
                            logger.info("Using full response as JSON")
                        else:
                            # Last resort: try to extract anything that looks like JSON
                            import re

                            json_match = re.search(r"\{[^}]*\}", response_text)
                            if json_match:
                                json_content = json_match.group(0)
                                logger.info("Extracted JSON using regex pattern")
                            else:
                                # If no JSON found, create a simple analysis
                                logger.warning(
                                    "No JSON structure found in response, creating simple analysis"
                                )
                                return {
                                    "analysis": response_text,
                                    "score": 3.0,
                                    "recommendations": [
                                        "Consider revising the title for better engagement"
                                    ],
                                }

                    # Parse the JSON content
                    title_analysis = json.loads(json_content)
                    logger.info(
                        f"Successfully parsed title analysis with keys: {title_analysis.keys()}"
                    )

                    # Log the recommendations format for debugging
                    if "recommendations" in title_analysis:
                        logger.info(
                            f"Title recommendations format: {json.dumps(title_analysis['recommendations'][:1], indent=2)}"
                        )

                        # Check if recommendations are in the new format
                        if title_analysis["recommendations"] and isinstance(
                            title_analysis["recommendations"], list
                        ):
                            sample_rec = title_analysis["recommendations"][0]
                            if isinstance(sample_rec, dict):
                                has_new_format = all(
                                    key in sample_rec
                                    for key in ["action", "suggestion", "justification"]
                                )
                                logger.info(
                                    f"Using new recommendation format: {has_new_format}"
                                )
                            else:
                                logger.info(
                                    "Recommendation is not a dictionary, using legacy format parsing"
                                )

                    # Ensure score is within 1-5 range
                    if "score" in title_analysis:
                        title_analysis["score"] = min(
                            5.0, max(1.0, float(title_analysis["score"]))
                        )
                    else:
                        title_analysis["score"] = 3.0  # Default score

                    return title_analysis

                except Exception as e:
                    logger.error(f"Error parsing title analysis response: {str(e)}")
                    # If JSON parsing fails, return a structured analysis with the raw text
                    return {
                        "analysis": response["text"],
                        "score": 3.0,
                        "error": f"JSON parsing failed: {str(e)}",
                    }
            else:
                # Handle error in response
                if response and isinstance(response, dict):
                    error = response.get("error", "Unknown error")
                else:
                    error = "Invalid response format or no response"

                logger.error(f"Error from Gemini for title analysis: {error}")
                return {"error": error}

        except Exception as e:
            logger.error(f"Error in title analysis: {str(e)}")
            return {"error": f"Title analysis failed: {str(e)}"}

    def _analyze_thumbnail(
        self, thumbnail_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the thumbnail image using Gemini Vision.

        Args:
            metadata: Video metadata
            thumbnail_data: Thumbnail data with url and base64 keys

        Returns:
            Dictionary with thumbnail analysis
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Use image_bytes from thumbnail_data (test-style, not base64)
            img_bytes = thumbnail_data.get("image_bytes")
            if not img_bytes:
                logger.error("No image_bytes found in thumbnail_data")
                return {"error": "No image_bytes found in thumbnail_data"}

            # Build all prompts
            title = metadata.get("title", "Unknown")
            logger.info(f"Starting thumbnail analysis for video: {title}")
            prompt = f"""
            Analyze this YouTube video thumbnail in detail. This is for the video titled: "{title}"
            
            Please provide a comprehensive analysis of the thumbnail covering:
            1. Visual elements (people, objects, text, colors, composition)
            2. Design effectiveness (clarity, branding, visual appeal)
            3. Thumbnail optimization (clickability, relevance to title, emotional appeal)
            4. Strengths and weaknesses
            5. Specific improvement recommendations
            6. Estimated CTR impact (how likely this thumbnail is to generate clicks)
            
            Format your response as a JSON with these keys:
            - visual_elements: detailed analysis of visual elements
            - design_effectiveness: analysis of design effectiveness
            - thumbnail_optimization: analysis of optimization opportunities
            - strengths_weaknesses: strengths and weaknesses analysis
            - recommendations: array of specific improvement suggestions
            - ctr_impact: estimated impact on CTR
            - score: a score from 1-5 (with one decimal precision) rating the overall thumbnail quality
            
            Format your response in clear, concise markdown with section headers.
            """
            simpler_prompt = f"""Analyze this YouTube thumbnail for the video titled: \"{metadata.get('title', 'Unknown')}\"\nProvide analysis of: 1) Visual elements 2) Design effectiveness 3) Strengths/weaknesses 4) Recommendations\nRate the thumbnail on a scale of 1-5."""
            minimal_prompt = (
                "Describe this YouTube thumbnail and rate it on a scale of 1-5."
            )

            # Prepare image bytes ONCE
            response = self.gemini_client.generate_content_with_image(
                prompt=prompt, image_bytes=img_bytes
            )
            # Pass the raw response directly to the robust parser
            thumbnail_analysis = self._parse_thumbnail_analysis(response)

            # Check if parsing itself returned an error structure
            if (
                "error" in thumbnail_analysis
                and thumbnail_analysis["error"] is not None
            ):
                logger.error(
                    f"Thumbnail analysis parsing failed. Error: {thumbnail_analysis.get('error')}. "
                    f"Details: {thumbnail_analysis.get('details', 'N/A')}. "
                    f"Raw: {thumbnail_analysis.get('raw_response', 'N/A')}"
                )
                # Return the error structure from the parser
                return thumbnail_analysis

            logger.info(
                f"Successfully parsed thumbnail analysis with keys: {list(thumbnail_analysis.keys())}"
            )
            return thumbnail_analysis
        except Exception as e:
            logger.error(
                f"Error analyzing thumbnail with Gemini: {str(e)}", exc_info=True
            )
            return {
                "error": f"Error analyzing thumbnail with Gemini: {str(e)}",
                "raw_response": "Analysis failed",
            }

    def _parse_thumbnail_analysis(self, response) -> dict:
        """Robustly parse Gemini thumbnail analysis, supporting:
        - dicts with 'text' key (code block or plain JSON)
        - plain string (code block, JSON, or markdown)
        - error dicts (from MCP or internal errors)
        Always returns a dict with all expected keys and correct default types.
        """
        import logging
        import re
        import json

        logger = logging.getLogger(__name__)

        FULL_DEFAULT_THUMBNAIL_ANALYSIS = {
            "visual_elements": {},
            "design_effectiveness": {},
            "thumbnail_optimization": {},
            "strengths_weaknesses": {"strengths": [], "weaknesses": []},
            "recommendations": [],
            "ctr_impact": "",
            "score": 0.0,
            "error": None,
            "details": None,
            "raw_response": None,
            "response_text": None,  # For debugging parsing issues
            "json_content": None,  # For debugging parsing issues
        }

        # Helper to create an error return value
        def _create_error_response(
            error_msg,
            raw_response_val,
            response_text_val=None,
            json_content_val=None,
            details_val=None,
        ):
            err_resp = FULL_DEFAULT_THUMBNAIL_ANALYSIS.copy()
            err_resp.update(
                {
                    "error": error_msg,
                    "raw_response": raw_response_val,
                    "response_text": (
                        response_text_val
                        if response_text_val is not None
                        else str(raw_response_val)
                    ),
                    "json_content": json_content_val,
                    "details": details_val,
                }
            )
            return err_resp

        if response is None:
            logger.warning("Received None response for thumbnail analysis.")
            return _create_error_response("No response received from Gemini", None)

        # Handle Gemini client's own error structure (e.g., from MCP_REQUEST_FAILED)
        if (
            isinstance(response, dict)
            and "error" in response
            and "rc" in response
            and "message" in response
        ):
            if isinstance(response.get("error"), str) and isinstance(
                response.get("message"), str
            ):
                logger.error(
                    f"Gemini client error for thumbnail: {response['error']} - {response['message']}"
                )
                return _create_error_response(
                    f"Gemini client error: {response['error']}",
                    response,
                    details_val=response["message"],
                )

        response_text = None
        json_to_parse_directly = (
            None  # Initialize to avoid UnboundLocalError in edge cases
        )

        if isinstance(response, dict):
            if "text" in response:
                response_text = response["text"]
            else:
                # If it's a dict, not a client error, and no 'text' key, assume it's pre-parsed JSON.
                json_to_parse_directly = response
                logger.info(
                    "Response is a dictionary without 'text' key, attempting to use it directly as parsed JSON."
                )
        elif isinstance(response, str):
            response_text = response
        else:
            logger.warning(
                f"Unexpected response type for thumbnail analysis: {type(response)}"
            )
            return _create_error_response(
                f"Unexpected response type: {type(response).__name__}", response
            )

        if response_text is not None:  # Process response_text if it was populated
            if len(response_text.strip()) >= 2:  # Min JSON is {}
                json_content_str = response_text.strip()
                # Regex to find ```json ... ``` or ``` ... ```
                code_block_match = re.search(
                    r"```(?:json)?\s*([\s\S]+?)\s*```", json_content_str, re.DOTALL
                )
                if code_block_match:
                    json_to_parse_directly = code_block_match.group(1).strip()
                    logger.info(
                        "Extracted JSON from code block for thumbnail analysis."
                    )
                else:
                    json_to_parse_directly = (
                        json_content_str  # Assume the whole string is JSON
                    )
                    logger.info(
                        "No code block detected, attempting to parse entire response_text as JSON for thumbnail."
                    )
            else:  # response_text is too short or empty
                logger.warning(
                    f"No valid text content for thumbnail analysis from response_text. Original response: {response}, Extracted text: {response_text}"
                )
                return _create_error_response(
                    "No valid response text from Gemini for parsing",
                    response,
                    response_text_val=response_text,
                )

        # If json_to_parse_directly is still None here, it means response was not a dict, not a string,
        # or response_text was empty/too short. This case should be caught by earlier checks, but as a safeguard:
        if json_to_parse_directly is None:
            logger.error(
                f"Failed to determine content for JSON parsing. Original response: {response}"
            )
            return _create_error_response(
                "Could not determine content for JSON parsing",
                response,
                response_text_val=response_text,
            )

        # Try parsing the extracted JSON content
        try:
            if isinstance(json_to_parse_directly, dict):  # Already a dict
                parsed_json = json_to_parse_directly
            elif isinstance(json_to_parse_directly, str):
                # Attempt to remove trailing backticks before parsing, as they can appear after markdown blocks
                clean_json_string = re.sub(r"`+$", "", json_to_parse_directly.strip())
                parsed_json = json.loads(clean_json_string)
            else:
                logger.error(
                    f"Internal error: content for JSON parsing is not str or dict. Type: {type(json_to_parse_directly)}"
                )
                return _create_error_response(
                    "Internal error: content for JSON parsing is not str or dict",
                    response,
                    response_text_val=response_text,
                    json_content_val=str(json_to_parse_directly),
                )

            if not isinstance(parsed_json, dict):
                logger.error(
                    f"Parsed JSON is not a dictionary: {type(parsed_json)}. Content: {str(json_to_parse_directly)[:500]}"
                )
                return _create_error_response(
                    f"Parsed content is not a JSON object (dictionary), but {type(parsed_json).__name__}",
                    response,
                    response_text_val=response_text,
                    json_content_val=str(json_to_parse_directly),
                )

            final_analysis = FULL_DEFAULT_THUMBNAIL_ANALYSIS.copy()
            for key, default_value in FULL_DEFAULT_THUMBNAIL_ANALYSIS.items():
                if key in parsed_json:
                    if isinstance(default_value, dict) and not isinstance(
                        parsed_json[key], dict
                    ):
                        logger.warning(
                            f"Key '{key}' in parsed JSON is not a dict as expected, using default. Got: {type(parsed_json[key])}"
                        )
                        final_analysis[key] = default_value.copy()
                    elif isinstance(default_value, list) and not isinstance(
                        parsed_json[key], list
                    ):
                        logger.warning(
                            f"Key '{key}' in parsed JSON is not a list as expected, using default. Got: {type(parsed_json[key])}"
                        )
                        final_analysis[key] = list(default_value)
                    else:
                        final_analysis[key] = parsed_json[key]
                else:
                    if isinstance(default_value, (dict, list)):
                        final_analysis[key] = default_value.copy()
                    else:
                        final_analysis[key] = default_value

            sw_value = final_analysis.get("strengths_weaknesses")
            if not isinstance(sw_value, dict):
                final_analysis["strengths_weaknesses"] = (
                    FULL_DEFAULT_THUMBNAIL_ANALYSIS["strengths_weaknesses"].copy()
                )
            else:
                if not isinstance(sw_value.get("strengths"), list):
                    sw_value["strengths"] = []  # Default to empty list
                if not isinstance(sw_value.get("weaknesses"), list):
                    sw_value["weaknesses"] = []  # Default to empty list

            try:
                score_val = float(
                    final_analysis.get(
                        "score", FULL_DEFAULT_THUMBNAIL_ANALYSIS["score"]
                    )
                )
                final_analysis["score"] = min(5.0, max(0.0, score_val))
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse score '{final_analysis.get('score')}' as float, defaulting to 0.0"
                )
                final_analysis["score"] = FULL_DEFAULT_THUMBNAIL_ANALYSIS["score"]

            final_analysis["error"] = None
            final_analysis["details"] = None
            final_analysis["raw_response"] = response
            final_analysis["response_text"] = response_text
            final_analysis["json_content"] = (
                str(json_to_parse_directly)
                if isinstance(json_to_parse_directly, str)
                else json.dumps(json_to_parse_directly)
            )

            logger.info(
                f"Successfully parsed thumbnail analysis. Score: {final_analysis['score']}"
            )
            return final_analysis

        except json.JSONDecodeError as e:
            logger.error(
                f"JSONDecodeError parsing thumbnail analysis: {e}. Content: {str(json_to_parse_directly)[:500]}"
            )
            # Try to provide more context if json_to_parse_directly was a string
            error_details = str(e)
            if isinstance(json_to_parse_directly, str):
                # Show the problematic part of the string if possible
                problem_area = json_to_parse_directly[max(0, e.pos - 15) : e.pos + 15]
                error_details = f"{e} (near character {e.pos}: '...{problem_area}...'). Check for unescaped quotes, trailing commas, or malformed structures."

            return _create_error_response(
                f"Could not parse thumbnail analysis JSON: {error_details}",
                response,
                response_text_val=response_text,
                json_content_val=str(json_to_parse_directly),
            )
        except Exception as e:
            logger.error(
                f"Unexpected error parsing thumbnail analysis: {e}. Content: {str(json_to_parse_directly)[:500]}",
                exc_info=True,
            )
            return _create_error_response(
                f"Unexpected error during thumbnail parsing: {e}",
                response,
                response_text_val=response_text,
                json_content_val=str(json_to_parse_directly),
            )

    def _build_analysis_prompt(
        self,
        metadata: Dict[str, Any],
        thumbnail_data: Optional[Dict[str, Any]] = None,
        related_videos: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the analysis prompt for Gemini.

        Args:
            metadata: Processed video metadata
            thumbnail_data: Optional thumbnail data
            related_videos: Optional list of related videos

        Returns:
            Formatted prompt string
        """
        # Format metadata for the prompt
        prompt_parts = [
            f"Title: {metadata.get('title', 'N/A')}\n",
            f"Channel: {metadata.get('channel_title', 'N/A')}\n",
            f"Duration: {metadata.get('formatted_duration', 'N/A')}\n",
            f"Published: {metadata.get('published_at', 'N/A')}\n",
            f"Views: {metadata.get('view_count', 'N/A')}\n",
            f"Likes: {metadata.get('like_count', 'N/A')}\n",
            f"Comments: {metadata.get('comment_count', 'N/A')}\n",
            f"Definition: {metadata.get('definition', 'N/A')}\n",
            f"Has Captions: {'Yes' if metadata.get('caption') else 'No'}\n\n",
        ]

        # Add description (truncated if too long)
        description = metadata.get("description", "")
        if description:
            prompt_parts.extend(
                [
                    "## Description\n",
                    f"{description[:1000]}{'...' if len(description) > 1000 else ''}\n\n",
                ]
            )

        # Add tags if available
        tags = metadata.get("tags", [])
        if tags:
            prompt_parts.extend(
                [
                    "## Tags\n",
                    ", ".join(tags[:20]) + (", ..." if len(tags) > 20 else ""),
                    "\n\n",
                ]
            )

        # Add related videos if available
        if related_videos:
            prompt_parts.append("## Related Videos\n")
            for i, video in enumerate(related_videos[:5], 1):
                prompt_parts.append(
                    f"{i}. {video.get('title', 'N/A')} "
                    + f"(Views: {video.get('view_count', 'N/A')}, "
                    + f"Duration: {video.get('formatted_duration', 'N/A')})\n"
                )
            prompt_parts.append("\n")

        # Add analysis instructions
        prompt_parts.extend(
            [
                "## Analysis Instructions\n\n",
                "Please analyze this YouTube video and provide a comprehensive report "
                "with the following sections:\n\n",
                "1. **Content Quality**: Assess the video's content quality based on metadata\n",
                "2. **SEO Analysis**: Evaluate title, description, and tags effectiveness\n",
                "3. **Audience Engagement**: Analyze view-to-like ratio and comment engagement\n",
                "4. **Optimization Recommendations**: Provide 3-5 specific, actionable suggestions\n",
                "5. **Performance Score**: Rate the video on a scale of 1-10 with explanation\n\n",
                "Format your response in Markdown with clear section headers.",
            ]
        )

        return "".join(prompt_parts)

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the analysis response from Gemini into a structured format.

        Args:
            response: Raw response text from Gemini

        Returns:
            Structured analysis data
        """
        import logging

        logger = logging.getLogger(__name__)

        # This is a simplified parser - in a real app, you'd want more robust parsing
        sections = {
            "content_quality": "",
            "seo_analysis": "",
            "audience_engagement": "",
            "optimization_recommendations": [],
            "performance_score": "",
            "raw_response": response,  # Keep the full response for reference
        }

        # Simple section-based parsing
        current_section = None
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith("## ") or line.startswith("# "):
                header = line.lstrip("#").strip().lower()

                if "content" in header and (
                    "quality" in header or "analysis" in header
                ):
                    current_section = "content_quality"
                    logger.debug(f"Found content quality section: {line}")
                    continue
                elif "seo" in header:
                    current_section = "seo_analysis"
                    logger.debug(f"Found SEO analysis section: {line}")
                    continue
                elif "audience" in header or "engagement" in header:
                    current_section = "audience_engagement"
                    logger.debug(f"Found audience engagement section: {line}")
                    continue
                elif (
                    "recommend" in header
                    or "optimiz" in header
                    or "suggestion" in header
                ):
                    current_section = "optimization_recommendations"
                    logger.debug(f"Found recommendations section: {line}")
                    continue
                elif "score" in header or "performance" in header or "rating" in header:
                    current_section = "performance_score"
                    logger.debug(f"Found performance score section: {line}")
                    continue

            # Add content to current section
            if current_section and line:
                if current_section == "optimization_recommendations":
                    if (
                        line.startswith("- ")
                        or line.startswith("* ")
                        or line.startswith("• ")
                        or (
                            len(line) > 2
                            and line[0].isdigit()
                            and line[1:].startswith(". ")
                        )
                    ):
                        # Extract the recommendation text
                        if (
                            line.startswith("- ")
                            or line.startswith("* ")
                            or line.startswith("• ")
                        ):
                            recommendation = line[2:].strip()
                        else:
                            recommendation = line[line.find(".") + 1 :].strip()

                        if (
                            recommendation
                            and recommendation
                            not in sections["optimization_recommendations"]
                        ):
                            sections["optimization_recommendations"].append(
                                recommendation
                            )
                            logger.debug(f"Added recommendation: {recommendation}")
                    elif not any(line.startswith(x) for x in ["#", "```"]):
                        # If not a bullet point but still in recommendations section
                        if line not in sections["optimization_recommendations"]:
                            sections["optimization_recommendations"].append(line)
                            logger.debug(f"Added non-bullet recommendation: {line}")
                else:
                    if sections[current_section]:
                        sections[current_section] += "\n" + line
                    else:
                        sections[current_section] = line

        # Extract numeric score if possible
        score_text = sections["performance_score"]
        try:
            # Look for patterns like "Score: 7/10" or "Rating: 8 out of 10"
            import re

            score_match = re.search(r"(\d+)(?:\s*\/\s*|\s+out\s+of\s+)10", score_text)
            if score_match:
                sections["numeric_score"] = int(score_match.group(1))
                logger.info(f"Extracted numeric score: {sections['numeric_score']}")
            else:
                # Just look for any number between 1-10
                numbers = re.findall(r"\b([1-9]|10)\b", score_text)
                if numbers:
                    sections["numeric_score"] = int(numbers[0])
                    logger.info(
                        f"Extracted numeric score from numbers: {sections['numeric_score']}"
                    )
        except Exception as e:
            # If we can't extract a numeric score, just continue without it
            logger.warning(f"Could not extract numeric score: {str(e)}")
            pass

        # Trim whitespace from text sections
        for key in [
            "content_quality",
            "seo_analysis",
            "audience_engagement",
            "performance_score",
        ]:
            if key in sections and isinstance(sections[key], str):
                sections[key] = sections[key].strip()

        return sections


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize the analyzer
    analyzer = VideoAnalyzer()

    # Example video URL
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Standard test video

    # Analyze the video
    result = analyzer.analyze_video(video_url)

    # Print the result
    import json

    print(json.dumps(result, indent=2))
