"""
Google Gemini API client for video content analysis and insights generation.
"""

import os
from typing import Dict, List, Optional, Any, Union
import google.generativeai as genai
import logging
import json
import re
from src.analysis.schemas import ContentQualityAnalysisSchema  # Added for JSON output

logger = logging.getLogger(__name__)

from pydantic import ValidationError  # Added for Pydantic validation


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client.

        Args:
            api_key: Google AI API key. If not provided, will try to get from
                   GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set GEMINI_API_KEY environment variable."
            )

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Use the specified models
        self.default_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        self.fallback_model = genai.GenerativeModel("gemini-2.0-flash")
        self.vision_model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20"
        )  # For image/video analysis

    def generate_content_with_image(
        self,
        prompt: str,
        image_bytes,
        model: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Generate content using the Gemini model with an image input.

        Args:
            prompt: The text prompt to send to the model
            image_bytes: The image bytes to analyze
            model: Optional model to use. If 'gemini-2.0-flash', use self.fallback_model, else use self.vision_model.
            generation_config: Optional configuration for generation

        Returns:
            Response object from the Gemini model
        """
        try:
            model_to_use = self.vision_model
            model_name = "gemini-2.5-flash-preview-05-20"
            if model == "gemini-2.0-flash":
                model_to_use = self.fallback_model
                model_name = "gemini-2.0-flash"
            logger.info(f"[GeminiClient] Using model: {model_name}")
            logger.info(
                f"[GeminiClient] Prompt (first 120 chars): {prompt.strip()[:120]}"
            )
            logger.info(
                f"[GeminiClient] Image bytes length: {len(image_bytes) if hasattr(image_bytes, '__len__') else 'N/A'}"
            )
            # Prepare image for Gemini API
            image_dict = {"mime_type": "image/jpeg", "data": image_bytes}
            # Gemini SDK expects image as dict with 'inline_data'
            gemini_image = {"inline_data": image_dict}
            # Call Gemini API
            import traceback

            try:
                response = model_to_use.generate_content(
                    [prompt, gemini_image], generation_config=generation_config
                )
            except Exception as e:
                logger.error(
                    f"[GeminiClient][DEBUG] Exception during Gemini API call: {str(e)}"
                )
                logger.error(traceback.format_exc())
                return {
                    "error": f"Exception during Gemini API call: {str(e)}",
                    "traceback": traceback.format_exc(),
                }

            # Ultra-verbose debug logging of raw Gemini response
            try:
                logger.info(
                    f"[GeminiClient][DEBUG] Raw response type: {type(response)}"
                )
                logger.info(
                    f"[GeminiClient][DEBUG] Raw response repr: {repr(response)}"
                )
                logger.info(f"[GeminiClient][DEBUG] Raw response dir: {dir(response)}")
                # Defensive Gemini response handling with detailed logging
                if hasattr(response, "text"):
                    response_text_val = response.text
                    logger.info(
                        f"[GeminiClient][Parse][Image] Response has direct text attribute."
                    )
                    logger.debug(
                        f"[GeminiClient][Parse][Image] Type of response.text: {type(response_text_val)}"
                    )
                    logger.debug(
                        f"[GeminiClient][Parse][Image] Value of response.text (first 500 chars): {str(response_text_val)[:500]}"
                    )
                    if isinstance(response_text_val, str):
                        return {"text": response_text_val}
                    else:
                        logger.warning(
                            f"[GeminiClient][Parse][Image] response.text is not a string ({type(response_text_val)}), attempting candidate extraction."
                        )
                        # Fall through to candidate check if response.text is not string

                if hasattr(response, "candidates") and response.candidates:
                    for candidate_idx, candidate in enumerate(response.candidates):
                        # Standard structure: candidate.content.parts[0].text
                        if (
                            hasattr(candidate, "content")
                            and candidate.content
                            and hasattr(candidate.content, "parts")
                            and candidate.content.parts
                            and hasattr(candidate.content.parts[0], "text")
                        ):
                            candidate_text_val = candidate.content.parts[0].text
                            logger.info(
                                f"[GeminiClient][Parse][Image] Found text in candidate {candidate_idx} via content.parts structure."
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Image] Type of candidate_text (content.parts): {type(candidate_text_val)}"
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Image] Value of candidate_text (content.parts, first 500 chars): {str(candidate_text_val)[:500]}"
                            )
                            if isinstance(candidate_text_val, str):
                                return {"text": candidate_text_val}
                            else:
                                logger.warning(
                                    f"[GeminiClient][Parse][Image] Text in candidate {candidate_idx} (content.parts) is not a string: {type(candidate_text_val)}."
                                )
                        # Simpler structure: candidate.text (seen in some non-vision contexts, added for robustness)
                        elif (
                            hasattr(candidate, "text") and candidate.text
                        ):  # Check this path if the above fails for a candidate
                            candidate_text_val = candidate.text
                            logger.info(
                                f"[GeminiClient][Parse][Image] Found text in candidate {candidate_idx} via direct .text attribute."
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Image] Type of candidate.text (direct): {type(candidate_text_val)}"
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Image] Value of candidate.text (direct, first 500 chars): {str(candidate_text_val)[:500]}"
                            )
                            if isinstance(candidate_text_val, str):
                                return {"text": candidate_text_val}
                            else:
                                logger.warning(
                                    f"[GeminiClient][Parse][Image] Text in candidate {candidate_idx} (direct .text) is not a string: {type(candidate_text_val)}."
                                )
                    logger.error(
                        f"[GeminiClient][Parse][Image] No valid string candidate text found after iterating all candidates. Full response: {response}"
                    )
                    return {
                        "error": "No valid string candidate text found in image response",
                        "raw_response": str(response),
                    }
                elif hasattr(
                    response, "error"
                ):  # Check for an explicit error attribute on the response object
                    logger.error(
                        f"[GeminiClient][Parse][Image] Gemini API response has .error attribute: {response.error}"
                    )
                    return {"error": str(response.error), "raw_response": str(response)}
                elif isinstance(
                    response, dict
                ):  # Handle cases where response might already be a dict (e.g. from a failed API call higher up)
                    logger.warning(
                        f"[GeminiClient][Parse][Image] Response is already a dict: {response}"
                    )
                    if "text" in response and isinstance(response["text"], str):
                        return {"text": response["text"]}
                    elif "error" in response:
                        return {
                            "error": response["error"],
                            "raw_response": (
                                str(response)
                                if "raw_response" not in response
                                else response["raw_response"]
                            ),
                        }
                    else:
                        return {
                            "error": "Unexpected dict response from Gemini (image context)",
                            "raw_response": str(response),
                        }
                elif isinstance(
                    response, str
                ):  # Handle if the response is unexpectedly a raw string
                    logger.info(
                        f"[GeminiClient][Parse][Image] Response is already a string of length: {len(response)}"
                    )
                    return {"text": response}
                else:
                    logger.error(
                        f"[GeminiClient][Parse][Image] Invalid/unhandled response format from Gemini: {type(response)} | {response}"
                    )
                    return {
                        "error": "Invalid response format from Gemini (image context)",
                        "raw_response": str(response),
                    }
            except Exception as e:
                logger.error(
                    f"[GeminiClient][DEBUG] Exception during Gemini response parsing: {str(e)}"
                )
                logger.error(traceback.format_exc())
                return {
                    "error": f"Exception during Gemini response parsing: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "raw_response": repr(response),
                }

        except Exception as e:
            logger.error(f"Error generating content with image: {str(e)}")
            return {"error": str(e), "text": f"Error: {str(e)}"}

    def generate_content(
        self, prompt: Union[str, List], use_vision: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Generate content using the Gemini model.

        Args:
            prompt: The prompt to send to the model. Can be a string or a list for multimodal content
            use_vision: Whether to use the vision-capable model for multimodal content
            **kwargs: Additional arguments to pass to the model

        Returns:
            Dictionary containing the generated text response and any error information
        """
        try:
            import logging

            logger = logging.getLogger(__name__)

            # Select the appropriate model based on the content type
            if use_vision:
                model = self.vision_model
            else:
                model = self.default_model

            # Try with the default model first
            try:
                # Handle generation safety settings
                generation_config = kwargs.get("generation_config", {})
                safety_settings = kwargs.get("safety_settings", [])

                # Set default safety settings if none provided
                if not safety_settings:
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                        },
                    ]

                logger.info(f"Sending prompt to Gemini model: {model.model_name}")
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # Extract text from the response based on its structure
                if hasattr(response, "text"):
                    response_text_val = response.text
                    logger.info(
                        "[GeminiClient][Parse][Text] Response has direct text attribute"
                    )
                    logger.debug(
                        f"[GeminiClient][Parse][Text] Type of response.text: {type(response_text_val)}"
                    )
                    logger.debug(
                        f"[GeminiClient][Parse][Text] Value of response.text (first 500 chars): {str(response_text_val)[:500]}"
                    )
                    if isinstance(response_text_val, str):
                        return {"text": response_text_val}
                    else:
                        logger.warning(
                            f"[GeminiClient][Parse][Text] response.text is not a string ({type(response_text_val)}), attempting candidate extraction."
                        )
                        # Fall through to candidate check if response.text is not string

                if hasattr(response, "candidates") and response.candidates:
                    # Assuming the first candidate and first part is what we need for standard text, as per original logic.
                    if (
                        hasattr(response.candidates[0], "content")
                        and response.candidates[0].content
                        and hasattr(response.candidates[0].content, "parts")
                        and response.candidates[0].content.parts
                        and hasattr(response.candidates[0].content.parts[0], "text")
                    ):
                        candidate_text_val = (
                            response.candidates[0].content.parts[0].text
                        )
                        logger.info(
                            "[GeminiClient][Parse][Text] Response has candidates structure (content.parts[0].text)"
                        )
                        logger.debug(
                            f"[GeminiClient][Parse][Text] Type of candidate_text: {type(candidate_text_val)}"
                        )
                        logger.debug(
                            f"[GeminiClient][Parse][Text] Value of candidate_text (first 500 chars): {str(candidate_text_val)[:500]}"
                        )
                        if isinstance(candidate_text_val, str):
                            # Return plain text directly instead of wrapping in a dictionary
                            logger.info(
                                "[GeminiClient][Parse][Text] Returning plain text response from candidate"
                            )
                            return candidate_text_val
                        else:
                            logger.error(
                                f"[GeminiClient][Parse][Text] Candidate text (content.parts[0].text) is not a string: {type(candidate_text_val)}. Full response: {response}"
                            )
                            return {
                                "error": "Candidate text (content.parts[0].text) is not a string",
                                "raw_response": str(response),
                            }
                    else:
                        logger.warning(
                            "[GeminiClient][Parse][Text] Candidate structure present but expected parts (content.parts[0].text) are missing or invalid. Full response: {response}"
                        )
                        # Attempt to check candidate.text directly as a fallback within candidates
                        if (
                            hasattr(response.candidates[0], "text")
                            and response.candidates[0].text
                        ):
                            candidate_text_val = response.candidates[0].text
                            logger.info(
                                "[GeminiClient][Parse][Text] Found text in candidate 0 via direct .text attribute (fallback)."
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Text] Type of candidate.text (direct): {type(candidate_text_val)}"
                            )
                            logger.debug(
                                f"[GeminiClient][Parse][Text] Value of candidate.text (direct, first 500 chars): {str(candidate_text_val)[:500]}"
                            )
                            if isinstance(candidate_text_val, str):
                                return {"text": candidate_text_val}
                        logger.error(
                            "[GeminiClient][Parse][Text] Candidate parts missing or invalid, and no direct .text on candidate[0]. Full response: {response}"
                        )
                        return {
                            "text": "No text content in response",
                            "error": "Candidate parts missing/invalid and no direct text",
                            "raw_response": str(response),
                        }
                else:
                    logger.warning(
                        "[GeminiClient][Parse][Text] No text content found in response via .text or .candidates path. Full response: {response}"
                    )
                    return {
                        "text": "No text content in response",
                        "error": "No text content found",
                        "raw_response": str(response),
                    }

            except Exception as primary_error:
                # If default model fails, try the fallback model
                if not use_vision:  # Only try fallback for text prompts
                    try:
                        logger.warning(
                            f"Default model failed: {str(primary_error)}. Trying fallback model."
                        )
                        response = self.fallback_model.generate_content(
                            prompt, **kwargs
                        )
                        if hasattr(response, "text"):
                            return {"text": response.text}
                        elif hasattr(response, "candidates") and response.candidates:
                            return {
                                "text": response.candidates[0].content.parts[0].text
                            }
                        else:
                            return {
                                "text": "No text content in response",
                                "error": "No text content found in fallback",
                            }
                    except Exception as fallback_error:
                        error_msg = f"Both models failed. Primary error: {str(primary_error)}. Fallback error: {str(fallback_error)}"
                        logger.error(error_msg)
                        return {"text": error_msg, "error": error_msg}
                else:
                    error_msg = f"Vision model failed: {str(primary_error)}"
                    logger.error(error_msg)
                    return {"text": error_msg, "error": error_msg}

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Error generating content: {str(e)}\n{error_details}")
            return {"text": f"Error: {str(e)}", "error": str(e)}

    def analyze_video_content(
        self,
        video_metadata: Dict[str, Any],
        transcript: Optional[str] = None,
        comments: Optional[List[str]] = None,
        thumbnail_image: Optional[Any] = None,
        video_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze video content using the Gemini model.

        Args:
            video_metadata: Dictionary containing video metadata
            transcript: Optional video transcript text
            comments: Optional list of video comments
            thumbnail_image: Optional thumbnail image for multimodal analysis
            video_url: Optional YouTube video URL for reference

        Returns:
            Dictionary containing analysis results
        """
        # Prepare the analysis prompt
        prompt = self._build_analysis_prompt(
            video_metadata, transcript, comments, video_url
        )

        try:
            # Determine if we should use multimodal analysis
            use_vision = thumbnail_image is not None

            if use_vision:
                # For multimodal analysis, combine text prompt with the thumbnail image
                multimodal_prompt = [prompt, thumbnail_image]
                raw_analysis_result = self.generate_content(
                    multimodal_prompt, use_vision=True
                )
            else:
                # For text-only analysis
                raw_analysis_result = self.generate_content(prompt, use_vision=False)

            # Check if there was an error during content generation itself
            if "error" in raw_analysis_result and raw_analysis_result["error"]:
                logger.error(
                    f"Error from generate_content: {raw_analysis_result['error']}"
                )
                return {
                    "error": f"Content generation failed: {raw_analysis_result['error']}"
                }

            # Extract the actual text response to parse
            response_text_to_parse = raw_analysis_result.get("text")
            logger.info(
                f"[GeminiClient][analyze_video_content] Attempting to parse response. Type of response_text_to_parse: {type(response_text_to_parse)}"
            )
            logger.debug(
                f"[GeminiClient][analyze_video_content] Value of response_text_to_parse (first 500 chars): {str(response_text_to_parse)[:500]}"
            )

            if response_text_to_parse is None:
                logger.error("No 'text' field in the result from generate_content.")
                return {"error": "No text content received from content generation."}

            # Parse the response text
            return self._parse_analysis_response(response_text_to_parse)

        except Exception as e:
            logger.error(
                f"[GeminiClient][analyze_video_content] Caught exception during analysis. Type of e: {type(e)}"
            )
            error_message_for_return = (
                "An unexpected error occurred during video content analysis."
            )
            try:
                # Attempt to get a string representation of the exception
                specific_error_details = str(e)
                logger.error(
                    f"[GeminiClient][analyze_video_content] Specific error details: {specific_error_details}",
                    exc_info=True,
                )
                error_message_for_return = (
                    f"Error analyzing video content: {specific_error_details}"
                )
            except Exception as str_conversion_err:
                # If converting the original exception to string fails, log that too
                logger.error(
                    f"[GeminiClient][analyze_video_content] Failed to convert original exception to string: {type(str_conversion_err)} - {str(str_conversion_err)}",
                    exc_info=True,
                )
                logger.error(
                    f"[GeminiClient][analyze_video_content] Original exception type was: {type(e)}",
                    exc_info=True,
                )

            return {"error": error_message_for_return}

    def _build_analysis_prompt(
        self,
        video_metadata: Dict[str, Any],
        transcript: Optional[str],
        comments: Optional[List[str]],
        video_url: Optional[str] = None,
    ) -> str:
        """Build the analysis prompt for the Gemini model.

        Args:
            video_metadata: Video metadata
            transcript: Optional video transcript
            comments: Optional video comments
            video_url: Optional YouTube video URL

        Returns:
            Formatted prompt string
        """
        # Import all schemas at function level to avoid circular imports
        from src.analysis.schemas import (
            ContentQualityAnalysisSchema,
            TitleAnalysisSchema,
            ThumbnailAnalysisSchema,
            SEOAnalysisSchema,
            AudienceEngagementSchema,
            TechnicalPerformanceSchema,
        )

        # Generate schema definitions for all sections
        content_quality_schema_definition = ContentQualityAnalysisSchema.schema_json(
            indent=2
        )
        title_analysis_schema_definition = TitleAnalysisSchema.schema_json(indent=2)
        thumbnail_analysis_schema_definition = ThumbnailAnalysisSchema.schema_json(
            indent=2
        )
        seo_analysis_schema_definition = SEOAnalysisSchema.schema_json(indent=2)
        audience_engagement_schema_definition = AudienceEngagementSchema.schema_json(
            indent=2
        )
        technical_performance_schema_definition = (
            TechnicalPerformanceSchema.schema_json(indent=2)
        )

        prompt_parts = [
            "# YouTube Video Intelligence Analysis\n\n",
            "## 📹 Video Verification (FROM API DATA)\n",
            f"**Title**: {video_metadata.get('title', 'N/A')}\n",
            f"**Channel**: {video_metadata.get('channel_title', 'N/A')}\n",
            f"**Duration**: {video_metadata.get('duration', 'N/A')}\n",
            f"**Views**: {video_metadata.get('view_count', 'N/A')}\n",
            f"**Published**: {video_metadata.get('published_at', 'N/A')}\n",
        ]

        # Add video URL with emphasis if available
        if video_url:
            prompt_parts.extend(
                [
                    f"**URL**: {video_url}\n",
                    "**IMPORTANT**: Please access this video URL to analyze the actual content for accurate evaluation.\n",
                    "You must watch at least parts of the video to provide accurate analysis of content quality and technical aspects.\n",
                ]
            )
        else:
            prompt_parts.append(f"**URL**: N/A\n")

        prompt_parts.append(f"**Data Source**: youtube_api\n\n")

        if transcript:
            prompt_parts.extend(
                [
                    "## Video Transcript\n",
                    f"{transcript[:5000]}...\n\n",  # Limit transcript length
                ]
            )

        if comments:
            prompt_parts.append("## Top Comments\n")
            for i, comment in enumerate(comments[:5], 1):  # Limit to top 5 comments
                prompt_parts.append(
                    f"{i}. {comment[:200]}...\n"
                )  # Truncate long comments
            prompt_parts.append("\n")

        # Add analysis instructions based on the system prompt framework
        prompt_parts.extend(
            [
                "## Analysis Instructions\n\n",
                "Analyze this YouTube video following the framework below. Use ONLY the provided metadata as ground truth for factual information.\n\n",
                "CRITICAL OUTPUT FORMATTING INSTRUCTIONS:\n",
                "- For each analysis section, the detailed JSON output MUST start with a JSON code block with no text before it\n",
                "- Always use ```json at the beginning and ``` at the end to delimit each JSON block\n",
                "- NEVER include any explanatory text before the JSON block - start directly with ```json\n",
                "- Ensure all JSON properties match exactly what's defined in the schema\n\n",
                "### 1. 📋 CONTENT SUMMARIZATION\n",
                "Provide a comprehensive 2-3 paragraph summary covering:\n",
                "- Main topics, themes, and key messages\n",
                "- Target audience and content type identification\n",
                "- Unique value proposition and standout elements\n\n",
                "### 2. 📊 STRUCTURED ANALYSIS & SCORING\n",
                "Score each factor 1-5 (5 = excellent) with detailed explanations. For EACH factor below, provide your analysis as a structured JSON object delimited by ```json and ``` tags:\n\n",
            ]
        )

        # Hook (Clickability) section with schema - combines Title and Thumbnail analysis
        prompt_parts.extend(
            [
                "**🎣 Hook (Clickability) (X/5)**: Provide your overall score (1-5) for how effectively the video attracts clicks through its title and thumbnail. THEN, provide a DETAILED BREAKDOWN of the title analysis as a JSON object adhering to the following schema:\n"
                + f"```json_schema\n{title_analysis_schema_definition}\n```\n\n",
                "THEN, provide a DETAILED BREAKDOWN of the thumbnail analysis as a JSON object adhering to the following schema:\n"
                + f"```json_schema\n{thumbnail_analysis_schema_definition}\n```\n\n",
            ]
        )

        # SEO Optimization section with schema
        prompt_parts.extend(
            [
                "**🔍 SEO Optimization (X/5)**: Provide your overall score (1-5). THEN, provide a DETAILED BREAKDOWN and justification for this score as a JSON object adhering to the following Pydantic schema. The JSON object for detailed breakdown is:\n"
                + f"```json_schema\n{seo_analysis_schema_definition}\n```\n\n",
            ]
        )

        # Content Quality section with schema
        prompt_parts.extend(
            [
                "**📝 Content Quality (X/5)**: Provide your overall score (1-5). THEN, provide a DETAILED BREAKDOWN and justification for this score as a JSON object adhering to the following Pydantic schema. The JSON object for detailed breakdown is:\n"
                + f"```json_schema\n{content_quality_schema_definition}\n```\n\n",
            ]
        )

        # Audience Engagement section with schema
        prompt_parts.extend(
            [
                "**👥 Audience Engagement (X/5)**: Provide your overall score (1-5) for audience engagement. Provide your overall score (1-5). THEN, provide a DETAILED BREAKDOWN and justification for this score as a JSON object adhering to the following Pydantic schema. The JSON object for detailed breakdown is:\n"
                + f"```json_schema\n{audience_engagement_schema_definition}\n```\n\n",
            ]
        )

        # Technical Performance section with schema
        prompt_parts.extend(
            [
                "**⚙️ Technical Performance (X/5)**: Provide your overall score (1-5) for technical quality. Provide your overall score (1-5). THEN, provide a DETAILED BREAKDOWN and justification for this score as a JSON object adhering to the following Pydantic schema. The JSON object for detailed breakdown is:\n"
                + f"```json_schema\n{technical_performance_schema_definition}\n```\n\n",
            ]
        )

        prompt_parts.extend(
            [
                "### 3. 🎯 IMPROVEMENT RECOMMENDATIONS\n\n",
            ]
        )

        # Other recommendations sections
        prompt_parts.extend(
            [
                "**Content Improvements**: Structure, pacing, production quality, accessibility suggestions\n",
                "**Engagement Optimization**: Hook improvements, CTA placement, interaction strategies\n\n",
                "### 4. 💡 ACTION PLAN\n",
                "**Quick Wins**: Immediate improvements\n",
                "**Medium-term**: Next 3 videos strategy\n",
                "**Long-term**: Channel growth tactics\n\n",
                "Format your response in Markdown with clear section headers. For JSON objects, ensure they are properly formatted and delimited with ```json and ``` tags.\n\n",
                "IMPORTANT INSTRUCTIONS:\n",
                "1. All metadata in Video Verification section MUST come from the provided API data, not from your own analysis.\n",
                "2. For each analysis section (Hook, SEO, Content Quality, Audience Engagement, Technical), provide both a clear score out of 5 AND a properly formatted JSON object with detailed analysis.\n",
                "3. Each JSON response MUST conform exactly to the schema provided for that section.\n",
            ]
        )

        return "".join(prompt_parts)

        return "".join(prompt_parts)

    def classify_user_prompt(
        self, prompt: str, video_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify a user prompt to determine the appropriate response strategy.

        Args:
            prompt: User's text prompt/question
            video_metadata: Dictionary containing video metadata

        Returns:
            Dictionary containing classification results with:
            - query_type: 'metadata', 'content_analysis', or 'general'
            - specific_data_needed: List of specific data fields needed (for metadata queries)
            - requires_ai_analysis: Whether AI analysis is needed
            - requires_thumbnail_analysis: Whether thumbnail analysis is needed
        """
        classification_prompt = f"""
        You are an AI assistant that classifies user queries about YouTube videos to determine the appropriate response strategy.
        
        USER QUERY: "{prompt}"
        
        AVAILABLE VIDEO METADATA: {', '.join(video_metadata.keys())}
        
        Classify this query into one of these categories:
        1. METADATA: Query asking for factual information directly available in the video metadata (title, channel, views, likes, etc.)
        2. CONTENT_ANALYSIS: Query requiring understanding or analysis of the video content, context, or thumbnail
        3. GENERAL: General question not specific to metadata or content analysis
        
        For METADATA queries, specify which exact metadata fields are needed.
        For CONTENT_ANALYSIS queries, specify whether thumbnail analysis is needed.
        
        Return your analysis as a JSON object with these fields:
        - query_type: "metadata", "content_analysis", or "general"
        - specific_data_needed: [list of specific metadata fields needed] (for metadata queries)
        - requires_ai_analysis: true/false (whether AI understanding is needed)
        - requires_thumbnail_analysis: true/false (whether thumbnail analysis is needed)
        - reasoning: brief explanation of your classification
        """

        try:
            # Use the text model for classification
            response = self.generate_content(classification_prompt, use_vision=False)

            # Extract JSON from the response
            import re
            import json

            # Look for JSON pattern in the response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    if not isinstance(json_str, str):
                        logger.error(
                            f"[GeminiClient] classification json_str is not a string! type: {type(json_str)}, repr: {repr(json_str)[:500]}"
                        )
                        return {
                            "error": "classification json_str is not a string",
                            "type": str(type(json_str)),
                            "repr": repr(json_str)[:500],
                        }
                    try:
                        classification = json.loads(json_str)
                    except Exception as e:
                        import traceback

                        logger.error(
                            f"[GeminiClient] Exception parsing classification json_str: {str(e)}"
                        )
                        logger.error(traceback.format_exc())
                        return {
                            "error": f"Exception parsing classification json_str: {str(e)}",
                            "traceback": traceback.format_exc(),
                            "type": str(type(json_str)),
                            "repr": repr(json_str)[:500],
                        }

                    return classification
                except json.JSONDecodeError:
                    # If JSON parsing fails, return a default classification
                    return {
                        "query_type": "general",
                        "specific_data_needed": [],
                        "requires_ai_analysis": True,
                        "requires_thumbnail_analysis": False,
                        "reasoning": "Failed to parse classification JSON",
                    }
            else:
                # If no JSON found, return a default classification
                return {
                    "query_type": "general",
                    "specific_data_needed": [],
                    "requires_ai_analysis": True,
                    "requires_thumbnail_analysis": False,
                    "reasoning": "No classification JSON found in response",
                }

        except Exception as e:
            # If classification fails, return a default classification
            return {
                "query_type": "general",
                "specific_data_needed": [],
                "requires_ai_analysis": True,
                "requires_thumbnail_analysis": False,
                "reasoning": f"Classification error: {str(e)}",
            }

    def _extract_and_validate_json(
        self, section_text: str, schema_class, key: str, result_dict: Dict[str, Any]
    ) -> None:
        """Extract JSON from section text and validate with Pydantic schema.

        Args:
            section_text: Text containing JSON block
            schema_class: Pydantic schema class for validation
            key: Key to store validated data under in result_dict
            result_dict: Dictionary to store results in
        """
        logger.debug(
            f"[GeminiClient][Parse][{key.capitalize()}] Attempting to parse detailed JSON. Section text length: {len(section_text)}"
        )

        # First try to find the JSON block delimited with ```json ... ```
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", section_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1).strip()
            logger.debug(
                f"[GeminiClient][Parse][{key.capitalize()}] Extracted JSON string (first 100 chars): {json_string[:100]}"
            )
            try:
                parsed_json = json.loads(json_string)
                logger.debug(
                    f"[GeminiClient][Parse][{key.capitalize()}] Successfully parsed JSON string. Type: {type(parsed_json)}"
                )
                validated_data = schema_class.parse_obj(parsed_json)
                result_dict["analysis"][f"{key}_analysis"] = validated_data.dict()

                # If score is present, also set it in the scores dict for consistency
                if (
                    hasattr(validated_data, "score")
                    and validated_data.score is not None
                ):
                    score_key = key if key != "content_quality" else "content_quality"
                    result_dict["scores"][score_key] = float(validated_data.score)
                    logger.info(
                        f"[GeminiClient][Parse][{key.capitalize()}] Set {score_key} score to {validated_data.score}"
                    )

                logger.info(
                    f"[GeminiClient][Parse][{key.capitalize()}] Successfully parsed and validated detailed JSON"
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"[GeminiClient][Parse][{key.capitalize()}] Failed to decode JSON: {e}\nJSON string (first 100 chars): {json_string[:100]}"
                )
                result_dict["analysis"][f"{key}_analysis"] = {
                    "error": f"JSONDecodeError: {str(e)}",
                    "json_sample": (
                        json_string[:500] if json_string else "Empty JSON string"
                    ),
                }
            except ValidationError as e:
                # Ensure parsed_json is defined for the error message
                data_for_error_log = (
                    str(parsed_json)[:200]
                    if "parsed_json" in locals()
                    else json_string[:200]
                )
                logger.error(
                    f"[GeminiClient][Parse][{key.capitalize()}] Failed to validate JSON against schema: {str(e)}\nData sample: {data_for_error_log}"
                )
                error_payload = {
                    "error": f"ValidationError: {str(e.errors())}",
                    "schema": schema_class.__name__,
                }
                if "parsed_json" in locals():
                    error_payload["data_received"] = parsed_json
                else:
                    error_payload["raw_json_string_attempted"] = json_string[:500]
                result_dict["analysis"][f"{key}_analysis"] = error_payload
            except Exception as e:
                logger.error(
                    f"[GeminiClient][Parse][{key.capitalize()}] An unexpected error occurred: {str(e)}"
                )
                result_dict["analysis"][f"{key}_analysis"] = {
                    "error": f"Unexpected error: {str(e)}",
                    "error_type": type(e).__name__,
                }
        else:
            # If no code block found, try a more aggressive approach to find JSON
            # Look for a standalone JSON object possibly after non-JSON text
            logger.warning(
                f"[GeminiClient][Parse][{key.capitalize()}] No ```json``` block found, trying to find JSON object directly."
            )

            # Try to find a JSON object starting with { and ending with }
            json_direct_match = re.search(
                r"(\{[\s\S]*?\})(?=\s*$|\s*\n\s*###|\s*\n\s*```)", section_text
            )

            if json_direct_match:
                json_string = json_direct_match.group(1).strip()
                logger.debug(
                    f"[GeminiClient][Parse][{key.capitalize()}] Found potential direct JSON object (first 100 chars): {json_string[:100]}"
                )

                try:
                    parsed_json = json.loads(json_string)
                    logger.debug(
                        f"[GeminiClient][Parse][{key.capitalize()}] Successfully parsed direct JSON. Type: {type(parsed_json)}"
                    )
                    validated_data = schema_class.parse_obj(parsed_json)
                    result_dict["analysis"][f"{key}_analysis"] = validated_data.dict()

                    # If score is present, also set it in the scores dict for consistency
                    if (
                        hasattr(validated_data, "score")
                        and validated_data.score is not None
                    ):
                        score_key = (
                            key if key != "content_quality" else "content_quality"
                        )
                        result_dict["scores"][score_key] = float(validated_data.score)
                        logger.info(
                            f"[GeminiClient][Parse][{key.capitalize()}] Set {score_key} score to {validated_data.score} (from direct JSON)"
                        )

                    logger.info(
                        f"[GeminiClient][Parse][{key.capitalize()}] Successfully parsed and validated direct JSON"
                    )
                except (json.JSONDecodeError, ValidationError, Exception) as e:
                    logger.error(
                        f"[GeminiClient][Parse][{key.capitalize()}] Failed to parse direct JSON object: {type(e).__name__}: {str(e)}"
                    )
                    # Fall through to the final error case
                    result_dict["analysis"][f"{key}_analysis"] = {
                        "error": f"Failed to parse JSON: {type(e).__name__}: {str(e)}",
                        "section_text_sample": (
                            section_text[:200] if section_text else "Empty section text"
                        ),
                    }
            else:
                logger.warning(
                    f"[GeminiClient][Parse][{key.capitalize()}] Could not find any JSON object in section text."
                )
                result_dict["analysis"][f"{key}_analysis"] = {
                    "error": "Detailed JSON block not found in Gemini response",
                    "section_text_sample": (
                        section_text[:200] if section_text else "Empty section text"
                    ),
                }

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured analysis response from Gemini.

        Args:
            response: Markdown-formatted response from Gemini

        Returns:
            Dictionary containing parsed analysis sections
        """
        # Initialize the result structure
        result = {
            "summary": "",
            "scores": {
                "hook": 0,
                "title": 0,  # Added to support hook score calculation in ScoringEngine
                "thumbnail": 0,  # Added to support hook score calculation in ScoringEngine
                "content_quality": 0,
                "seo_optimization": 0,
                "audience_engagement": 0,
                "technical_performance": 0,
                "overall_score": 0,
            },
            "analysis": {
                "hook": "",
                "content_quality": "",
                "seo_optimization": "",
                "audience_engagement": "",
                "technical_performance": "",
            },
            "recommendations": {
                "hook_optimization": [],
                "content_improvements": [],
                "seo_optimization": [],
                "audience_engagement": [],
                "technical_improvements": [],
            },
            "action_plan": {"quick_wins": [], "medium_term": [], "long_term": []},
            "raw_response": response,  # Store the full response for reference
        }

        # Extract scores using regex

        # Parse hook score - we'll let the scoring engine calculate this from title + thumbnail
        # Still try to extract a provided hook score from Gemini if available
        hook_match = re.search(r"Hook \(Clickability\)[^\d]*(\d+)[^\d]*5", response)
        if hook_match:
            # Store the raw hook score from Gemini, but this will be recalculated by ScoringEngine
            result["scores"]["hook"] = int(hook_match.group(1))

        # Also try to extract separate title and thumbnail scores if provided
        title_score_match = re.search(r"Title Analysis[^\d]*(\d+)[^\d]*5", response)
        if title_score_match:
            result["scores"]["title"] = int(title_score_match.group(1))

        thumbnail_score_match = re.search(
            r"Thumbnail Analysis[^\d]*(\d+)[^\d]*5", response
        )
        if thumbnail_score_match:
            result["scores"]["thumbnail"] = int(thumbnail_score_match.group(1))

        # Parse SEO optimization score
        seo_match = re.search(r"SEO Optimization[^\d]*(\d+)[^\d]*5", response)
        if seo_match:
            result["scores"]["seo_optimization"] = int(seo_match.group(1))

        # Parse content quality score
        content_quality_match = re.search(
            r"Content Quality[^\d]*(\d+)[^\d]*5", response
        )
        if content_quality_match:
            result["scores"]["content_quality"] = int(content_quality_match.group(1))

        # Parse audience engagement score
        engagement_match = re.search(r"Audience Engagement[^\d]*(\d+)[^\d]*5", response)
        if engagement_match:
            result["scores"]["audience_engagement"] = int(engagement_match.group(1))

        # Parse technical performance score
        technical_match = re.search(
            r"Technical Performance[^\d]*(\d+)[^\d]*5", response
        )
        if technical_match:
            result["scores"]["technical_performance"] = int(technical_match.group(1))

        # Calculate overall score (average of all scores)
        scores = [v for v in result["scores"].values() if v > 0]
        if scores:
            result["scores"]["overall_score"] = round(
                sum(scores) / len(scores) * 2, 1
            )  # Convert to scale of 10

        # Extract summary (look for content summarization section)
        summary_match = re.search(
            r"CONTENT SUMMARIZATION.*?\n(.*?)(?=###|$)", response, re.DOTALL
        )
        if summary_match:
            result["summary"] = summary_match.group(1).strip()

        # Extract analysis sections
        for section, key in [
            (r"Hook \(Clickability\).*?(?=\*\*SEO|###)", "hook"),
            (r"SEO Optimization.*?(?=\*\*Content|###)", "seo_optimization"),
            (r"Content Quality.*?(?=\*\*Audience|###)", "content_quality"),
            (r"Audience Engagement.*?(?=\*\*Technical|###)", "audience_engagement"),
            (r"Technical Performance.*?(?=###|$)", "technical_performance"),
        ]:
            match = re.search(section, response, re.DOTALL)
            if match:
                section_text = match.group(0).strip()
                result["analysis"][key] = section_text

                # Import necessary schemas here to avoid circular imports
                from src.analysis.schemas import (
                    ContentQualityAnalysisSchema,
                    TitleAnalysisSchema,
                    ThumbnailAnalysisSchema,
                    SEOAnalysisSchema,
                    AudienceEngagementSchema,
                    TechnicalPerformanceSchema,
                )

                # Use the helper function to extract and validate JSON for each section
                if key == "hook":
                    # For hook, we need to check for both title and thumbnail JSON blocks
                    logger.info(
                        f"[GeminiClient][Parse][Hook] Processing hook section with both title and thumbnail analysis"
                    )

                    # First look for title analysis JSON
                    title_section_match = re.search(
                        r"DETAILED BREAKDOWN of the title analysis.*?(?=THEN|```json_schema|$)",
                        section_text,
                        re.DOTALL,
                    )
                    if title_section_match:
                        title_section = title_section_match.group(0)
                        logger.debug(
                            f"[GeminiClient][Parse][Hook] Found title analysis section (first 100 chars): {title_section[:100]}"
                        )
                        self._extract_and_validate_json(
                            title_section, TitleAnalysisSchema, "title", result
                        )
                    else:
                        logger.warning(
                            f"[GeminiClient][Parse][Hook] Could not find title analysis section in hook analysis"
                        )

                    # Then look for thumbnail analysis JSON
                    thumbnail_section_match = re.search(
                        r"DETAILED BREAKDOWN of the thumbnail analysis.*?(?=###|$)",
                        section_text,
                        re.DOTALL,
                    )
                    if thumbnail_section_match:
                        thumbnail_section = thumbnail_section_match.group(0)
                        logger.debug(
                            f"[GeminiClient][Parse][Hook] Found thumbnail analysis section (first 100 chars): {thumbnail_section[:100]}"
                        )
                        self._extract_and_validate_json(
                            thumbnail_section,
                            ThumbnailAnalysisSchema,
                            "thumbnail",
                            result,
                        )
                    else:
                        logger.warning(
                            f"[GeminiClient][Parse][Hook] Could not find thumbnail analysis section in hook analysis"
                        )

                    # If we have both title and thumbnail analysis with scores, calculate combined hook score
                    title_score = float(
                        result["analysis"].get("title_analysis", {}).get("score", 0.0)
                    )
                    thumbnail_score = float(
                        result["analysis"]
                        .get("thumbnail_analysis", {})
                        .get("score", 0.0)
                    )

                    if title_score > 0 and thumbnail_score > 0:
                        # Calculate combined hook score (50% title + 50% thumbnail)
                        hook_score = (title_score + thumbnail_score) / 2.0
                        logger.info(
                            f"[GeminiClient][Parse][Hook] Calculated hook score: {hook_score} from title: {title_score} and thumbnail: {thumbnail_score}"
                        )

                        # Update the hook score in the result (will be overwritten by ScoringEngine later)
                        result["scores"]["title"] = title_score
                        result["scores"]["thumbnail"] = thumbnail_score
                        result["scores"]["hook"] = hook_score

                elif key == "seo_optimization":
                    self._extract_and_validate_json(
                        section_text, SEOAnalysisSchema, "seo_optimization", result
                    )

                elif key == "content_quality":
                    self._extract_and_validate_json(
                        section_text,
                        ContentQualityAnalysisSchema,
                        "content_quality",
                        result,
                    )

                elif key == "audience_engagement":
                    self._extract_and_validate_json(
                        section_text,
                        AudienceEngagementSchema,
                        "audience_engagement",
                        result,
                    )
                    # Ensure the score key matches
                    if "audience_engagement_analysis" in result["analysis"] and "score" in result["analysis"]["audience_engagement_analysis"]:
                        result["scores"]["audience_engagement"] = float(result["analysis"]["audience_engagement_analysis"]["score"])

                elif key == "technical_performance":
                    self._extract_and_validate_json(
                        section_text,
                        TechnicalPerformanceSchema,
                        "technical_performance",
                        result,
                    )

        # Extract recommendations
        # Extract hook recommendations (combining title and thumbnail)
        hook_match = re.search(
            r"Hook Optimization.*?(?=\*\*Content|###)", response, re.DOTALL
        )
        if hook_match:
            hook_recs = re.findall(r'- ["\']?(.*?)["\']?\n', hook_match.group(0))
            result["recommendations"]["hook_optimization"] = [
                t.strip() for t in hook_recs if t.strip()
            ]

        content_match = re.search(
            r"Content Improvements.*?(?=\*\*SEO|\*\*Audience|###)", response, re.DOTALL
        )
        if content_match:
            content_impr = re.findall(r"- (.*?)\n", content_match.group(0))
            result["recommendations"]["content_improvements"] = [
                c.strip() for c in content_impr if c.strip()
            ]

        seo_match = re.search(
            r"SEO Optimization.*?(?=\*\*Audience|###)", response, re.DOTALL
        )
        if seo_match:
            seo_impr = re.findall(r"- (.*?)\n", seo_match.group(0))
            result["recommendations"]["seo_optimization"] = [
                s.strip() for s in seo_impr if s.strip()
            ]

        engagement_match = re.search(
            r"Audience Engagement.*?(?=\*\*Technical|###)", response, re.DOTALL
        )
        if engagement_match:
            engagement_impr = re.findall(r"- (.*?)\n", engagement_match.group(0))
            result["recommendations"]["audience_engagement"] = [
                e.strip() for e in engagement_impr if e.strip()
            ]

        technical_match = re.search(
            r"Technical Improvements.*?(?=###|$)", response, re.DOTALL
        )
        if technical_match:
            technical_impr = re.findall(r"- (.*?)\n", technical_match.group(0))
            result["recommendations"]["technical_improvements"] = [
                t.strip() for t in technical_impr if t.strip()
            ]

        # Extract action plan
        quick_wins_match = re.search(
            r"Quick Wins.*?(?=\*\*Medium|###)", response, re.DOTALL
        )
        if quick_wins_match:
            quick_wins = re.findall(r"- (.*?)\n", quick_wins_match.group(0))
            result["action_plan"]["quick_wins"] = [
                w.strip() for w in quick_wins if w.strip()
            ]

        medium_match = re.search(r"Medium-term.*?(?=\*\*Long|###)", response, re.DOTALL)
        if medium_match:
            medium_term = re.findall(r"- (.*?)\n", medium_match.group(0))
            result["action_plan"]["medium_term"] = [
                m.strip() for m in medium_term if m.strip()
            ]

        long_match = re.search(r"Long-term.*?(?=###|$)", response, re.DOTALL)
        if long_match:
            long_term = re.findall(r"- (.*?)\n", long_match.group(0))
            result["action_plan"]["long_term"] = [
                l.strip() for l in long_term if l.strip()
            ]

        return result

    def generate_score_with_guidelines(
        self,
        score_type: str,
        video_data: Dict[str, Any],
        guidelines: Dict[str, str],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a score and analysis based on guidelines and video data.

        Args:
            score_type: Type of score to generate (hook, content, seo, technical)
            video_data: Dictionary containing video metadata and content
            guidelines: Dictionary containing guideline documents
            generation_config: Optional configuration for generation

        Returns:
            Dictionary with score, analysis, and recommendations
        """
        try:
            # Set default generation config if not provided
            if not generation_config:
                generation_config = {
                    "temperature": 0.1,  # Lower temperature for more consistent scoring
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                }

            # Prepare the prompt based on score type
            prompt = f"""You are an expert YouTube video analyst. Your task is to evaluate the {score_type} of a YouTube video and provide a score from 1-5 (where 5 is excellent and 1 is poor).

Here is the video data to analyze:

Title: {video_data.get('title', 'N/A')}
Description: {video_data.get('description', 'N/A')}
Tags: {', '.join(video_data.get('tags', []))}
Duration: {video_data.get('duration', 'N/A')}
Definition: {video_data.get('definition', 'N/A')}
"""

            # Add video URL if available for content access
            if "url" in video_data:
                prompt += f"\nVideo URL: {video_data['url']}\n"
                prompt += "\nIMPORTANT: Please access the video URL to analyze the actual content for a more accurate evaluation.\n"

            # Add video ID if available but URL is not
            elif "video_id" in video_data:
                video_url = f"https://www.youtube.com/watch?v={video_data['video_id']}"
                prompt += f"\nVideo URL: {video_url}\n"
                prompt += "\nIMPORTANT: Please access the video URL to analyze the actual content for a more accurate evaluation.\n"

            prompt += "\n"

            # Add relevant guidelines based on score type
            if "metrics_methodology" in guidelines:
                prompt += f"\nHere are the scoring criteria from our methodology document:\n{guidelines['metrics_methodology']}\n"

            if "general_guidelines" in guidelines:
                prompt += f"\nHere are our general guidelines for YouTube content:\n{guidelines['general_guidelines']}\n"

            # Add specific instructions based on score type
            if score_type.lower() == "hook":
                prompt += "\nFocus on evaluating the hook quality (title and thumbnail effectiveness). Consider how well they work together to attract clicks and viewer attention."
            elif score_type.lower() == "content":
                prompt += "\nFocus on evaluating the content quality. Consider production value, information quality, and viewer value."
            elif score_type.lower() == "seo":
                prompt += "\nFocus on evaluating the SEO optimization. Consider keyword usage, metadata completeness, and search relevance."
            elif score_type.lower() == "technical":
                prompt += "\nFocus on evaluating the technical quality. Consider video quality, audio quality, and platform optimization."

            # Request structured output
            prompt += """

Provide your response in the following JSON format:
{
  "score": [a number between 1.0 and 5.0],
  "analysis": [detailed analysis explaining the score],
  "strengths": [list of strengths],
  "weaknesses": [list of weaknesses],
  "recommendations": [list of specific improvement recommendations]
}

Ensure your score is a decimal between 1.0 and 5.0, where:
- 5.0: Excellent
- 4.0-4.9: Very Good
- 3.0-3.9: Good
- 2.0-2.9: Below Average
- 1.0-1.9: Poor

Be objective and thorough in your analysis.
"""

            # Generate the response
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Generating {score_type} score with guidelines")
            response = self.generate_content(prompt, generation_config)

            if response and "error" not in response:
                # Parse the response - expecting JSON format
                try:
                    import json
                    import re

                    # Handle different response types (string or object with text attribute)
                    if isinstance(response, str):
                        response_text = response
                    elif hasattr(response, "text"):
                        response_text = response.text
                    elif isinstance(response, dict) and "text" in response:
                        response_text = response["text"]
                    else:
                        # Try to convert to string as a last resort
                        response_text = str(response)

                    # Try to find JSON in markdown code blocks
                    if "```json" in response_text:
                        json_content = (
                            response_text.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in response_text:
                        json_content = (
                            response_text.split("```")[1].split("```")[0].strip()
                        )
                    else:
                        # Look for JSON-like structure with regex
                        json_match = re.search(
                            r'\{[\s\S]*?"score"[\s\S]*?\}', response_text
                        )
                        if json_match:
                            json_content = json_match.group(0)
                        else:
                            json_content = response_text

                    # Clean up the JSON content
                    json_content = json_content.strip()

                    # Try to parse JSON
                    if not isinstance(json_content, str):
                        logger.error(
                            f"[GeminiClient] score_analysis json_content is not a string! type: {type(json_content)}, repr: {repr(json_content)[:500]}"
                        )
                        return {
                            "error": "score_analysis json_content is not a string",
                            "type": str(type(json_content)),
                            "repr": repr(json_content)[:500],
                        }
                    try:
                        score_analysis = json.loads(json_content)
                    except Exception as e:
                        import traceback

                        logger.error(
                            f"[GeminiClient] Exception parsing score_analysis json_content: {str(e)}"
                        )
                        logger.error(traceback.format_exc())
                        return {
                            "error": f"Exception parsing score_analysis json_content: {str(e)}",
                            "traceback": traceback.format_exc(),
                            "type": str(type(json_content)),
                            "repr": repr(json_content)[:500],
                        }

                    # Ensure score is within 1-5 range
                    if "score" in score_analysis:
                        score_analysis["score"] = min(
                            5.0, max(1.0, float(score_analysis["score"]))
                        )
                    else:
                        score_analysis["score"] = 3.0  # Default score

                    return score_analysis

                except Exception as e:
                    logger.error(
                        f"Error parsing {score_type} score analysis response: {str(e)}"
                    )
                    # If JSON parsing fails, create a structured response with the raw text
                    response_text = str(response) if response else ""
                    return {
                        "score": 3.0,  # Default score
                        "analysis": f"Analysis generated (parsing error): {response_text[:200]}...",
                        "strengths": ["Unable to parse strengths due to format error"],
                        "weaknesses": [
                            "Unable to parse weaknesses due to format error"
                        ],
                        "recommendations": [
                            "Try again or adjust the prompt for better structured output"
                        ],
                    }
            else:
                # Handle case when there's no valid response
                error_msg = "No response" if not response else "Unknown error"
                if isinstance(response, dict) and "error" in response:
                    error_msg = response["error"]

                logger.error(
                    f"Error from Gemini for {score_type} score analysis: {error_msg}"
                )

                # Return a structured response with error information
                return {
                    "score": 3.0,  # Default score
                    "analysis": f"Unable to generate analysis: {error_msg}",
                    "strengths": ["Analysis unavailable due to API error"],
                    "weaknesses": ["Analysis unavailable due to API error"],
                    "recommendations": ["Try again later or check API configuration"],
                }

        except Exception as e:
            logger.error(f"Error in {score_type} score analysis: {str(e)}")
            return {
                "error": f"{score_type} score analysis failed: {str(e)}",
                "score": 3.0,
            }


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize the client
    client = GeminiClient()

    # Example metadata (in a real app, this would come from YouTube API)
    video_metadata = {
        "title": "How to Build a YouTube Chatbot with AI",
        "channel_title": "AI Tutorials",
        "published_at": "2025-01-15T14:30:00Z",
        "view_count": "10,245",
        "like_count": "1,234",
        "comment_count": "89",
        "video_id": "dQw4w9WgXcQ",  # Example video ID
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example video URL
    }

    # Example transcript (truncated for brevity)
    transcript = """
    In this video, we'll build a YouTube chatbot using AI. First, we'll set up our 
    development environment, then we'll integrate with the YouTube API, and finally 
    we'll add AI-powered analysis using Google's Gemini model.
    """

    # Example comments
    comments = [
        "Great tutorial! Very clear and easy to follow.",
        "Could you make a follow-up video on advanced features?",
        "I had some trouble with the API setup, any tips?",
        "This is exactly what I was looking for, thanks!",
        "The AI analysis part was particularly interesting.",
    ]

    # Analyze the video - explicitly passing the video URL for content access
    analysis = client.analyze_video_content(
        video_metadata=video_metadata,
        transcript=transcript,
        comments=comments,
        video_url=video_metadata.get("url"),  # Explicitly pass the video URL
    )
    print("Video Analysis:")
    for section, content in analysis.items():
        print(f"\n## {section.upper()}")
        if isinstance(content, list):
            for item in content:
                print(f"- {item}")
        else:
            print(content)

    # Example of using the generate_score_with_guidelines method with video URL
    print("\n\nGenerating Content Quality Score:")
    guidelines = {
        "metrics_methodology": "Content quality should be scored based on information accuracy, production value, and viewer value.",
        "general_guidelines": "High-quality content is engaging, informative, and well-produced.",
    }

    # Score content quality with access to the video URL
    content_score = client.generate_score_with_guidelines(
        score_type="content",
        video_data=video_metadata,  # Contains the video URL
        guidelines=guidelines,
    )

    print(f"Content Quality Score: {content_score.get('score', 'N/A')}/5.0")
    print(f"Analysis: {content_score.get('analysis', 'N/A')[:100]}...")
    print("Recommendations:")
    for rec in content_score.get("recommendations", [])[:3]:
        print(f"- {rec}")
