"""
Extension to GeminiClient for guideline-based scoring of YouTube videos.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


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
        logger.info(f"Generating {score_type} score with guidelines")
        response = self.generate_content(prompt, generation_config)

        if response and "error" not in response:
            # Parse the response - expecting JSON format
            try:
                # Extract JSON from the response if it contains markdown code blocks
                response_text = response.get("text", "")

                # Try to find JSON in markdown code blocks
                if "```json" in response_text:
                    json_content = (
                        response_text.split("```json")[1].split("```", 1)[0].strip()
                    )
                elif "```" in response_text:
                    json_content = (
                        response_text.split("```", 1)[1].split("```", 1)[0].strip()
                    )
                else:
                    json_content = response_text

                if not isinstance(json_content, str):
                    logger.error(
                        f"[GuidelineScorer] score_analysis json_content is not a string! type: {type(json_content)}, repr: {repr(json_content)[:500]}"
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
                        f"[GuidelineScorer] Exception parsing score_analysis json_content: {str(e)}"
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
                    f"Error parsing response for {score_type} score analysis: {str(e)}"
                )
                return {
                    "error": f"Error parsing response for {score_type} score analysis: {str(e)}",
                    "score": 3.0,
                }
        else:
            error = (
                response.get("error", "Unknown error") if response else "No response"
            )
            logger.error(f"Error from Gemini for {score_type} score analysis: {error}")
            return {"error": error, "score": 3.0}

    except Exception as e:
        logger.error(f"Error in {score_type} score analysis: {str(e)}")
        return {"error": f"{score_type} score analysis failed: {str(e)}", "score": 3.0}
