"""Scoring engine for YouTube video analysis.

This module provides functionality for scoring YouTube videos based on various factors
including hook quality, content quality, SEO optimization, engagement metrics, and technical quality.
It uses a Retrieval-Augmented Generation (RAG) approach with the Gemini model to generate scores
based on guideline documents.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import logging
from pathlib import Path
import os

from src.api.gemini_client import GeminiClient
from src.utils.guideline_loader import load_guidelines, get_category_guidelines

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Scores YouTube videos based on a 5-factor scoring system using Gemini and guidelines.

    The scoring factors are:
    1. Hook Quality - Title and thumbnail effectiveness for attracting viewers
    2. Content Quality - Based on AI analysis of video content
    3. SEO Optimization - Title, description, tags effectiveness
    4. Engagement Metrics - Views, likes, comments ratios (rule-based calculation)
    5. Technical Quality - Resolution, audio quality, production value

    All factors except Engagement Metrics use the Gemini model with guideline documents
    to generate scores. Engagement Metrics uses a rule-based approach based on video statistics.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize the ScoringEngine.

        Args:
            gemini_api_key: Optional API key for Gemini. If not provided, will try to get from
                          environment variable.
        """
        self.gemini_client = GeminiClient(api_key=gemini_api_key)
        self.guidelines = load_guidelines()

        # Log the loaded guidelines
        if self.guidelines:
            logger.info(f"Loaded guidelines: {', '.join(self.guidelines.keys())}")
        else:
            logger.warning("No guidelines loaded. Falling back to rule-based scoring.")

    def score_video(
        self, video_data: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive score for a YouTube video using Gemini and guidelines.

        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results from Gemini

        Returns:
            Dictionary containing scores for each factor and overall score
        """
        try:
            logger.info("Starting video scoring process with Gemini and guidelines")

            # Calculate individual factor scores
            # Hook quality - Gemini-based with guidelines
            hook_score = self._score_hook_quality(video_data, analysis_result)

            # Content quality - Gemini-based with guidelines
            content_score = self._score_content_quality(video_data, analysis_result)

            # SEO optimization - Gemini-based with guidelines
            seo_score = self._score_seo_optimization(video_data, analysis_result)

            # Technical quality - Gemini-based with guidelines
            technical_score = self._score_technical_quality(video_data, analysis_result)

            # Engagement metrics - Rule-based calculation
            engagement_score = self._score_engagement_metrics(video_data)

            # Define weights for each factor
            weights = {
                "hook_quality": 0.25,
                "content_quality": 0.25,
                "seo_optimization": 0.15,
                "engagement_metrics": 0.20,
                "technical_quality": 0.15,
            }

            # Calculate weighted average for overall score
            overall_score = (
                hook_score * weights["hook_quality"]
                + content_score * weights["content_quality"]
                + seo_score * weights["seo_optimization"]
                + engagement_score * weights["engagement_metrics"]
                + technical_score * weights["technical_quality"]
            )

            # Round to 1 decimal place
            overall_score = round(overall_score, 1)
            logger.info(f"Calculated overall score: {overall_score}")

            # Create score breakdown
            score_breakdown = {
                "overall_score": overall_score,
                "overall_description": self._get_score_description(overall_score),
                "factors": {
                    "hook_quality": {
                        "score": hook_score,
                        "weight": weights["hook_quality"],
                        "description": self._get_score_description(hook_score),
                    },
                    "content_quality": {
                        "score": content_score,
                        "weight": weights["content_quality"],
                        "description": self._get_score_description(content_score),
                    },
                    "seo_optimization": {
                        "score": seo_score,
                        "weight": weights["seo_optimization"],
                        "description": self._get_score_description(seo_score),
                    },
                    "engagement_metrics": {
                        "score": engagement_score,
                        "weight": weights["engagement_metrics"],
                        "description": self._get_score_description(engagement_score),
                    },
                    "technical_quality": {
                        "score": technical_score,
                        "weight": weights["technical_quality"],
                        "description": self._get_score_description(technical_score),
                    },
                },
            }

            # Add detailed analyses to the score breakdown if available
            for factor in ["hook", "content", "seo", "technical"]:
                analysis_key = f"{factor}_analysis"
                if analysis_key in analysis_result and isinstance(
                    analysis_result[analysis_key], dict
                ):
                    factor_key = (
                        f"{factor}_quality" if factor != "seo" else "seo_optimization"
                    )
                    if "analysis" in analysis_result[analysis_key]:
                        score_breakdown["factors"][factor_key]["analysis"] = (
                            analysis_result[analysis_key]["analysis"]
                        )
                    if "strengths" in analysis_result[analysis_key]:
                        score_breakdown["factors"][factor_key]["strengths"] = (
                            analysis_result[analysis_key]["strengths"]
                        )
                    if "weaknesses" in analysis_result[analysis_key]:
                        score_breakdown["factors"][factor_key]["weaknesses"] = (
                            analysis_result[analysis_key]["weaknesses"]
                        )
                    if "recommendations" in analysis_result[analysis_key]:
                        score_breakdown["factors"][factor_key]["recommendations"] = (
                            analysis_result[analysis_key]["recommendations"]
                        )

            return score_breakdown

        except Exception as e:
            logger.error(f"Error scoring video: {str(e)}", exc_info=True)
            return {"error": f"Error scoring video: {str(e)}", "overall_score": 0}

    def _score_content_quality(
        self, video_data: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> float:
        """
        Score the content quality of the video using Gemini and guidelines.

        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results

        Returns:
            Score from 1-5
        """
        try:
            # Prepare enhanced video data with any existing analyses
            enhanced_video_data = video_data.copy()
            
            # Ensure video URL is available for Gemini to access content if needed
            if "url" not in enhanced_video_data and "video_id" in enhanced_video_data:
                enhanced_video_data["url"] = f"https://www.youtube.com/watch?v={enhanced_video_data['video_id']}"

            # Add transcript if available
            if "transcript" in video_data and video_data["transcript"]:
                # If transcript is very long, truncate it to avoid token limits
                transcript = video_data["transcript"]
                if len(transcript) > 5000:  # Limit transcript length
                    enhanced_video_data["transcript_summary"] = (
                        transcript[:5000] + "... [truncated]"
                    )
                    logger.info(
                        "Truncated long transcript for content quality evaluation"
                    )
                else:
                    enhanced_video_data["transcript_summary"] = transcript

            # Add any existing content analysis
            if "content_analysis" in analysis_result and isinstance(
                analysis_result["content_analysis"], dict
            ):
                enhanced_video_data["content_analysis"] = analysis_result[
                    "content_analysis"
                ]
                logger.info(
                    "Including existing content analysis in content quality evaluation"
                )

            # Extract relevant guidelines for content quality
            content_guidelines = get_category_guidelines(self.guidelines, "content")
            guidelines_dict = {
                "metrics_methodology": content_guidelines,
                "general_guidelines": self.guidelines.get("general_guidelines", ""),
            }

            # Use Gemini to generate a score based on guidelines
            logger.info("Generating content quality score using Gemini and guidelines")
            result = self.gemini_client.generate_score_with_guidelines(
                score_type="content",
                video_data=enhanced_video_data,
                guidelines=guidelines_dict,
            )

            # Extract the score
            score = float(result["score"])
            logger.info(f"Content quality score from Gemini: {score}")

            # Store the analysis in the analysis_result for UI display
            if "analysis" in result:
                if "content_analysis" not in analysis_result:
                    analysis_result["content_analysis"] = {}
                analysis_result["content_analysis"]["analysis"] = result["analysis"]
                analysis_result["content_analysis"]["strengths"] = result.get(
                    "strengths", []
                )
                analysis_result["content_analysis"]["weaknesses"] = result.get(
                    "weaknesses", []
                )
                analysis_result["content_analysis"]["recommendations"] = result.get(
                    "recommendations", []
                )

            return min(5.0, max(1.0, score))
        except Exception as e:
            logger.error(f"Error in Gemini-based content quality scoring: {str(e)}")
            logger.warning(
                "Using default score for content quality due to scoring error"
            )

            # Simple fallback - return default score
            return 3.0

        # Ensure score is within 1-5 range
        return max(1.0, min(5.0, score))

    def _score_seo_optimization(
        self, video_data: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> float:
        """
        Score the SEO optimization of the video using Gemini and guidelines.

        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results

        Returns:
            Score from 1-5
        """
        try:
            # Prepare enhanced video data with any existing analyses
            enhanced_video_data = video_data.copy()
            
            # Ensure video URL is available for Gemini to access content if needed
            if "url" not in enhanced_video_data and "video_id" in enhanced_video_data:
                enhanced_video_data["url"] = f"https://www.youtube.com/watch?v={enhanced_video_data['video_id']}"

            # Add any existing SEO analysis
            if "seo_analysis" in analysis_result and isinstance(
                analysis_result["seo_analysis"], dict
            ):
                enhanced_video_data["seo_analysis"] = analysis_result["seo_analysis"]
                logger.info(
                    "Including existing SEO analysis in SEO optimization evaluation"
                )

            # Extract relevant guidelines for SEO optimization
            seo_guidelines = get_category_guidelines(self.guidelines, "seo")
            guidelines_dict = {
                "metrics_methodology": seo_guidelines,
                "general_guidelines": self.guidelines.get("general_guidelines", ""),
            }

            # Use Gemini to generate a score based on guidelines
            logger.info("Generating SEO optimization score using Gemini and guidelines")
            result = self.gemini_client.generate_score_with_guidelines(
                score_type="seo",
                video_data=enhanced_video_data,
                guidelines=guidelines_dict,
            )

            # Extract the score
            score = float(result["score"])
            logger.info(f"SEO optimization score from Gemini: {score}")

            # Store the analysis in the analysis_result for UI display
            if "analysis" in result:
                if "seo_analysis" not in analysis_result:
                    analysis_result["seo_analysis"] = {}
                analysis_result["seo_analysis"]["analysis"] = result["analysis"]
                analysis_result["seo_analysis"]["strengths"] = result.get(
                    "strengths", []
                )
                analysis_result["seo_analysis"]["weaknesses"] = result.get(
                    "weaknesses", []
                )
                analysis_result["seo_analysis"]["recommendations"] = result.get(
                    "recommendations", []
                )

            return min(5.0, max(1.0, score))
        except Exception as e:
            logger.error(f"Error in Gemini-based SEO optimization scoring: {str(e)}")
            logger.warning(
                "Using default score for SEO optimization due to scoring error"
            )

            # Simple fallback - return default score
            return 3.0

    def _score_engagement_metrics(self, video_data: Dict[str, Any]) -> float:
        """
        Score the engagement metrics of the video.

        Args:
            video_data: Processed video metadata

        Returns:
            Score from 1-5
        """
        # Start with a base score
        score = 3.0

        try:
            # Calculate view-to-like ratio
            view_count = int(video_data.get("view_count", 0))
            like_count = int(video_data.get("like_count", 0))
            comment_count = int(video_data.get("comment_count", 0))

            if view_count > 0:
                # Calculate like-to-view ratio (as percentage)
                like_ratio = (like_count / view_count) * 100

                # Score based on like ratio
                if like_ratio >= 10:  # Excellent: 10%+ likes
                    score += 3
                elif like_ratio >= 5:  # Very good: 5-10% likes
                    score += 2
                elif like_ratio >= 2:  # Good: 2-5% likes
                    score += 1
                elif like_ratio < 1:  # Poor: <1% likes
                    score -= 1

                # Calculate comment-to-view ratio (as percentage)
                comment_ratio = (comment_count / view_count) * 100

                # Score based on comment ratio
                if comment_ratio >= 1:  # Excellent: 1%+ comments
                    score += 2
                elif comment_ratio >= 0.5:  # Good: 0.5-1% comments
                    score += 1
                elif comment_ratio < 0.1:  # Poor: <0.1% comments
                    score -= 1

                # Bonus for high absolute numbers
                if view_count >= 100000:
                    score += 0.5
                if like_count >= 10000:
                    score += 0.5
                if comment_count >= 1000:
                    score += 0.5
        except:
            # If there's an error parsing the metrics, don't adjust the score
            pass

        # Ensure score is within 1-5 range
        return max(1.0, min(5.0, score))

    def _score_technical_quality(
        self, video_data: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> float:
        """
        Score the technical quality of the video using Gemini and guidelines.

        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results

        Returns:
            Score from 1-5
        """
        try:
            # Prepare enhanced video data with any existing analyses
            enhanced_video_data = video_data.copy()
            
            # Ensure video URL is available for Gemini to access content if needed
            if "url" not in enhanced_video_data and "video_id" in enhanced_video_data:
                enhanced_video_data["url"] = f"https://www.youtube.com/watch?v={enhanced_video_data['video_id']}"

            # Add any existing technical analysis
            if "technical_analysis" in analysis_result and isinstance(
                analysis_result["technical_analysis"], dict
            ):
                enhanced_video_data["technical_analysis"] = analysis_result[
                    "technical_analysis"
                ]
                logger.info(
                    "Including existing technical analysis in technical quality evaluation"
                )

            # Extract relevant guidelines for technical quality
            technical_guidelines = get_category_guidelines(self.guidelines, "technical")
            guidelines_dict = {
                "metrics_methodology": technical_guidelines,
                "general_guidelines": self.guidelines.get("general_guidelines", ""),
            }

            # Use Gemini to generate a score based on guidelines
            logger.info(
                "Generating technical quality score using Gemini and guidelines"
            )
            result = self.gemini_client.generate_score_with_guidelines(
                score_type="technical",
                video_data=enhanced_video_data,
                guidelines=guidelines_dict,
            )

            # Extract the score
            score = float(result["score"])
            logger.info(f"Technical quality score from Gemini: {score}")

            # Store the analysis in the analysis_result for UI display
            if "analysis" in result:
                if "technical_analysis" not in analysis_result:
                    analysis_result["technical_analysis"] = {}
                analysis_result["technical_analysis"]["analysis"] = result["analysis"]
                analysis_result["technical_analysis"]["strengths"] = result.get(
                    "strengths", []
                )
                analysis_result["technical_analysis"]["weaknesses"] = result.get(
                    "weaknesses", []
                )
                analysis_result["technical_analysis"]["recommendations"] = result.get(
                    "recommendations", []
                )

            return min(5.0, max(1.0, score))
        except Exception as e:
            logger.error(f"Error in Gemini-based technical quality scoring: {str(e)}")
            logger.warning(
                "Using default score for technical quality due to scoring error"
            )

            # Simple fallback - return default score
            return 3.0

    # Market positioning score has been removed and replaced with hook score

    def _score_hook_quality(
        self, video_data: Dict[str, Any], analysis_result: Dict[str, Any]
    ) -> float:
        """
        Score the hook quality based on title and thumbnail analysis using Gemini and guidelines.

        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results

        Returns:
            Score from 1-5
        """
        try:
            # Prepare enhanced video data with any existing analyses
            enhanced_video_data = video_data.copy()
            
            # Ensure video URL is available for Gemini to access content if needed
            if "url" not in enhanced_video_data and "video_id" in enhanced_video_data:
                enhanced_video_data["url"] = f"https://www.youtube.com/watch?v={enhanced_video_data['video_id']}"

            # Add title and thumbnail analysis if available
            if "title_analysis" in analysis_result and isinstance(
                analysis_result["title_analysis"], dict
            ):
                enhanced_video_data["title_analysis"] = analysis_result[
                    "title_analysis"
                ]
                logger.info("Including title analysis in hook quality evaluation")

            if "thumbnail_analysis" in analysis_result and isinstance(
                analysis_result["thumbnail_analysis"], dict
            ):
                enhanced_video_data["thumbnail_analysis"] = analysis_result[
                    "thumbnail_analysis"
                ]
                logger.info("Including thumbnail analysis in hook quality evaluation")

            # Extract relevant guidelines for hook quality
            hook_guidelines = get_category_guidelines(self.guidelines, "hook")
            guidelines_dict = {
                "metrics_methodology": hook_guidelines,
                "general_guidelines": self.guidelines.get("general_guidelines", ""),
            }

            # Use Gemini to generate a score based on guidelines
            logger.info("Generating hook quality score using Gemini and guidelines")
            result = self.gemini_client.generate_score_with_guidelines(
                score_type="hook",
                video_data=enhanced_video_data,
                guidelines=guidelines_dict,
            )

            # Log Gemini's suggested score for hook (useful for context/debugging)
            gemini_suggested_hook_score = float(result.get("score", 3.0)) # Default if not found
            logger.info(f"Hook quality score suggested by Gemini: {gemini_suggested_hook_score}")

            # Store the detailed textual analysis from Gemini for the hook in analysis_result
            # This allows the UI to still show Gemini's qualitative assessment of the overall hook.
            if "analysis" in result:
                if "hook_analysis" not in analysis_result:
                    analysis_result["hook_analysis"] = {} # Ensure 'hook_analysis' key exists
                analysis_result["hook_analysis"]["analysis"] = result["analysis"]
                analysis_result["hook_analysis"]["strengths"] = result.get("strengths", [])
                analysis_result["hook_analysis"]["weaknesses"] = result.get("weaknesses", [])
                analysis_result["hook_analysis"]["recommendations"] = result.get("recommendations", [])
            
            # Calculate hook score as average of title and thumbnail scores
            # These scores are from the granular analyses done by VideoAnalyzer
            title_score = float(analysis_result.get("title_analysis", {}).get("score", 0.0))
            thumbnail_score = float(analysis_result.get("thumbnail_analysis", {}).get("score", 0.0))
            
            calculated_hook_score = (title_score + thumbnail_score) / 2.0
            
            logger.info(f"Calculated hook_quality score (avg of title: {title_score}, thumbnail: {thumbnail_score}): {calculated_hook_score}")
            
            # Ensure the final score is within the 1.0 to 5.0 range and rounded to one decimal place
            clamped_hook_score = min(5.0, max(1.0, calculated_hook_score))
            final_hook_score = round(clamped_hook_score, 1)
            logger.info(f"Final clamped and rounded hook_quality score for UI: {final_hook_score}")

            return final_hook_score
        except Exception as e:
            logger.error(f"Error in Gemini-based hook quality scoring: {str(e)}")
            logger.warning("Using default score for hook quality due to scoring error")

            # Simple fallback - return default score
            return 3.0

    def _get_score_description(self, score: float) -> str:
        """
        Get a qualitative description for a numeric score.

        Args:
            score: Numeric score from 1-5

        Returns:
            Qualitative description with actionable context
        """
        if score >= 4.5:
            return "Excellent - Exceptional performance, continue with current strategy"
        elif score >= 4.0:
            return "Very Good - Strong performance with minor room for improvement"
        elif score >= 3.5:
            return "Good - Solid performance with specific areas to enhance"
        elif score >= 3.0:
            return "Above Average - Performing well but has clear improvement opportunities"
        elif score >= 2.5:
            return (
                "Average - Meeting basic standards but needs significant improvements"
            )
        elif score >= 2.0:
            return "Below Average - Underperforming with multiple issues to address"
        elif score >= 1.5:
            return "Poor - Major problems requiring immediate attention"
        else:
            return "Very Poor - Critical issues requiring complete revision"


# Example usage
if __name__ == "__main__":
    # Example video data
    video_data = {
        "title": "How to Build a YouTube Chatbot with AI",
        "description": "In this comprehensive tutorial, we explore how to build a YouTube chatbot...",
        "view_count": "10000",
        "like_count": "500",
        "comment_count": "50",
        "definition": "hd",
        "caption": True,
        "tags": ["AI", "chatbot", "YouTube", "tutorial", "programming"],
    }

    # Example analysis result
    analysis_result = {
        "content_quality": "This video provides high-quality, informative content...",
        "seo_analysis": "The title is well optimized with relevant keywords...",
        "numeric_score": 8,
    }

    # Score the video
    scoring_engine = ScoringEngine()
    score_result = scoring_engine.score_video(video_data, analysis_result)

    # Print the result
    import json

    print(json.dumps(score_result, indent=2))
