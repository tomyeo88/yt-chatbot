"""
Pydantic schemas for structuring analysis data.
"""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, conlist, confloat


class AspectDetail(BaseModel):
    """Describes a specific aspect of content quality."""

    rating: Optional[str] = Field(
        None,
        description="Qualitative rating for the aspect, e.g., 'Good', 'Needs Improvement'.",
    )
    comment: Optional[str] = Field(
        None, description="Specific comments or justification for the rating."
    )


class TitleRecommendationDetail(BaseModel):
    emphasis: Optional[str] = None
    clarify: Optional[str] = None
    explanation: Optional[str] = None


class TitleAnalysisSchema(BaseModel):
    score: Optional[confloat(ge=0.0, le=5.0)] = Field(
        None, description="Overall title effectiveness score (0.0-5.0)"
    )
    effectiveness: Optional[str] = Field(
        None, description="Narrative effectiveness analysis of the title"
    )
    recommendations: Optional[List[Union[str, TitleRecommendationDetail]]] = Field(
        None, description="List of actionable recommendations for improving the title"
    )


class ThumbnailVisualElements(BaseModel):
    people: Optional[str]
    objects: Optional[str]
    text: Optional[str]
    colors: Optional[str]
    composition: Optional[str]


class ThumbnailAnalysisSchema(BaseModel):
    score: Optional[confloat(ge=0.0, le=5.0)]
    design_effectiveness: Optional[str]
    clarity: Optional[str]
    branding: Optional[str]
    visual_appeal: Optional[str]
    visual_elements: Optional[ThumbnailVisualElements]
    thumbnail_optimization: Optional[str]
    clickability: Optional[str]
    relevance_to_title: Optional[str]
    emotional_appeal: Optional[str]
    recommendations: Optional[List[str]]


class SEOAnalysisSchema(BaseModel):
    score: Optional[confloat(ge=0.0, le=5.0)]
    title_effectiveness: Optional[str]
    description: Optional[str]
    tags: Optional[str]
    thumbnail_clickability: Optional[str]
    recommendations: Optional[List[str]]


class AudienceEngagementSchema(BaseModel):
    score: Optional[confloat(ge=0.0, le=5.0)]
    hook_strength: Optional[str]
    storytelling: Optional[str]
    ctas: Optional[str]
    community_potential: Optional[str]
    recommendations: Optional[List[str]]


class TechnicalPerformanceSchema(BaseModel):
    score: Optional[confloat(ge=0.0, le=5.0)]
    video_quality: Optional[str]
    audio_quality: Optional[str]
    length_appropriateness: Optional[str]
    accessibility: Optional[str]
    recommendations: Optional[List[str]]


class ContentQualityAnalysisSchema(BaseModel):
    """Defines the structure for detailed content quality analysis from Gemini."""

    score: Optional[confloat(ge=0.0, le=5.0)] = Field(
        None,
        description="Overall content quality score (0.0-5.0). This should match the main Content Quality score if possible.",
    )
    summary: Optional[str] = Field(
        None, description="A brief overall summary of the content quality."
    )

    # Specific evaluated aspects
    clarity: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the video's clarity."
    )
    depth_of_information: Optional[Union[str, AspectDetail]] = Field(
        None,
        description="Assessment of the depth and completeness of information provided.",
    )
    structure_and_flow: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the video's organization and logical flow."
    )
    value_proposition: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the value offered to the viewer."
    )
    engagement_factors: Optional[Union[str, AspectDetail]] = Field(
        None, description="Analysis of elements that drive engagement."
    )
    originality: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the content's uniqueness and creativity."
    )
    accuracy: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the factual correctness of the content."
    )
    call_to_action_effectiveness: Optional[Union[str, AspectDetail]] = Field(
        None, description="Evaluation of any calls to action."
    )
    script_quality: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of the script's quality."
    )
    presentation_style: Optional[Union[str, AspectDetail]] = Field(
        None, description="Evaluation of the presenter's style and delivery."
    )
    editing_and_pacing: Optional[Union[str, AspectDetail]] = Field(
        None, description="Assessment of video editing and pacing."
    )

    key_topics: Optional[List[str]] = Field(
        None, description="List of key topics or subjects covered in the video."
    )
    strengths: Optional[List[str]] = Field(
        None, description="List of identified strengths of the video content."
    )
    weaknesses: Optional[List[str]] = Field(
        None, description="List of identified weaknesses or areas for improvement."
    )
    recommendations: Optional[List[Union[str, Dict[str, str]]]] = Field(
        None,
        description="Specific, actionable recommendations to improve content quality. Can be simple strings or dicts for structured recommendations.",
    )

    class Config:
        extra = "ignore"  # Allow Gemini to include extra fields without failing validation initially
        anystr_strip_whitespace = True
