"""
Recommendation engine for YouTube video optimization.

This module provides functionality for generating actionable recommendations
to improve YouTube video performance based on analysis results and guidelines.
"""
from typing import Dict, Any, List, Optional, Tuple
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Generates actionable recommendations for YouTube video optimization.
    
    This class uses analysis results and predefined guidelines to generate
    specific, actionable recommendations for improving video performance.
    """
    
    def __init__(self, guidelines_provider=None):
        """
        Initialize the RecommendationEngine.
        
        Args:
            guidelines_provider: Optional provider for optimization guidelines
        """
        self.guidelines_provider = guidelines_provider
        
        # Default guidelines if no provider is specified
        self.default_guidelines = {
            'title_optimization': [
                'Include target keywords in the first half of your title',
                'Keep titles between 50-60 characters for optimal display',
                'Use numbers, questions, or emotional triggers in titles',
                'Avoid clickbait but create curiosity',
                'Include bracketed clarifications like [Tutorial], [Guide], or [Review]'
            ],
            'thumbnail_optimization': [
                'Use high contrast colors to stand out in search results',
                'Include clear, readable text (3-5 words maximum)',
                'Show facial expressions to create emotional connection',
                'Ensure thumbnail is clear even at small sizes',
                'Use consistent branding elements across all thumbnails'
            ],
            'description_optimization': [
                'Place most important information and keywords in first 2-3 lines',
                'Include a clear call-to-action',
                'Add timestamps for longer videos',
                'Include relevant links and resources',
                'Aim for 200+ words with natural keyword usage'
            ],
            'engagement_optimization': [
                'Ask a question in the video to encourage comments',
                'Include a clear call-to-action for likes and subscriptions',
                'Respond to comments to boost engagement signals',
                'Create a discussion point that viewers can debate',
                'Add an end screen with suggested videos to watch next'
            ],
            'content_optimization': [
                'Start with a strong hook in the first 15 seconds',
                'Keep editing tight with minimal dead space',
                'Include pattern interrupts every 2-3 minutes',
                'End with a clear next step for viewers',
                'Consider adding captions to improve accessibility'
            ],
            'technical_optimization': [
                'Upload in at least 1080p resolution',
                'Ensure clear audio quality with minimal background noise',
                'Use proper lighting to enhance visual quality',
                'Optimize rendering settings for YouTube compression',
                'Add chapters to improve navigation'
            ]
        }
    
    def generate_recommendations(self, video_data: Dict[str, Any], 
                               analysis_result: Dict[str, Any],
                               score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on analysis and scores.
        
        Args:
            video_data: Processed video metadata
            analysis_result: AI analysis results
            score_breakdown: Scoring results from the ScoringEngine
            
        Returns:
            Dictionary containing prioritized recommendations
        """
        try:
            # Get factor scores
            factor_scores = score_breakdown.get('factors', {})
            
            # Identify weakest areas (lowest scores)
            sorted_factors = sorted(
                factor_scores.items(),
                key=lambda x: x[1]['score']
            )
            
            # Focus on the 2-3 weakest areas
            focus_areas = sorted_factors[:3]
            
            # Extract AI recommendations if available
            ai_recommendations = analysis_result.get('optimization_recommendations', [])
            
            # Generate recommendations for each focus area
            recommendations = {}
            for factor_name, factor_data in focus_areas:
                factor_score = factor_data['score']
                
                # Only generate recommendations for factors scoring below 7
                if factor_score >= 7:
                    continue
                    
                # Generate recommendations for this factor
                factor_recommendations = self._generate_factor_recommendations(
                    factor_name,
                    factor_score,
                    video_data,
                    analysis_result
                )
                
                recommendations[factor_name] = {
                    'score': factor_score,
                    'description': factor_data['description'],
                    'recommendations': factor_recommendations
                }
            
            # Add AI recommendations as a separate category
            if ai_recommendations:
                recommendations['ai_suggestions'] = {
                    'description': 'AI-Generated Suggestions',
                    'recommendations': ai_recommendations
                }
                
            # Generate overall priority recommendations
            priority_recommendations = self._generate_priority_recommendations(
                recommendations,
                video_data,
                analysis_result
            )
            
            return {
                'overall_recommendations': priority_recommendations,
                'factor_recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            return {
                'error': f"Error generating recommendations: {str(e)}",
                'recommendations': []
            }
    
    def _generate_factor_recommendations(self, factor_name: str, factor_score: float,
                                      video_data: Dict[str, Any],
                                      analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for a specific factor.
        
        Args:
            factor_name: Name of the factor (e.g., 'content_quality')
            factor_score: Score for this factor
            video_data: Processed video metadata
            analysis_result: AI analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get guidelines for this factor
        guidelines = self._get_guidelines_for_factor(factor_name)
        
        # Factor-specific recommendation logic
        if factor_name == 'content_quality':
            # Check if video is too short or too long
            duration_str = video_data.get('formatted_duration', '')
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:  # MM:SS
                    minutes = int(parts[0])
                    if minutes < 3:
                        recommendations.append(
                            'Consider creating longer content (3-15 minutes) to provide more value'
                        )
                    elif minutes > 20:
                        recommendations.append(
                            'Consider breaking long content into multiple videos for better retention'
                        )
            
            # Check if video has captions
            if not video_data.get('caption'):
                recommendations.append(
                    'Add captions to improve accessibility and engagement'
                )
                
            # Add general content quality recommendations
            recommendations.extend(random.sample(guidelines, min(3, len(guidelines))))
            
        elif factor_name == 'seo_optimization':
            # Check title length
            title = video_data.get('title', '')
            if len(title) < 30:
                recommendations.append(
                    f'Expand your title (currently {len(title)} characters) to 50-60 characters with relevant keywords'
                )
            elif len(title) > 70:
                recommendations.append(
                    f'Shorten your title (currently {len(title)} characters) to 50-60 characters while keeping keywords'
                )
                
            # Check description length
            description = video_data.get('description', '')
            if len(description) < 100:
                recommendations.append(
                    f'Expand your description (currently {len(description)} characters) to at least 200 characters with keywords'
                )
                
            # Check tags count
            tags = video_data.get('tags', [])
            if len(tags) < 5:
                recommendations.append(
                    f'Add more tags (currently {len(tags)}) - aim for 8-12 relevant tags'
                )
            elif len(tags) > 15:
                recommendations.append(
                    f'Reduce tags (currently {len(tags)}) to 8-12 most relevant ones'
                )
                
            # Add general SEO recommendations
            recommendations.extend(random.sample(guidelines, min(2, len(guidelines))))
            
        elif factor_name == 'engagement_metrics':
            # Check view-to-like ratio
            try:
                view_count = int(video_data.get('view_count', 0))
                like_count = int(video_data.get('like_count', 0))
                
                if view_count > 0:
                    like_ratio = (like_count / view_count) * 100
                    if like_ratio < 4:
                        recommendations.append(
                            f'Improve like ratio (currently {like_ratio:.1f}%) by adding a clear call-to-action for likes'
                        )
            except:
                pass
                
            # Add general engagement recommendations
            recommendations.extend(random.sample(guidelines, min(3, len(guidelines))))
            
        elif factor_name == 'technical_quality':
            # Check video definition
            definition = video_data.get('definition', '').lower()
            if definition != 'hd':
                recommendations.append(
                    'Upload in at least 1080p resolution to improve video quality'
                )
                
            # Add general technical recommendations
            recommendations.extend(random.sample(guidelines, min(3, len(guidelines))))
            
        elif factor_name == 'market_positioning':
            # Add general market positioning recommendations
            recommendations.extend([
                'Research trending topics in your niche to create more relevant content',
                'Analyze competitor videos with higher engagement to identify opportunities',
                'Consider creating content that answers specific questions in your niche',
                'Develop a unique angle or perspective that differentiates your content',
                'Target a more specific audience segment to increase relevance'
            ])
        
        # Ensure we have at least 3 recommendations
        while len(recommendations) < 3 and guidelines:
            # Add random recommendations from guidelines
            remaining = [g for g in guidelines if g not in recommendations]
            if not remaining:
                break
            recommendations.append(random.choice(remaining))
            
        return recommendations
    
    def _generate_priority_recommendations(self, factor_recommendations: Dict[str, Any],
                                        video_data: Dict[str, Any],
                                        analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate prioritized overall recommendations.
        
        Args:
            factor_recommendations: Recommendations for each factor
            video_data: Processed video metadata
            analysis_result: AI analysis results
            
        Returns:
            List of priority recommendations
        """
        all_recommendations = []
        
        # Collect all recommendations
        for factor, data in factor_recommendations.items():
            if factor == 'ai_suggestions':
                # Add AI recommendations with high priority
                all_recommendations.extend([
                    {'text': rec, 'priority': 5, 'source': 'AI'}
                    for rec in data.get('recommendations', [])
                ])
            else:
                # Add factor recommendations with priority based on score
                score = data.get('score', 5)
                priority = max(1, 10 - int(score))  # Lower scores get higher priority
                
                all_recommendations.extend([
                    {'text': rec, 'priority': priority, 'source': factor}
                    for rec in data.get('recommendations', [])
                ])
        
        # Sort by priority (highest first)
        all_recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        # Take top 5-7 recommendations
        top_count = min(7, len(all_recommendations))
        top_recommendations = all_recommendations[:top_count]
        
        # Return just the text
        return [rec['text'] for rec in top_recommendations]
    
    def _get_guidelines_for_factor(self, factor_name: str) -> List[str]:
        """
        Get optimization guidelines for a specific factor.
        
        Args:
            factor_name: Name of the factor
            
        Returns:
            List of guideline strings
        """
        # Try to get guidelines from provider if available
        if self.guidelines_provider:
            try:
                return self.guidelines_provider.get_guidelines(factor_name)
            except:
                # Fall back to default guidelines
                pass
        
        # Map factor names to guideline categories
        factor_to_category = {
            'content_quality': 'content_optimization',
            'seo_optimization': ['title_optimization', 'description_optimization'],
            'engagement_metrics': 'engagement_optimization',
            'technical_quality': 'technical_optimization',
            'market_positioning': 'content_optimization'
        }
        
        # Get guidelines for this factor
        categories = factor_to_category.get(factor_name, [])
        if not isinstance(categories, list):
            categories = [categories]
            
        # Collect guidelines from all relevant categories
        guidelines = []
        for category in categories:
            if category in self.default_guidelines:
                guidelines.extend(self.default_guidelines[category])
                
        return guidelines or [
            # Generic fallback recommendations
            'Research successful videos in your niche for inspiration',
            'Analyze your audience demographics and create content that resonates with them',
            'Experiment with different content formats to find what works best',
            'Create a content calendar to maintain consistent uploads',
            'Engage with your audience by responding to comments'
        ]


# Example usage
if __name__ == "__main__":
    # Example video data
    video_data = {
        'title': 'How to Build a YouTube Chatbot with AI',
        'description': 'In this tutorial, we explore how to build a YouTube chatbot...',
        'view_count': '10000',
        'like_count': '300',
        'comment_count': '25',
        'definition': 'sd',
        'caption': False,
        'tags': ['AI', 'chatbot']
    }
    
    # Example analysis result
    analysis_result = {
        'content_quality': 'The video provides good information but could be more engaging.',
        'seo_analysis': 'The title is good but the description lacks keywords.',
        'optimization_recommendations': [
            'Add more detailed timestamps in the description',
            'Improve thumbnail with clearer text',
            'Include a stronger call-to-action at the end'
        ]
    }
    
    # Example score breakdown
    score_breakdown = {
        'overall_score': 6.5,
        'factors': {
            'content_quality': {'score': 7.0, 'description': 'Good'},
            'seo_optimization': {'score': 5.5, 'description': 'Average'},
            'engagement_metrics': {'score': 6.0, 'description': 'Above Average'},
            'technical_quality': {'score': 4.5, 'description': 'Below Average'},
            'market_positioning': {'score': 7.5, 'description': 'Good'}
        }
    }
    
    # Generate recommendations
    engine = RecommendationEngine()
    recommendations = engine.generate_recommendations(video_data, analysis_result, score_breakdown)
    
    # Print the result
    import json
    print(json.dumps(recommendations, indent=2))
