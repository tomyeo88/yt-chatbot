"""
Utility for loading and processing guideline documents for YouTube video analysis.
"""
import os
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_guidelines(base_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load guideline documents for YouTube video analysis.
    
    Args:
        base_path: Optional base path to look for guideline documents.
                  If not provided, will use the project root directory.
    
    Returns:
        Dictionary containing guideline document contents with keys:
        - 'general_guidelines': Content of guideline.md
        - 'metrics_methodology': Content of docs/performance_metrics_methodology.md
    """
    try:
        # Determine base path
        if not base_path:
            # Try to find the project root
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            project_root = current_dir.parent.parent  # Assuming src/utils -> src -> project_root
        else:
            project_root = Path(base_path)
        
        guidelines = {}
        
        # Load general guidelines
        guideline_path = project_root / "guideline.md"
        if guideline_path.exists():
            with open(guideline_path, 'r', encoding='utf-8') as f:
                guidelines['general_guidelines'] = f.read()
            logger.info(f"Loaded general guidelines from {guideline_path}")
        else:
            logger.warning(f"General guidelines file not found at {guideline_path}")
        
        # Load performance metrics methodology
        metrics_path = project_root / "docs" / "performance_metrics_methodology.md"
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                guidelines['metrics_methodology'] = f.read()
            logger.info(f"Loaded metrics methodology from {metrics_path}")
        else:
            logger.warning(f"Metrics methodology file not found at {metrics_path}")
        
        return guidelines
    
    except Exception as e:
        logger.error(f"Error loading guidelines: {str(e)}")
        return {}

def get_category_guidelines(guidelines: Dict[str, str], category: str) -> str:
    """
    Extract guidelines for a specific category from the full guidelines.
    
    Args:
        guidelines: Dictionary containing guideline document contents
        category: Category to extract (hook, content, seo, technical)
    
    Returns:
        String containing the relevant section of the guidelines
    """
    try:
        if 'metrics_methodology' not in guidelines:
            return ""
        
        metrics = guidelines['metrics_methodology']
        
        # Map categories to their section headers in the methodology document
        category_map = {
            'hook': '🎣 Hook (Clickability)',
            'content': '📝 Content Quality',
            'seo': '🔍 SEO Optimization',
            'technical': '⚙️ Technical Performance'
        }
        
        if category.lower() not in category_map:
            return ""
        
        section_header = category_map[category.lower()]
        
        # Find the section in the document
        start_idx = metrics.find(f"### {section_header}")
        if start_idx == -1:
            return ""
        
        # Find the end of the section (next section or end of document)
        end_idx = metrics.find("###", start_idx + 1)
        if end_idx == -1:
            section = metrics[start_idx:]
        else:
            section = metrics[start_idx:end_idx]
        
        return section
    
    except Exception as e:
        logger.error(f"Error extracting category guidelines: {str(e)}")
        return ""
