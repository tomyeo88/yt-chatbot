"""Utility functions for working with YouTube URLs and video IDs."""

import re


def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from a URL.
    
    Args:
        url (str): YouTube URL in various formats
        
    Returns:
        str: YouTube video ID or empty string if not found
    """
    if not url:
        return ""
        
    # Try youtube.com/watch?v= format
    pattern1 = r'(?:youtube\.com/watch\?v=)([\w-]+)'
    match = re.search(pattern1, url)
    if match:
        return match.group(1)
    
    # Try youtu.be/ format
    pattern2 = r'(?:youtu\.be/)([\w-]+)'
    match = re.search(pattern2, url)
    if match:
        return match.group(1)
    
    # Try youtube.com/v/ format
    pattern3 = r'(?:youtube\.com/v/)([\w-]+)'
    match = re.search(pattern3, url)
    if match:
        return match.group(1)
    
    # Try youtube.com/embed/ format
    pattern4 = r'(?:youtube\.com/embed/)([\w-]+)'
    match = re.search(pattern4, url)
    if match:
        return match.group(1)
        
    return ""
