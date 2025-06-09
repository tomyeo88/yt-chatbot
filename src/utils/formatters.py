"""
Utility functions for formatting data in user-friendly ways.
"""

import re
from datetime import datetime


def format_duration(duration_str: str) -> str:
    """
    Format ISO 8601 duration string (e.g., PT19M15S) to a readable format (e.g., 19 minutes 15 seconds)
    
    Args:
        duration_str: ISO 8601 duration string
        
    Returns:
        Human-readable duration string
    """
    if not duration_str or not isinstance(duration_str, str):
        return "Unknown duration"
        
    # Handle PT format (ISO 8601 duration)
    if duration_str.startswith('PT'):
        # Extract hours, minutes, seconds
        hours_match = re.search(r'(\d+)H', duration_str)
        minutes_match = re.search(r'(\d+)M', duration_str)
        seconds_match = re.search(r'(\d+)S', duration_str)
        
        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        seconds = int(seconds_match.group(1)) if seconds_match else 0
        
        # Format the duration string
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0 and hours == 0:  # Only show seconds if less than an hour
            parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
            
        if parts:
            return " ".join(parts)
        else:
            return "0 seconds"
    
    # If it's already in a readable format or not recognized, return as is
    return duration_str


def format_published_date(date_str: str) -> str:
    """
    Format ISO 8601 date string (e.g., 2025-04-09T17:50:55Z) to a readable format (e.g., April 9, 2025)
    
    Args:
        date_str: ISO 8601 date string
        
    Returns:
        Human-readable date string
    """
    if not date_str or not isinstance(date_str, str):
        return "Unknown date"
        
    try:
        # Parse the ISO format date
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        # Format the date in a readable way
        return date_obj.strftime("%B %d, %Y")
    except Exception:
        # If parsing fails, return the original string
        return date_str
