"""
YouTube Data API client for retrieving video metadata and statistics.

This client provides methods to interact with the YouTube Data API v3,
with built-in caching to reduce API quota usage.
"""
import os
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# For thumbnail download and analysis
import requests
from io import BytesIO
from PIL import Image


import requests
import re

# Constants
CACHE_DIR = Path(".cache/youtube")
QUOTA_FILE_PATH = CACHE_DIR / "quota_usage.json"
CACHE_TTL = timedelta(hours=24)  # Cache time-to-live
QUOTA_LIMIT = 10000  # Default daily quota limit
MIN_REQUEST_INTERVAL = 0.1  # Minimum seconds between requests

# API cost per endpoint (in quota units)
QUOTA_COSTS = {
    'videos.list': 1,
    'channels.list': 1,
    'commentThreads.list': 1,
    'comments.list': 1,
    'search.list': 100,
    'thumbnail.download': 0,  # Custom method, not an actual API call
    'playlistItems.list': 1,
    'playlists.list': 1,
    'subscriptions.list': 1,
    'activities.list': 1,
}

class QuotaExceededError(Exception):
    """Raised when the YouTube API quota is exceeded."""
    pass


class YouTubeClient:
    """Client for interacting with the YouTube Data API v3."""
    
    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True, 
                 quota_limit: int = QUOTA_LIMIT):
        """Initialize the YouTube client.
        
        Args:
            api_key: YouTube Data API v3 key. If not provided, will try to get from
                   YOUTUBE_API_KEY environment variable.
            cache_enabled: Whether to enable response caching (default: True)
            quota_limit: Daily quota limit (default: 10000)
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "YouTube Data API v3 key is required. Set YOUTUBE_API_KEY environment variable."
            )
        self.service = build('youtube', 'v3', developerKey=self.api_key)
        self.cache_enabled = cache_enabled
        self.quota_limit = quota_limit
        self.quota_used = 0
        self.last_request_time = 0
        
        # Create cache directory if caching is enabled
        if cache_enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
        # Create quota tracking file
        self._load_quota_usage()
    
    def _get_cache_key(self, method: str, **params) -> str:
        """Generate a cache key for the given method and parameters.
        
        Args:
            method: API method name (e.g., 'videos.list')
            **params: API request parameters
            
        Returns:
            Cache key string
        """
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        # Create a hash of the method and parameters
        return hashlib.md5(f"{method}:{param_str}".encode('utf-8')).hexdigest()
    
    def _load_quota_usage(self) -> None:
        """Load quota usage from tracking file."""
        quota_file = CACHE_DIR / "quota_usage.json"
        if not quota_file.exists():
            self._save_quota_usage()
            return
            
        try:
            with open(quota_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Reset quota if it's a new day
            last_reset = datetime.fromisoformat(data.get('last_reset', '2000-01-01T00:00:00'))
            now = datetime.now()
            if last_reset.date() < now.date():
                self.quota_used = 0
            else:
                self.quota_used = data.get('quota_used', 0)
        except (json.JSONDecodeError, KeyError, ValueError):
            self.quota_used = 0
    
    def _save_quota_usage(self) -> None:
        """Save quota usage to tracking file."""
        quota_file = CACHE_DIR / "quota_usage.json"
        data = {
            'quota_used': self.quota_used,
            'last_reset': datetime.now().isoformat(),
            'quota_limit': self.quota_limit
        }
        
        try:
            with open(quota_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _check_quota(self, method: str) -> None:
        """Check if we have enough quota for the request.
        
        Args:
            method: API method name
            
        Raises:
            QuotaExceededError: If the quota is exceeded
        """
        cost = QUOTA_COSTS.get(method, 1)
        if self.quota_used + cost > self.quota_limit:
            raise QuotaExceededError(
                f"YouTube API quota exceeded. Used {self.quota_used}/{self.quota_limit} units today."
            )
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting to avoid hitting API limits."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
            
        self.last_request_time = time.time()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Path to the cache file
        """
        return CACHE_DIR / f"{cache_key}.json"
    
    def _get_cached_response(self, method: str, **params) -> Optional[Dict[str, Any]]:
        """Get a cached response if available and not expired.
        
        Args:
            method: API method name
            **params: API request parameters
            
        Returns:
            Cached response or None if not available
        """
        if not self.cache_enabled:
            return None
            
        cache_key = self._get_cache_key(method, **params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            # Read the cache file
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Check if the cache is expired
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > CACHE_TTL:
                return None
                
            return cache_data['response']
        except (json.JSONDecodeError, KeyError, ValueError):
            # If there's any error reading the cache, ignore it
            return None
    
    def _save_to_cache(self, method: str, response: Dict[str, Any], **params) -> None:
        """Save an API response to the cache.
        
        Args:
            method: API method name
            response: API response to cache
            **params: API request parameters
        """
        if not self.cache_enabled:
            return
            
        cache_key = self._get_cache_key(method, **params)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'response': response
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # If there's any error saving the cache, just ignore it
            pass

    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get metadata for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict containing video metadata
            
        Raises:
            HttpError: If the API request fails
            QuotaExceededError: If the quota is exceeded
        """
        # Check cache first
        method = 'videos.list'
        params = {
            'part': "snippet,contentDetails,statistics,status,topicDetails",
            'id': video_id
        }
        
        cached_response = self._get_cached_response(method, **params)
        if cached_response:
            if not cached_response.get('items'):
                return {"error": "No video found with the provided ID"}
            return cached_response['items'][0]
        
        try:
            # Check quota and apply rate limiting
            self._check_quota(method)
            self._apply_rate_limit()
            
            # Make the request
            request = self.service.videos().list(**params)
            response = request.execute()
            
            # Update quota usage
            self.quota_used += QUOTA_COSTS.get(method, 1)
            self._save_quota_usage()
            
            # Save to cache
            self._save_to_cache(method, response, **params)
            
            if not response.get('items'):
                return {"error": "No video found with the provided ID"}
                
            return response['items'][0]
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', '')
            
            if error_reason == 'quotaExceeded':
                self.quota_used = self.quota_limit  # Mark quota as used up
                self._save_quota_usage()
                return {"error": "YouTube API quota exceeded. Please try again tomorrow."}
            
            return {"error": f"YouTube API error: {str(e)}"}

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from a YouTube URL.
        
        Args:
            url: YouTube video URL

        Returns:
            Video ID or None if not found
        """
        import re
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu.be\/)([^\&\n?#]+)',
            r'(?:youtube\.com\/embed\/)([^\&\n?#]+)',
            r'(?:youtube\.com\/v\/)([^\&\n?#]+)',
            r'(?:youtube\.com\/watch\?.*&v=)([^&\n?#]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Path to the cache file
        """
        return CACHE_DIR / f"{cache_key}.json"
    
    def _get_cached_response(self, method: str, **params) -> Optional[Dict[str, Any]]:
        """Get a cached response if available and not expired.
        
        Args:
            method: API method name
            **params: API request parameters
            
        Returns:
            Cached response or None if not available
        """
        if not self.cache_enabled:
            return None
            
        cache_key = self._get_cache_key(method, **params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            # Read the cache file
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Check if the cache is expired
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > CACHE_TTL:
                return None
                
            return cache_data['response']
        except (json.JSONDecodeError, KeyError, ValueError):
            # If there's any error reading the cache, ignore it
            return None

    def _save_to_cache(self, method: str, response: Dict[str, Any], **params) -> None:
        """Save an API response to the cache.
        
        Args:
            method: API method name
            response: API response to cache
            **params: API request parameters
        """
        if not self.cache_enabled:
            return
            
        cache_key = self._get_cache_key(method, **params)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'response': response
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # If there's any error saving the cache, just ignore it
            pass
    
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get metadata for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict containing video metadata
            
        Raises:
            HttpError: If the API request fails
            QuotaExceededError: If the quota is exceeded
        """
        # Check cache first
        method = 'videos.list'
        params = {
            'part': "snippet,contentDetails,statistics,status,topicDetails",
            'id': video_id
        }
        
        cached_response = self._get_cached_response(method, **params)
        if cached_response:
            if not cached_response.get('items'):
                return {"error": "No video found with the provided ID"}
            return cached_response['items'][0]
        
        try:
            # Check quota and apply rate limiting
            self._check_quota(method)
            self._apply_rate_limit()
            
            # Make the request
            request = self.service.videos().list(**params)
            response = request.execute()
            
            # Update quota usage
            self.quota_used += QUOTA_COSTS.get(method, 1)
            self._save_quota_usage()
            
            # Save to cache
            self._save_to_cache(method, response, **params)
            
            if not response.get('items'):
                return {"error": "No video found with the provided ID"}
                
            return response['items'][0]
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', '')
            
            if error_reason == 'quotaExceeded':
                self.quota_used = self.quota_limit  # Mark quota as used up
                self._save_quota_usage()
                return {"error": "YouTube API quota exceeded. Please try again tomorrow."}
            
            return {"error": f"YouTube API error: {str(e)}"}

    def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get information about a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Dict containing channel information
            
        Raises:
            HttpError: If the API request fails
            QuotaExceededError: If the quota is exceeded
        """
        # Check cache first
        method = 'channels.list'
        params = {
            'part': "snippet,statistics,contentDetails,brandingSettings,status",
            'id': channel_id
        }
        
        cached_response = self._get_cached_response(method, **params)
        if cached_response:
            if not cached_response.get('items'):
                return {"error": "No channel found with the provided ID"}
            return cached_response['items'][0]
        
        try:
            # Check quota and apply rate limiting
            self._check_quota(method)
            self._apply_rate_limit()
            
            # Make the request
            request = self.service.channels().list(**params)
            response = request.execute()
            
            # Update quota usage
            self.quota_used += QUOTA_COSTS.get(method, 1)
            self._save_quota_usage()
            
            # Save to cache
            self._save_to_cache(method, response, **params)
            
            if not response.get('items'):
                return {"error": "No channel found with the provided ID"}
                
            return response['items'][0]
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', '')
            
            if error_reason == 'quotaExceeded':
                self.quota_used = self.quota_limit  # Mark quota as used up
                self._save_quota_usage()
                return {"error": "YouTube API quota exceeded. Please try again tomorrow."}
            
            return {"error": f"YouTube API error: {str(e)}"}

    def get_video_comments(self, video_id: str, max_results: int = 20) -> Dict[str, Any]:
        """Get comments for a YouTube video.

        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to return (default: 20)

        Returns:
            Dict containing video comments
            
        Raises:
            HttpError: If the API request fails
            QuotaExceededError: If the quota is exceeded
        """
        # Check cache first
        method = 'commentThreads.list'
        params = {
            'part': "snippet,replies",
            'videoId': video_id,
            'maxResults': min(max_results, 100),  # API max is 100
            'order': 'relevance'
        }
        
        cached_response = self._get_cached_response(method, **params)
        if cached_response:
            if not cached_response.get('items'):
                return {"error": "No comments found or comments disabled for this video"}
            return cached_response
        
        try:
            # Check quota and apply rate limiting
            self._check_quota(method)
            self._apply_rate_limit()
            
            # Make the request
            request = self.service.commentThreads().list(**params)
            response = request.execute()
            
            # Update quota usage
            self.quota_used += QUOTA_COSTS.get(method, 1)
            self._save_quota_usage()
            
            # Save to cache
            self._save_to_cache(method, response, **params)
            
            if not response.get('items'):
                return {"error": "No comments found or comments disabled for this video"}
                
            return response
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', '')
            
            if error_reason == 'commentsDisabled':
                return {"error": "Comments are disabled for this video"}
            elif error_reason == 'quotaExceeded':
                self.quota_used = self.quota_limit  # Mark quota as used up
                self._save_quota_usage()
                return {"error": "YouTube API quota exceeded. Please try again tomorrow."}
            
            return {"error": f"YouTube API error: {str(e)}"}

    def _reset_quota_usage(self):
        """Reset the quota usage to 0."""
        self.quota_used = 0
        self._save_quota_usage()
        

        
    def get_video_thumbnail(self, video_id: str, download: bool = True, format: str = "jpg", save_dir: str = None) -> Dict[str, Any]:
        """
        Get thumbnail information for a YouTube video and optionally download it.
        
        Args:
            video_id: YouTube video ID or URL
            download: Whether to download the thumbnail image
            format: Format to save the image as (jpg, png, webp)
            save_dir: Optional directory to save the thumbnail to (if None, uses cache dir)

            
        Returns:
            Dict containing thumbnail data
        """
        # Extract video ID if URL is provided
        if "youtube.com" in video_id or "youtu.be" in video_id:
            extracted_id = self.extract_video_id(video_id)
            if not extracted_id:
                return {"error": "Invalid YouTube URL"}
            video_id = extracted_id
            
        # First get video metadata to get thumbnail URLs
        video_data = self.get_video_metadata(video_id)
        
        if "error" in video_data:
            return {"error": video_data["error"]}
            
        # Extract thumbnail URLs from video metadata
        thumbnails = video_data.get("snippet", {}).get("thumbnails", {})
        if not thumbnails:
            return {"error": "No thumbnails found for this video"}
            
        # Get video title for better file naming
        video_title = video_data.get("snippet", {}).get("title", "")
        safe_title = "".join(c for c in video_title if c.isalnum() or c in " -_").strip()[:50]
        
        # Prepare result with all available thumbnail URLs
        result = {
            "video_id": video_id,
            "video_title": video_title,
            "thumbnails": thumbnails,
            "default_thumbnail_url": thumbnails.get("default", {}).get("url"),
            "high_resolution_thumbnail_url": thumbnails.get("maxres", thumbnails.get("high", thumbnails.get("standard", thumbnails.get("default", {})))).get("url")
        }
        
        # Return early if no download requested
        if not download:
            return result
            
        # Get the highest resolution thumbnail available
        thumbnail_url = result["high_resolution_thumbnail_url"]
        if not thumbnail_url:
            return {"error": "No thumbnail URL found", "thumbnails": thumbnails}
            
        # Check cache first
        method = 'thumbnail.download'
        params = {
            'video_id': video_id,
            'url': thumbnail_url,
            'format': format
        }
        
        cache_key = f"{method}_{video_id}_{format}"
        cached_response = self._get_cached_response(cache_key, **params)
        if cached_response and not save_dir:  # Only use cache if not saving to custom directory
            return cached_response
            
        try:
            # Download the thumbnail
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code != 200:
                return {"error": f"Failed to download thumbnail: HTTP {response.status_code}", "thumbnails": thumbnails}
                
            # Load the image
            img = Image.open(BytesIO(response.content))
            # Always provide raw image bytes (test-style)
            img_format = img.format if img.format else 'JPEG'
            buffered = BytesIO()
            img.save(buffered, format=img_format)
            result["image_bytes"] = buffered.getvalue()
            # Add basic image information
            result["image_info"] = {
                "format": img_format,
                "width": img.width,
                "height": img.height,
                "size_bytes": len(result["image_bytes"])
            }
            
            




                    


                    


                    
                    
            
            # Save a local copy
            try:
                # Use the actual image format or fallback to JPEG
                img_format = img.format if img.format else 'JPEG'
                file_ext = img_format.lower()
                
                if save_dir:
                    # Use custom directory
                    thumbnail_dir = Path(save_dir)
                    thumbnail_dir.mkdir(exist_ok=True, parents=True)
                    # Use a more descriptive filename with video ID and title
                    filename = f"{video_id}_{safe_title}.{file_ext}"
                else:
                    # Use cache directory
                    thumbnail_dir = CACHE_DIR / "thumbnails"
                    thumbnail_dir.mkdir(exist_ok=True, parents=True)
                    filename = f"{video_id}.{file_ext}"
                
                img.save(thumbnail_dir / filename, format=img_format)
                result["local_path"] = str(thumbnail_dir / filename)
                print(f"Saved thumbnail to {thumbnail_dir / filename}")
            except Exception as e:
                print(f"Error saving thumbnail locally: {str(e)}")
                # Continue without local file
            
            # Save to cache
            self._save_to_cache(cache_key, result, **params)
            
            return result
            
        except Exception as e:
            return {"error": f"Error processing thumbnail: {str(e)}", "thumbnails": thumbnails}
    
    def _get_dominant_colors(self, img, num_colors=5):
        """
        Extract dominant colors from an image.
        
        Args:
            img: PIL Image object
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors in hex format
        """
        # Resize image to speed up processing
        img = img.copy()
        img.thumbnail((100, 100))
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Get colors from image
        colors = img.getcolors(img.width * img.height)
        if not colors:
            return []
            
        # Sort by count (most frequent first)
        colors.sort(reverse=True)
        
        # Convert to hex
        result = []
        for count, color in colors[:num_colors]:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            result.append({"hex": hex_color, "rgb": color, "count": count})
            
        return result
        
    def _analyze_image_properties(self, img):
        """
        Analyze image properties like brightness and contrast.
        
        Args:
            img: PIL Image object
            
        Returns:
            Tuple of (brightness, contrast) values
        """
        # Convert to grayscale for brightness/contrast analysis
        if img.mode != "L":
            gray_img = img.convert("L")
        else:
            gray_img = img
            
        # Calculate histogram
        hist = gray_img.histogram()
        
        # Calculate brightness (0-100)
        pixels = sum(hist)
        brightness = sum(i * count for i, count in enumerate(hist)) / pixels if pixels > 0 else 0
        brightness_percent = round((brightness / 255) * 100, 1)
        
        # Calculate contrast
        if pixels > 0:
            # Standard deviation as a measure of contrast
            mean = brightness
            variance = sum(((i - mean) ** 2) * count for i, count in enumerate(hist)) / pixels
            std_dev = variance ** 0.5
            # Normalize to 0-100 scale (empirical max std_dev is around 80)
            contrast_percent = round(min(100, (std_dev / 80) * 100), 1)
        else:
            contrast_percent = 0
            
        return brightness_percent, contrast_percent
        
    def get_related_videos(self, video_id: str, max_results: int = 10, order: str = 'relevance', 
    fetch_details: bool = False) -> Dict[str, Any]:
        """
        Get related videos for a YouTube video using search with the video title.
        
        Note: YouTube API v3 no longer directly supports 'relatedToVideoId', so we use
        a search based on the video's title as a workaround.
        
        Args:
            video_id: YouTube video ID or URL
            max_results: Maximum number of related videos to retrieve
            order: Order of results ('relevance', 'date', 'rating', 'title', 'viewCount')
            fetch_details: Whether to fetch additional details for each video
            
        Returns:
            Dict containing related videos data or error message
        """
        # Extract video ID if URL is provided
        if "youtube.com" in video_id or "youtu.be" in video_id:
            extracted_id = self.extract_video_id(video_id)
            if not extracted_id:
                return {"error": "Invalid YouTube URL"}
            video_id = extracted_id
        
        # Create a unique cache key
        cache_key = f"related_videos_{video_id}_{max_results}_{order}_{fetch_details}"
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        try:
            # First, get the video metadata to get the title
            video_data = self.get_video_metadata(video_id)
            if "error" in video_data:
                return {"error": video_data["error"]}
                
            # Extract title and channel for search
            video_title = video_data.get("snippet", {}).get("title", "")
            channel_title = video_data.get("snippet", {}).get("channelTitle", "")
            
            # Generate search query from title (use first 5 words to avoid being too specific)
            title_words = video_title.split()[:5]
            search_query = " ".join(title_words)
            
            # Check quota and apply rate limiting
            method = 'search.list'
            self._check_quota(method)
            self._apply_rate_limit()
            
            # Set up search parameters
            search_params = {
                'part': "snippet",
                'q': search_query,
                'type': 'video',
                'maxResults': max_results + 5,  # Get extra results to filter out the original video
                'order': order
            }
            
            # Make the request
            request = self.service.search().list(**search_params)
            response = request.execute()
            
            # Update quota usage
            self.quota_used += QUOTA_COSTS.get(method, 100)  # search.list costs 100 units
            self._save_quota_usage()
            
            # Process results - filter out the original video
            result = self._process_related_videos(response, fetch_details, exclude_video_id=video_id, max_results=max_results)
            
            # Add original video info to result
            result["original_video"] = {
                "video_id": video_id,
                "title": video_title,
                "channel_title": channel_title
            }
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except HttpError as e:
            error_details = json.loads(e.content.decode('utf-8'))
            error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', '')
            
            if error_reason == 'quotaExceeded':
                self.quota_used = self.quota_limit  # Mark quota as used up
                self._save_quota_usage()
                return {"error": "YouTube API quota exceeded. Please try again tomorrow."}
            
            return {"error": f"YouTube API error: {str(e)}"}
    
    def _process_related_videos(self, response: Dict[str, Any], fetch_details: bool = False, exclude_video_id: str = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Process search response to extract related videos information.
        
        Args:
            response: Response from search.list API call
            fetch_details: Whether to fetch additional details for each video
            exclude_video_id: Video ID to exclude from results (e.g., the original video)
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing processed related videos data
        """
        items = response.get('items', [])
        if not items:
            return {"related_videos": [], "total_results": 0}
            
        related_videos = []
        video_ids = []
        
        # First pass: extract basic info from search results
        for item in items:
            # Get video ID from the search result
            # Search results have a different structure than direct video queries
            video_id = item.get('id', {}).get('videoId')
            if not video_id:
                continue
                
            # Skip the original video if specified
            if exclude_video_id and video_id == exclude_video_id:
                continue
                
            video_ids.append(video_id)
            snippet = item.get('snippet', {})
            
            video_info = {
                "video_id": video_id,
                "title": snippet.get('title'),
                "description": snippet.get('description'),
                "published_at": snippet.get('publishedAt'),
                "channel_id": snippet.get('channelId'),
                "channel_title": snippet.get('channelTitle'),
                "thumbnails": snippet.get('thumbnails', {}),
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
            
            related_videos.append(video_info)
            
            # Limit the number of results
            if len(related_videos) >= max_results:
                break
        
        # Second pass: fetch additional details if requested
        if fetch_details and video_ids:
            try:
                # Batch fetch video details to save quota
                self._check_quota('videos.list')
                self._apply_rate_limit()
                
                # Make the request for detailed video info
                details_request = self.service.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids)
                )
                details_response = details_request.execute()
                
                # Update quota usage
                self.quota_used += QUOTA_COSTS.get('videos.list', 1)
                self._save_quota_usage()
                
                # Process the detailed information
                details_map = {}
                for item in details_response.get('items', []):
                    details_map[item.get('id')] = item
                
                # Add the detailed information to our results
                for video in related_videos:
                    video_id = video.get('video_id')
                    if video_id in details_map:
                        details = details_map[video_id]
                        
                        # Add content details (duration, etc.)
                        content_details = details.get('contentDetails', {})
                        video['duration'] = content_details.get('duration')
                        video['dimension'] = content_details.get('dimension')
                        video['definition'] = content_details.get('definition')
                        
                        # Add statistics
                        statistics = details.get('statistics', {})
                        video['view_count'] = statistics.get('viewCount')
                        video['like_count'] = statistics.get('likeCount')
                        video['comment_count'] = statistics.get('commentCount')
                        
                        # Add category ID
                        video['category_id'] = details.get('snippet', {}).get('categoryId')
                        
                        # Format the duration for easier reading
                        if 'duration' in video:
                            video['formatted_duration'] = self._format_duration(video['duration'])
            except Exception as e:
                # If fetching details fails, continue with basic info
                pass
            
        return {
            "related_videos": related_videos,
            "total_results": len(related_videos),
            "has_details": fetch_details
        }
        
    def _format_duration(self, iso_duration: str) -> str:
        """
        Format ISO 8601 duration to a more readable format.
        
        Args:
            iso_duration: Duration in ISO 8601 format (e.g., 'PT1H2M3S')
            
        Returns:
            Formatted duration string (e.g., '1:02:03')
        """
        if not iso_duration or not iso_duration.startswith('PT'):
            return '0:00'
            
        # Remove 'PT' prefix
        duration = iso_duration[2:]
        
        hours = 0
        minutes = 0
        seconds = 0
        
        # Extract hours if present
        if 'H' in duration:
            hours_part = duration.split('H')[0]
            hours = int(hours_part)
            duration = duration.split('H')[1]
        
        # Extract minutes if present
        if 'M' in duration:
            minutes_part = duration.split('M')[0]
            minutes = int(minutes_part)
            duration = duration.split('M')[1]
        
        # Extract seconds if present
        if 'S' in duration:
            seconds_part = duration.split('S')[0]
            seconds = int(seconds_part)
        
        # Format the duration
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"


def main():
    import argparse
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="YouTube API Client")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Video metadata command
    video_parser = subparsers.add_parser("video", help="Get video metadata")
    video_parser.add_argument("video_id", help="YouTube video ID or URL")

    # Channel info command
    channel_parser = subparsers.add_parser("channel", help="Get channel info")
    channel_parser.add_argument("channel_id", help="YouTube channel ID")

    # Comments command
    comments_parser = subparsers.add_parser("comments", help="Get video comments")
    comments_parser.add_argument("video_id", help="YouTube video ID or URL")
    comments_parser.add_argument(
        "--max", type=int, default=20, help="Maximum number of comments to retrieve"
    )
    
    # Thumbnail command
    thumbnail_parser = subparsers.add_parser("thumbnail", help="Get video thumbnail")
    thumbnail_parser.add_argument("video_id", help="YouTube video ID or URL")
    thumbnail_parser.add_argument("--no-download", action="store_true", help="Don't download the thumbnail")
    thumbnail_parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg", help="Format to save the thumbnail as")
    thumbnail_parser.add_argument("--save-dir", help="Directory to save the thumbnail to")
    
    # Related videos command
    related_parser = subparsers.add_parser("related", help="Get related videos")
    related_parser.add_argument("video_id", help="YouTube video ID or URL")
    related_parser.add_argument("--max", type=int, default=10, help="Maximum number of related videos to retrieve")
    related_parser.add_argument("--order", choices=["relevance", "date", "rating", "title", "viewCount"], 
                              default="relevance", help="Order of results")
    related_parser.add_argument("--details", action="store_true", help="Fetch additional details for each video")
    
    # Cache management commands
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_parser.add_argument("action", choices=["clear", "status"], help="Cache action to perform")

    # Quota management commands
    quota_parser = subparsers.add_parser("quota", help="Quota management")
    quota_parser.add_argument("action", choices=["status", "reset"], help="Quota action to perform")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return
        
    # Initialize the YouTube client
    client = YouTubeClient()

    try:
        # Check if API key is set
        if not os.environ.get("YOUTUBE_API_KEY"):
            print("Error: YOUTUBE_API_KEY environment variable is not set.")
            print("Please set it using: export YOUTUBE_API_KEY='your-api-key'")
            return
            
        client = YouTubeClient()

        if args.command == "video":
            video_id = args.video_id
            if not video_id.startswith("http"):
                video_id = args.video_id
            else:
                extracted_id = client.extract_video_id(video_id)
                if not extracted_id:
                    print(f"Error: Could not extract video ID from URL: {video_id}")
                    return
                video_id = extracted_id
                
            print(f"Fetching metadata for video ID: {video_id}")
            result = client.get_video_metadata(video_id)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "channel":
            print(f"Fetching info for channel ID: {args.channel_id}")
            result = client.get_channel_info(args.channel_id)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "comments":
            video_id = args.video_id
            if not video_id.startswith("http"):
                video_id = args.video_id
            else:
                extracted_id = client.extract_video_id(video_id)
                if not extracted_id:
                    print(f"Error: Could not extract video ID from URL: {video_id}")
                    return
                video_id = extracted_id
                
            print(f"Fetching comments for video ID: {video_id} (max: {args.max})")
            result = client.get_video_comments(video_id, args.max)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        elif args.command == "cache":
            if args.action == "clear":
                import shutil
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    print(f"Cache cleared: {CACHE_DIR}")
                else:
                    print(f"Cache directory doesn't exist: {CACHE_DIR}")
            elif args.action == "status":
                if CACHE_DIR.exists():
                    cache_files = list(CACHE_DIR.glob("*.json"))
                    cache_size = sum(f.stat().st_size for f in cache_files)
                    print(f"Cache status:\n- Directory: {CACHE_DIR}\n- Files: {len(cache_files)}\n- Size: {cache_size/1024:.2f} KB")
                else:
                    print(f"Cache directory doesn't exist: {CACHE_DIR}")
                    
        elif args.command == "quota":
            if args.action == "status":
                quota_file = QUOTA_FILE_PATH
                if quota_file.exists():
                    with open(quota_file, 'r') as f:
                        quota_data = json.load(f)
                    print(f"Quota status:\n- Used: {quota_data.get('quota_used', 0)}\n- Limit: {quota_data.get('quota_limit', 10000)}\n- Last reset: {quota_data.get('last_reset', 'Never')}")
                else:
                    print(f"No quota data found. File doesn't exist: {quota_file}")
            elif args.action == "reset":
                client._reset_quota_usage()
                print("Quota usage reset to 0")
                
        elif args.command == "thumbnail":
            video_id = args.video_id
            if not video_id.startswith("http"):
                video_id = args.video_id
            else:
                extracted_id = client.extract_video_id(video_id)
                if not extracted_id:
                    print(f"Error: Could not extract video ID from URL: {video_id}")
                    return
                video_id = extracted_id
                
            print(f"Fetching thumbnail for video ID: {video_id}")
            result = client.get_video_thumbnail(
                video_id, 
                download=not args.no_download, 
                format=args.format,
                save_dir=args.save_dir
            )
            
            # Show local path if available
            if "local_path" in result:
                print(f"\nThumbnail saved to: {result['local_path']}")
            
            print(json.dumps(result, indent=2, ensure_ascii=False))
                
        elif args.command == "related":
            video_id = args.video_id
            if not video_id.startswith("http"):
                video_id = args.video_id
            else:
                extracted_id = client.extract_video_id(video_id)
                if not extracted_id:
                    print(f"Error: Could not extract video ID from URL: {video_id}")
                    return
                video_id = extracted_id
                
            print(f"Fetching related videos for video ID: {video_id} (max: {args.max}, order: {args.order})")
            result = client.get_related_videos(
                video_id, 
                max_results=args.max,
                order=args.order,
                fetch_details=args.details
            )
            
            # Print summary before the full JSON output
            if args.details:
                print(f"Fetched {len(result.get('related_videos', []))} related videos with additional details")
            else:
                print(f"Fetched {len(result.get('related_videos', []))} related videos")
                
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
