# YouTube Client Usage Guide

This document provides instructions for using the YouTube Data API client in the YT-Chatbot project.

## Environment Setup

Before using the client, make sure to set up your YouTube API key:

```bash
# Set the API key in your environment
export YOUTUBE_API_KEY='your-api-key'

# Or create a .env file with the following content
# YOUTUBE_API_KEY=your-api-key
```

## Command Line Interface

The YouTube client provides a command-line interface for testing and management:

### Video Metadata

Retrieve metadata for a YouTube video:

```bash
# Using video ID
python -m src.api.youtube_client video dQw4w9WgXcQ

# Using video URL
python -m src.api.youtube_client video https://youtu.be/dQw4w9WgXcQ  # Short URL format
```

### Channel Information

Retrieve information about a YouTube channel:

```bash
python -m src.api.youtube_client channel UCuAXFkgsw1L7xaCfnd5JJOw
```

### Video Comments

Retrieve comments for a YouTube video:

```bash
# Default (20 comments)
python -m src.api.youtube_client comments dQw4w9WgXcQ

# Specify maximum number of comments
python -m src.api.youtube_client comments dQw4w9WgXcQ --max 50

# Using video URL
python -m src.api.youtube_client comments https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s --max 30  # With timestamp
```

### Cache Management

Manage the local cache:

```bash
# View cache status
python -m src.api.youtube_client cache status

# Clear the cache
python -m src.api.youtube_client cache clear
```

### Quota Management

Manage API quota usage:

```bash
# View quota status
python -m src.api.youtube_client quota status

# Reset quota usage counter
python -m src.api.youtube_client quota reset
```

## Python API Usage

You can also use the YouTube client in your Python code:

```python
from src.api.youtube_client import YouTubeClient

# Initialize the client
client = YouTubeClient()

# Get video metadata
video_id = "dQw4w9WgXcQ"
metadata = client.get_video_metadata(video_id)

# Extract video ID from URL
# Example with standard test video
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Standard test video URL
video_id = client.extract_video_id(url)

# Get channel information
channel_id = "UCuAXFkgsw1L7xaCfnd5JJOw"
channel_info = client.get_channel_info(channel_id)

# Get video comments
comments = client.get_video_comments(video_id, max_results=30)
```

## Features

The YouTube client includes the following features:

1. **Caching**: All API responses are cached locally in `.cache/youtube` with a 24-hour TTL
2. **Rate Limiting**: Ensures minimum time intervals between API requests
3. **Quota Management**: Tracks daily quota usage and persists it in a JSON file
4. **Error Handling**: Gracefully handles quota exceeded and other API errors
