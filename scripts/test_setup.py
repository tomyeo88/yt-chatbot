"""
Test script to verify the basic setup of the YouTube Video Intelligence Chatbot.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
load_dotenv()


def check_environment_variables():
    """Check if all required environment variables are set."""
    required_vars = ["YOUTUBE_API_KEY", "GEMINI_API_KEY"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False

    print("✅ All required environment variables are set")
    return True


def test_youtube_client():
    """Test the YouTube client."""
    try:
        from src.api.youtube_client import YouTubeClient

        print("\n🔍 Testing YouTube client...")
        client = YouTubeClient()

        # Test video ID extraction
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Standard test video
        video_id = client.extract_video_id(test_url)

        if video_id != "dQw4w9WgXcQ":
            print(
                f"❌ Failed to extract video ID. Expected 'dQw4w9WgXcQ', got '{video_id}'"
            )
            return False

        print("✅ Successfully extracted video ID")

        # Test video metadata retrieval
        print("Fetching video metadata...")
        metadata = client.get_video_metadata(video_id)

        if "error" in metadata:
            print(f"❌ Failed to fetch video metadata: {metadata['error']}")
            return False

        print(
            f"✅ Successfully fetched metadata for: {metadata.get('snippet', {}).get('title', 'Unknown')}"
        )
        return True

    except Exception as e:
        print(f"❌ Error testing YouTube client: {str(e)}")
        return False


def test_gemini_client():
    """Test the Gemini client."""
    try:
        from src.api.gemini_client import GeminiClient

        print("\n🤖 Testing Gemini client...")
        client = GeminiClient()

        # Test simple content generation
        test_prompt = "Say 'Hello, World!' in a creative way."
        response = client.generate_content(test_prompt)

        if not response or "error" in response.lower():
            print(f"❌ Failed to generate content: {response}")
            return False

        print(f"✅ Successfully generated content: {response[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Error testing Gemini client: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting YouTube Video Intelligence Chatbot setup test...")

    # Check environment variables first
    if not check_environment_variables():
        print("\n❌ Please set all required environment variables in the .env file")
        return

    # Run tests
    tests = [
        ("YouTube Client", test_youtube_client),
        ("Gemini Client", test_gemini_client),
    ]

    all_passed = True
    for name, test_func in tests:
        print(f"\n=== {name} Test ===")
        if not test_func():
            all_passed = False

    # Print summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Your setup is ready to go!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Open http://localhost:8501 in your browser")
    print("3. Start analyzing YouTube videos!")


if __name__ == "__main__":
    main()
