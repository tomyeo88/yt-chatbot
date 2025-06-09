import os
import pytest
from PIL import Image
from io import BytesIO

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api.youtube_client import YouTubeClient
from src.api.gemini_client import GeminiClient
from src.analysis.video_analyzer import VideoAnalyzer
from unittest.mock import MagicMock, patch


def get_test_thumbnail_():
    # Download a real YouTube thumbnail for the standard test video
    yt = YouTubeClient()
    video_id = "dQw4w9WgXcQ"  # Standard test video: Rick Astley - Never Gonna Give You Up
    result = yt.get_video_thumbnail(video_id, download=True, format="jpg")
    url = result.get("high_resolution_thumbnail_url") or result.get(
        "default_thumbnail_url"
    )
    assert url and url.endswith((".jpg", ".jpeg", ".png", ".webp"))
    # Always re-download and encode  from the selected URL
    import requests

    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200
    img = Image.open(BytesIO(resp.content))
    img_format = img.format if img.format else "JPEG"
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    return buffered.getvalue(), url


def test_gemini_thumbnail_analysis_single_url():
    """Minimal test: analyze a single known YouTube thumbnail using the main app logic."""
    yt = YouTubeClient()
    video_id = "dQw4w9WgXcQ"  # Standard test video: Rick Astley - Never Gonna Give You Up
    result = yt.get_video_thumbnail(video_id, download=True)
    url = result.get("high_resolution_thumbnail_url") or result.get(
        "default_thumbnail_url"
    )
    assert url and url.endswith((".jpg", ".jpeg", ".png", ".webp"))
    import requests

    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(resp.content))
    img_format = img.format if img.format else "JPEG"
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    img_bytes = buffered.getvalue()
    gemini = GeminiClient()
    prompt = f"Analyze this YouTube video thumbnail in detail for video: {video_id}"
    result = gemini.generate_content_with_image(prompt, img_bytes)
    assert isinstance(result, dict)
    assert "error" not in result
    print("Gemini thumbnail analysis result:")
    print(result)


def test_gemini_thumbnail_analysis_with_all_thumbnail_urls():
    print("Executing test_gemini_thumbnail_analysis_with_all_thumbnail_urls")
    """Test that Gemini can analyze a real YouTube thumbnail image."""
    img_bytes, url = get_test_thumbnail_()
    gemini = GeminiClient()
    prompt = f"Analyze this YouTube video thumbnail in detail for video: _HZa9bgKPvQ (URL: {url})"
    # Validate  decodes to an image
    """Test Gemini thumbnail analysis with all available thumbnail URLs (default, medium, high, standard), both as JPEG and PNG, and both original and resized (512x512)."""
    yt = YouTubeClient()
    video_id = "_HZa9bgKPvQ"  # Known public video
    gemini = GeminiClient()
    prompt = f"Analyze this YouTube video thumbnail in detail for video: {video_id}"

    result = yt.get_video_thumbnail(video_id, download=True)
    for thumb in result["thumbnails"].values():
        for format in ["jpg", "png"]:
            # Always re-download and encode  from the selected URL
            import requests

            resp = requests.get(thumb["url"], timeout=10)
            assert resp.status_code == 200
            img = Image.open(BytesIO(resp.content))
            img_format = img.format if img.format else "JPEG"
            buffered = BytesIO()
            img.save(buffered, format=img_format)
            img_bytes = buffered.getvalue()

            # Resize to 512x512
            img = img.resize((512, 512))
            buffered = BytesIO()
            img.save(buffered, format=img_format)
            resized_img_bytes = buffered.getvalue()

            # Send to Gemini
            result = gemini.generate_content_with_image(
                prompt, img_bytes, model="gemini-2.0-flash"
            )
            if "error" in result:
                pytest.fail(
                    f"Failed to analyze {thumb['url']} {format} thumbnail: {result['error']}"
                )
            else:
                print(f"Successfully analyzed {thumb['url']} {format} thumbnail")
                print(result)

            result = gemini.generate_content_with_image(
                prompt, resized_img_bytes, model="gemini-2.0-flash"
            )
            if "error" in result:
                pytest.fail(
                    f"Failed to analyze resized {thumb['url']} {format} thumbnail: {result['error']}"
                )
            else:
                print(
                    f"Successfully analyzed resized {thumb['url']} {format} thumbnail"
                )
                print(result)


def test_gemini_thumbnail_analysis_with_instrument_img():
    """Test Gemini thumbnail analysis with https://goo.gle/instrument-img."""
    gemini = GeminiClient()
    prompt = "Analyze this image in detail."
    url = "https://goo.gle/instrument-img"
    import requests

    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200
    img = Image.open(BytesIO(resp.content))
    img_format = img.format if img.format else "JPEG"
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    img_bytes = buffered.getvalue()

    # Resize to 512x512
    img = img.resize((512, 512))
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    resized_img_bytes = buffered.getvalue()

    # Send to Gemini
    for format in ["jpg", "png", "webp"]:
        result = gemini.generate_content_with_image(
            prompt, img_bytes, model="gemini-2.0-flash"
        )
        if "error" in result:
            pytest.fail(
                f"Failed to analyze {url} {format} thumbnail: {result['error']}"
            )
        else:
            print(f"Successfully analyzed {url} {format} thumbnail")
            print(result)

        result = gemini.generate_content_with_image(
            prompt, resized_img_bytes, model="gemini-2.0-flash"
        )
        if "error" in result:
            pytest.fail(
                f"Failed to analyze resized {url} {format} thumbnail: {result['error']}"
            )
        else:
            print(f"Successfully analyzed resized {url} {format} thumbnail")
            print(result)


def test_invalid__thumbnail():
    """Test Gemini error handling with invalid  image."""
    gemini = GeminiClient()
    prompt = "Analyze this YouTube video thumbnail in detail."
    # Provide an invalid  string
    invalid_ = "data:image/jpeg;,notarealstring=="
    result = gemini.generate_content_with_image(prompt, invalid_)
    assert "error" in result
    assert isinstance(result["error"], str)


# Constants for VideoAnalyzer parsing tests
DUMMY_THUMBNAIL_DATA = {"image_bytes": b"dummy_image_bytes"}
DUMMY_METADATA = {"title": "Test Video"}

EXPECTED_THUMBNAIL_ANALYSIS_KEYS = [
    "visual_elements",
    "design_effectiveness",
    "thumbnail_optimization",
    "strengths_weaknesses",
    "recommendations",
    "ctr_impact",
    "score",
]
DEFAULT_THUMBNAIL_ANALYSIS_VALUES = {
    "visual_elements": {},
    "design_effectiveness": {},
    "thumbnail_optimization": {},
    "strengths_weaknesses": {"strengths": [], "weaknesses": []},
    "recommendations": [],
    "ctr_impact": "",
    "score": 0.0,
    "error": None,  # Default error state is None
    "details": None,
    "raw_response": None,
}


@pytest.fixture
def video_analyzer_instance():
    # Mock dependencies if VideoAnalyzer's __init__ requires them and they make external calls
    # For this test, we primarily care about mocking gemini_client.generate_content_with_image
    # The actual YouTubeClient and GeminiClient initialization might make network calls or require API keys.
    # So, we patch them at the class level for VideoAnalyzer instantiation.
    with patch("src.analysis.video_analyzer.YouTubeClient") as MockYouTubeClient, patch(
        "src.analysis.video_analyzer.GeminiClient"
    ) as MockGeminiClient_class:

        # Configure the class mock for GeminiClient if needed, or instance mock below
        mock_gemini_instance = MagicMock(spec=GeminiClient)
        MockGeminiClient_class.return_value = mock_gemini_instance

        analyzer = VideoAnalyzer(
            youtube_api_key="dummy_yt_key", gemini_api_key="dummy_gemini_key"
        )
        # Ensure the analyzer uses our specific mock instance for gemini_client behavior
        analyzer.gemini_client = mock_gemini_instance
        return analyzer


# Test cases for different Gemini response formats for _analyze_thumbnail
# Format: (gemini_return_value, expected_partial_result_for_assertion, expect_error_in_final_result)
gemini_response_test_cases_for_analyzer = [
    # 1. Valid JSON in markdown code block with "json" tag
    (
        '```json\n{"score": 4.5, "visual_elements": {"people": "yes"}}\n```',
        {"score": 4.5, "visual_elements": {"people": "yes"}},
        False,
    ),
    # 2. Valid JSON in markdown code block without "json" tag
    (
        '```\n{"score": 4.0, "ctr_impact": "high"}\n```',
        {"score": 4.0, "ctr_impact": "high"},
        False,
    ),
    # 3. Valid raw JSON string
    (
        '{"score": 3.5, "recommendations": ["rec1"]}',
        {"score": 3.5, "recommendations": ["rec1"]},
        False,
    ),
    # 4. Valid JSON in a dict with 'text' key (code block)
    (
        {
            "text": '```json\n{"score": 5.0, "design_effectiveness": {"clarity": "good"}}\n```'
        },
        {"score": 5.0, "design_effectiveness": {"clarity": "good"}},
        False,
    ),
    # 5. Valid JSON in a dict with 'text' key (raw JSON)
    (
        {"text": '{"score": 2.5, "strengths_weaknesses": {"strengths": ["s1"]}}'},
        {"score": 2.5, "strengths_weaknesses": {"strengths": ["s1"]}},
        False,
    ),
    # 6. Gemini client returns an error dict (simulating MCP error)
    # This error should be caught by _analyze_thumbnail's call to _parse_thumbnail_analysis
    # and _parse_thumbnail_analysis should structure it.
    (
        {"error": "Gemini API error", "details": "Quota exceeded"},
        {
            "error": "Gemini API error",
            "details": "Quota exceeded",
            "raw_response": "{'error': 'Gemini API error', 'details': 'Quota exceeded'}",
        },
        True,
    ),
    # 7. Invalid JSON string
    (
        "this is not json",
        {
            "error": "JSON parsing failed: Expecting value: line 1 column 1 (char 0)",
            "raw_response": "this is not json",
        },
        True,
    ),
    # 8. Empty string response from Gemini
    ("", {"error": "No valid response text from Gemini", "raw_response": ""}, True),
    # 9. None response from Gemini
    (
        None,
        {"error": "No valid response text from Gemini", "raw_response": "None"},
        True,
    ),
    # 10. Dict response that isn't a Gemini error and doesn't have 'text' (handled by _parse_thumbnail_analysis)
    (
        {"some_other_key": "value"},
        {
            "error": "Invalid Gemini response format",
            "raw_response": "{'some_other_key': 'value'}",
        },
        True,
    ),
    # 11. Valid JSON but missing some expected keys (should be filled with defaults by _parse_thumbnail_analysis)
    ('```json\n{"score": 3.0}\n```', {"score": 3.0}, False),
    # 12. Valid JSON with all keys
    (
        '```json\n{"visual_elements": {"people": "A person"}, "design_effectiveness": {"clarity": "Good"}, "thumbnail_optimization": {"clickability": "High"}, "strengths_weaknesses": {"strengths": ["Clear text"], "weaknesses": ["Too busy"]}, "recommendations": ["Simplify"]}, "ctr_impact": "Positive", "score": 4.2}\n```',
        {
            "visual_elements": {"people": "A person"},
            "design_effectiveness": {"clarity": "Good"},
            "thumbnail_optimization": {"clickability": "High"},
            "strengths_weaknesses": {
                "strengths": ["Clear text"],
                "weaknesses": ["Too busy"],
            },
            "recommendations": ["Simplify"],
            "ctr_impact": "Positive",
            "score": 4.2,
        },
        False,
    ),
]


@pytest.mark.parametrize(
    "gemini_return_value, expected_partial_data, expect_error_in_final_result",
    gemini_response_test_cases_for_analyzer,
)
def test_video_analyzer_analyze_thumbnail_parsing_scenarios(
    video_analyzer_instance,
    gemini_return_value,
    expected_partial_data,
    expect_error_in_final_result,
):
    analyzer = video_analyzer_instance
    # Mock the direct call made by _analyze_thumbnail
    analyzer.gemini_client.generate_content_with_image = MagicMock(
        return_value=gemini_return_value
    )

    # Call the method under test
    result = analyzer._analyze_thumbnail(DUMMY_THUMBNAIL_DATA, DUMMY_METADATA)

    assert isinstance(result, dict)

    # Check for all standard keys defined in EXPECTED_THUMBNAIL_ANALYSIS_KEYS
    for key in EXPECTED_THUMBNAIL_ANALYSIS_KEYS:
        assert (
            key in result
        ), f"Expected key '{key}' not found in result. Keys: {result.keys()}"

    if expect_error_in_final_result:
        assert (
            result.get("error") is not None
        ), "Expected an error in the result, but 'error' key was missing or None"
        # Check specific error details if provided in expected_partial_data
        if "error" in expected_partial_data:
            assert result["error"] == expected_partial_data["error"]
        if "details" in expected_partial_data:
            assert result.get("details") == expected_partial_data["details"]
        if "raw_response" in expected_partial_data:
            assert result.get("raw_response") == expected_partial_data["raw_response"]

        # For error cases, other analysis keys should have default values from DEFAULT_THUMBNAIL_ANALYSIS_VALUES
        for k, v_default in DEFAULT_THUMBNAIL_ANALYSIS_VALUES.items():
            if k not in [
                "error",
                "details",
                "raw_response",
            ]:  # These are part of the error structure itself
                assert (
                    result[k] == v_default
                ), f"Key '{k}' should have default value '{v_default}' on error, got '{result[k]}'"
    else:
        assert (
            result.get("error") is None
        ), f"Expected no error, but got: {result.get('error')}, Details: {result.get('details')}"
        # Check that the parsed values match the expected_partial_data
        for k, v_expected in expected_partial_data.items():
            assert (
                result[k] == v_expected
            ), f"For key '{k}', expected '{v_expected}', but got '{result[k]}'"

        # Check that keys not in expected_partial_data (but are standard analysis keys) have default values
        for k_default_key, v_default_val in DEFAULT_THUMBNAIL_ANALYSIS_VALUES.items():
            if k_default_key not in expected_partial_data and k_default_key not in [
                "error",
                "details",
                "raw_response",
            ]:
                assert (
                    result[k_default_key] == v_default_val
                ), f"Key '{k_default_key}' should have default value '{v_default_val}', got '{result[k_default_key]}'"

    # Specific check for score range and type if no error and score is present
    if (
        not expect_error_in_final_result
        and "score" in result
        and isinstance(result["score"], (int, float))
    ):
        assert (
            0.0 <= result["score"] <= 5.0
        ), f"Score {result['score']} out of range 0.0-5.0"
    elif not expect_error_in_final_result and (
        "score" not in result or not isinstance(result["score"], (int, float))
    ):
        # If score wasn't in expected_partial_data and it's not an error case, it should default to DEFAULT_THUMBNAIL_ANALYSIS_VALUES["score"]
        assert result["score"] == DEFAULT_THUMBNAIL_ANALYSIS_VALUES["score"]
