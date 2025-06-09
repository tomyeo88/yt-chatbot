import os
import json
import ast
import re
import time
import uuid
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import copy # Added for deepcopying chat states

# Suppress DeltaGenerator warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="streamlit.runtime.scriptrunner.script_runner",
)

import streamlit as st
from dotenv import load_dotenv

# Import our custom modules
from src.api.youtube_client import YouTubeClient
from src.api.gemini_client import GeminiClient
from src.analysis.video_analyzer import VideoAnalyzer
from src.analysis.scoring_engine import ScoringEngine
from src.analysis.recommendation_engine import RecommendationEngine
from src.utils.formatters import format_duration, format_published_date

# Load environment variables
load_dotenv()

# Setup logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHAT_HISTORY_FILE = "chat_history.json"
MAX_HISTORY_ITEMS = 20  # Maximum number of conversations to keep


def sanitize_for_json(data: Any) -> Any:
    """Recursively sanitize data to ensure it's JSON serializable."""
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    return data


def persist_chat_history(history: List[Dict[str, Any]]):
    """Persist the entire chat history to file."""
    # Ensure history is a list
    if not isinstance(history, list):
        st.error("Error: Chat history is not in the correct format for saving.")
        logger.error("Chat history is not a list during persist_chat_history.")
        st.error("Error: Chat history is not in the correct format for saving.")
        return

    try:
        sanitized_history = sanitize_for_json(history) # Sanitize before saving
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sanitized_history, f, indent=2)
        logger.info(f"Chat history persisted successfully with {len(sanitized_history)} items.")
    except Exception as e:
        logger.error(f"Error persisting chat history: {e}", exc_info=True)
        st.toast(f"⚠️ Error saving chat history: {str(e)}", icon="❌")


def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from file if it exists."""
    logger.info(f"Attempting to load chat history from {CHAT_HISTORY_FILE}")
    if os.path.exists(CHAT_HISTORY_FILE):
        logger.info(f"File {CHAT_HISTORY_FILE} exists.")
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                logger.info(f"Successfully loaded {len(history_data)} chats from {CHAT_HISTORY_FILE}.")
                return history_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {CHAT_HISTORY_FILE}: {e}")
        except FileNotFoundError:
            logger.error(f"File {CHAT_HISTORY_FILE} not found during open, though os.path.exists was true.")
    else:
        logger.info(f"File {CHAT_HISTORY_FILE} does not exist. Returning empty history.")
    return []


def save_chat_history(conversation: Dict[str, Any]):
    """Save a conversation to the chat history."""
    history = load_chat_history()

    # Add timestamp if not present
    if "timestamp" not in conversation:
        conversation["timestamp"] = datetime.now().isoformat()

    # Add to history and keep only the most recent conversations
    history.insert(0, conversation)
    history = history[:MAX_HISTORY_ITEMS]

    # Save to file
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def init_session_state():
    """Initialize session state variables."""
    # Initialize current_chat_id if not present
    if "current_chat_id" not in st.session_state or not st.session_state.current_chat_id:
        st.session_state.current_chat_id = str(uuid.uuid4())
        # Initialize title and messages only if creating a truly new chat session ID
        st.session_state.current_chat_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi there! I'm your YouTube Video Intelligence Assistant. "
                           "Paste a YouTube URL to get started with video analysis!",
            }
        ]
        st.session_state.analysis = None # Ensure analysis is reset for a new chat ID
        st.session_state.current_analysis = None # Ensure current_analysis is reset

    # Ensure messages list exists, even if current_chat_id was already set
    # This handles cases where the app might have been reloaded without full re-initialization
    if "messages" not in st.session_state:
         st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi there! I'm your YouTube Video Intelligence Assistant. "
                           "Paste a YouTube URL to get started with video analysis!",
            }
        ]

    
    if "current_chat_title" not in st.session_state or not st.session_state.current_chat_title:
        # If current_chat_id exists but title doesn't, try to load from history or set a default
        existing_chat = next((c for c in st.session_state.get("conversation_history", []) if c.get("id") == st.session_state.current_chat_id), None)
        if existing_chat and existing_chat.get("title"):
            st.session_state.current_chat_title = existing_chat.get("title")
        else:
            st.session_state.current_chat_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Initialize API clients and engines
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = VideoAnalyzer()

    if "scoring_engine" not in st.session_state:
        st.session_state.scoring_engine = ScoringEngine()

    if "recommendation_engine" not in st.session_state:
        st.session_state.recommendation_engine = RecommendationEngine()

    # Ensure Gemini client is initialized and working
    if "gemini_client" not in st.session_state:
        try:
            st.session_state.gemini_client = GeminiClient()
            print("Gemini client initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini client: {str(e)}")
            st.error(
                f"Failed to initialize Gemini AI client. Please check your API key and try again."
            )

    # Add debugging variables
    if "last_gemini_response" not in st.session_state:
        st.session_state.last_gemini_response = None

    # Initialize current conversation ID
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    # Initialize analysis state
    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    # Initialize current analysis for contextual responses
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None

    # Load conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = load_chat_history()

    # Flag for managing 'Thinking...' message display
    if "is_generating_response" not in st.session_state:
        st.session_state.is_generating_response = False

    # Store prompt being processed across reruns for 'Thinking...' indicator
    if "current_processing_prompt" not in st.session_state:
        st.session_state.current_processing_prompt = None

    if "gemini_futures" not in st.session_state:
        st.session_state.gemini_futures = {}

    if "thread_executor" not in st.session_state:
        st.session_state.thread_executor = ThreadPoolExecutor(max_workers=2)


def setup_sidebar():
    """Set up the sidebar components."""
    logger.info(f"ENTERING setup_sidebar. Conversation history count: {len(st.session_state.get('conversation_history', []))}")
    with st.sidebar:
        st.title("💬 Conversations")  # Original line 134
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            # *** IMPROVED FIX: Complete refactor of New Chat behavior ***
            # 1. Generate a completely new unique ID for this chat
            new_chat_id = str(uuid.uuid4())
            
            # 2. Store the new IDs and set a clear title
            st.session_state.current_chat_id = new_chat_id
            st.session_state.conversation_id = new_chat_id  # Keep both ID types consistent
            st.session_state.current_chat_title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # 3. Set the welcome message
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hi there! I'm your YouTube Video Intelligence Assistant. "
                               "Paste a YouTube URL to get started with video analysis!",
                }
            ]
            
            # 4. Reset analysis data completely
            st.session_state.analysis = None
            st.session_state.current_analysis = None
            
            # 5. Set critical flags to prevent auto-saving which causes duplication
            st.session_state.just_created_new_chat = True
            st.session_state.skip_save_for_current_chat = True
            
            # 6. Clear any existing URL parameters that might affect state
            st.query_params.clear()
            
            # 7. Log the new chat creation and trigger UI refresh
            logger.info(f"Started new chat with ID: {st.session_state.current_chat_id}")
            st.rerun() # Refresh the UI to show the new chat state with input box

        st.divider()  # This divider separates New Chat from Recent Chats

        # Show conversation history
        # logger.info(f"In setup_sidebar, number of chats in conversation_history: {len(st.session_state.get('conversation_history', []))}") # Moved to top of function
        if st.session_state.conversation_history:
            st.subheader("Recent Chats")
            for idx, conv in enumerate(st.session_state.conversation_history):
                # Safely get analysis and metadata object
                current_conv_analysis = conv.get("analysis")
                metadata_obj = None  # Use a clear variable name
                if isinstance(
                    current_conv_analysis, dict
                ):  # Check if analysis is a dict
                    metadata_obj = current_conv_analysis.get(
                        "metadata"
                    )  # Get the "metadata" dictionary

                # Determine button label based on conversation content
                button_label = "🎬 New Chat"  # Default

                if metadata_obj and isinstance(
                    metadata_obj, dict
                ):  # Check if metadata_obj exists and is a dict
                    video_title = metadata_obj.get("title")
                    channel_title = metadata_obj.get(
                        "channel_title"
                    )  # Use "channel_title" (snake_case) from chat_history.json

                    if channel_title and video_title:
                        button_label = f"🎬 {channel_title} - {video_title}"
                    elif video_title:  # If only title is found
                        button_label = f"🎬 {video_title}"
                    # If neither title nor channel_title are found,
                    # button_label remains its default "🎬 New Chat"
                elif conv.get("messages"):
                    first_message_content = conv["messages"][0].get("content", "")
                    is_initial_greeting = (
                        "Hi there! I'm your YouTube Video Intelligence Assistant."
                        in first_message_content
                    )
                    if not is_initial_greeting and first_message_content:
                        button_label = f"💬 {first_message_content[:30]}..."
                    # Else, it remains "🎬 New Chat"

                # Truncate button_label
                max_len = 45  # Slightly shorter to accommodate delete button
                if len(button_label) > max_len:
                    button_label = button_label[: max_len - 3] + "..."

                col1, col2 = st.columns([0.8, 0.2])  # Adjust column ratio as needed
                with col1:
                    if st.button(
                        button_label,
                        key=f"load_conv_idx_{idx}_{conv.get('id', 'no_id')}", # Ensure unique key with idx
                        use_container_width=True,
                    ):
                        save_current_conversation() # Save outgoing chat

                        st.session_state.current_chat_id = conv.get('id')
                        st.session_state.current_chat_title = conv.get('title', f"Chat {conv.get('id')[:8]}")
                        st.session_state.messages = copy.deepcopy(conv.get("messages", []))  # Use deepcopy
                        st.session_state.analysis = copy.deepcopy(conv.get("analysis", None))  # Use deepcopy
                        st.session_state.current_analysis = copy.deepcopy(conv.get("analysis", None))  # Use deepcopy for current_analysis as well
                        logger.info(f"Loaded chat: {st.session_state.current_chat_id}")
                        st.rerun()
                with col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_conv_idx_{idx}_{conv.get('id', 'no_id')}", # Ensure unique key with idx
                        use_container_width=True,
                        help="Delete this chat",
                    ):
                        chat_id_to_delete = conv.get('id')
                        logger.info(f"Attempting to delete chat: {chat_id_to_delete}")
                        
                        # Filter out the conversation to delete
                        st.session_state.conversation_history = [
                            c
                            for c in st.session_state.get("conversation_history", []) # Add .get for safety
                            if c.get("id") != chat_id_to_delete
                        ]
                        persist_chat_history(st.session_state.conversation_history)
                        
                        # If the deleted chat was the current one, reset to a new chat state
                        if st.session_state.current_chat_id == chat_id_to_delete:
                            logger.info(f"Current chat {chat_id_to_delete} was deleted. Resetting to new chat state.")
                            st.session_state.current_chat_id = str(uuid.uuid4())
                            st.session_state.current_chat_title = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                            st.session_state.messages = [
                                {
                                    "role": "assistant",
                                    "content": "👋 Hi there! I'm your YouTube Video Intelligence Assistant. "
                                               "Paste a YouTube URL to get started with video analysis!",
                                }
                            ]
                            st.session_state.analysis = None
                            st.session_state.current_analysis = None
                        
                        st.rerun()

        st.divider()

        # Add a clear history button
        if st.button("🗑️ Clear All History", type="secondary"):
            if os.path.exists(CHAT_HISTORY_FILE):
                os.remove(CHAT_HISTORY_FILE)
            st.session_state.conversation_history = []
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hi there! I'm your YouTube Video Intelligence Assistant. "
                    "Paste a YouTube URL to get started with video analysis!",
                }
            ]
            st.rerun()


def setup_ui():
    """Set up the Streamlit UI components."""
    st.set_page_config(
        page_title="YouTube Video Intelligence",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    setup_sidebar()

    # Main content
    st.title("🎬 YouTube Video Intelligence Chatbot")
    st.caption("Analyze and optimize your YouTube videos with AI")


def format_recommendation(
    rec_type: str, rec_desc: str, justification: str = ""
) -> None:
    """Format a single recommendation with consistent styling.

    Args:
        rec_type: The type/action of the recommendation (e.g., "Add", "Emphasize")
        rec_desc: The recommendation description/suggestion
        justification: Optional justification for the recommendation
    """
    # Clean up the action/type
    action = rec_type.strip()
    if action.endswith(":"):
        action = action[:-1].strip()
    action = action.capitalize()

    # Clean up the suggestion
    suggestion = rec_desc.strip()
    if suggestion.startswith(":"):
        suggestion = suggestion[1:].strip()

    # Clean up justification
    justification_text = justification.strip() if justification else ""
    if justification_text.startswith(":"):
        justification_text = justification_text[1:].strip()

    # Display the formatted recommendation
    st.markdown(
        f"- **[{action}]** <span style='color:white;'>{suggestion}</span>",
        unsafe_allow_html=True,
    )
    if justification_text:
        st.markdown(
            f"    - <span style='color:white; font-style:italic;'>{justification_text}</span>",
            unsafe_allow_html=True,
        )


def display_recommendation(rec: Any) -> None:
    """Display a single recommendation in the appropriate format.

    Args:
        rec: The recommendation data (dict, str, or other format)
    """
    if not isinstance(rec, dict):
        # If it's not a dictionary, just display it as is
        st.markdown(f"- {str(rec)}")
        return

    # Handle different recommendation formats
    if "type" in rec and "description" in rec:
        # Format with type and description fields
        rec_type = rec.get("type", "").replace("'", "").strip()
        rec_desc = rec.get("description", "").replace("'", "").strip()
        justification = rec.get("justification", "")
        format_recommendation(rec_type, rec_desc, justification)

    elif "action" in rec and "suggestion" in rec and "justification" in rec:
        # New standardized format with action/suggestion/justification
        rec_type = rec.get("action", "")
        rec_desc = rec.get("suggestion", "")
        justification = rec.get("justification", "")
        format_recommendation(rec_type, rec_desc, justification)

    elif "text" in rec:
        # Simple text format
        st.markdown(f"- {rec['text']}")

    else:
        # Fallback: display all key-value pairs
        formatted_rec = ", ".join([f"{k}: {v}" for k, v in rec.items()])
        st.markdown(f"- {formatted_rec}")


def display_analysis_results(analysis: Dict[str, Any]) -> None:
    """Display video analysis results in a structured format.

    Args:
        analysis: The analysis results from the VideoAnalyzer
    """
    # Debug information to verify what data we have
    print(f"[display_analysis_results] Analysis type: {type(analysis)}")
    if hasattr(analysis, 'keys'):
        print(f"[display_analysis_results] Analysis keys: {list(analysis.keys())}")
    else:
        print(f"[display_analysis_results] Analysis doesn't have keys method: {analysis}")
    
    # Check for specific analysis sections we expect
    if isinstance(analysis, dict):
        if "analysis" in analysis:
            print(f"[display_analysis_results] 'analysis' key present with type: {type(analysis['analysis'])}")
        if "metadata" in analysis:
            print(f"[display_analysis_results] 'metadata' key present")
        if "scores" in analysis:
            print(f"[display_analysis_results] 'scores' key present")
    
    # Make sure we don't accidentally convert analysis to plain text
    if isinstance(analysis, str) and (analysis.startswith('{') or analysis.startswith('[')):
        print(f"[display_analysis_results] WARNING: Analysis appears to be a string representation of JSON!")
        try:
            # Try to parse it back to a dictionary
            import json
            analysis = json.loads(analysis)
            print(f"[display_analysis_results] Successfully parsed string back to dict with keys: {analysis.keys()}")
        except Exception as e:
            print(f"[display_analysis_results] Failed to parse string as JSON: {str(e)}")

    # Initialize thumbnail_analysis with a default structure at the start of the function
    thumbnail_analysis = {
        "score": 0.0,
        "design_effectiveness": {
            "analysis": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        },
        "visual_elements": {
            "analysis": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        },
        "thumbnail_optimization": {
            "analysis": "",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        },
    }

    # Update with actual data if available
    if "thumbnail_analysis" in analysis and isinstance(
        analysis["thumbnail_analysis"], dict
    ):
        thumbnail_analysis.update(analysis["thumbnail_analysis"])
    elif (
        "analysis" in analysis
        and "thumbnail_analysis" in analysis["analysis"]
        and isinstance(analysis["analysis"]["thumbnail_analysis"], dict)
    ):
        thumbnail_analysis.update(analysis["analysis"]["thumbnail_analysis"])
    elif (
        "scores" in analysis
        and "factors" in analysis["scores"]
        and "hook_quality" in analysis["scores"]["factors"]
        and isinstance(analysis["scores"]["factors"]["hook_quality"], dict)
        and "hook_analysis" in analysis["scores"]["factors"]["hook_quality"]
        and isinstance(
            analysis["scores"]["factors"]["hook_quality"]["hook_analysis"], dict
        )
    ):
        hook_data = analysis["scores"]["factors"]["hook_quality"]["hook_analysis"]
        if "thumbnail_analysis" in hook_data and isinstance(
            hook_data["thumbnail_analysis"], dict
        ):
            thumbnail_analysis.update(hook_data["thumbnail_analysis"])

    # Debug print statements to understand the analysis data structure
    print("\n\n==== DEBUG: ANALYSIS DATA STRUCTURE =====")
    print("Top-level keys in analysis:", list(analysis.keys()))

    if "title_analysis" in analysis:
        print(
            "Title analysis found with keys:", list(analysis["title_analysis"].keys())
        )
        if "score" in analysis["title_analysis"]:
            print(f"Title analysis score: {analysis['title_analysis']['score']}")
    else:
        print("No title_analysis found in analysis dictionary")

    if "thumbnail_analysis" in analysis:
        print(
            "Thumbnail analysis found with keys:",
            list(analysis["thumbnail_analysis"].keys()),
        )
        if "score" in analysis["thumbnail_analysis"]:
            print(
                f"Thumbnail analysis score: {analysis['thumbnail_analysis']['score']}"
            )
    else:
        print("No thumbnail_analysis found in analysis dictionary")

    if "scores" in analysis and "factors" in analysis["scores"]:
        print("Performance factors:", list(analysis["scores"]["factors"].keys()))
        if "hook" in analysis["scores"]["factors"]:
            print(f"Hook factor data: {analysis['scores']['factors']['hook']}")
    print("==== END DEBUG ====\n\n")
    # Debug section to understand available data - REMOVED
    # with st.expander("Debug Information", expanded=False):
    #     st.write("### Analysis Data Structure")
    #     st.write("Top-level keys in analysis dictionary:", list(analysis.keys()))
    #
    #     if "title_analysis" in analysis:
    #         st.write("Title analysis is present with keys:", list(analysis["title_analysis"].keys()))
    #         if "score" in analysis["title_analysis"]:
    #             st.write(f"Title analysis score: {analysis['title_analysis']['score']}")
    #     else:
    #         st.write("Title analysis is NOT present in the analysis dictionary")
    #
    #     if "thumbnail_analysis" in analysis:
    #         st.write("Thumbnail analysis is present with keys:", list(analysis["thumbnail_analysis"].keys()))
    #         if "score" in analysis["thumbnail_analysis"]:
    #             st.write(f"Thumbnail analysis score: {analysis['thumbnail_analysis']['score']}")
    #     else:
    #         st.write("Thumbnail analysis is NOT present in the analysis dictionary")
    #
    #     if "scores" in analysis and "factors" in analysis["scores"]:
    #         st.write("Performance factors available:", list(analysis["scores"]["factors"].keys()))
    #         if "hook" in analysis["scores"]["factors"]:
    #             st.write(f"Hook score in factors: {analysis['scores']['factors']['hook'].get('score', 'Not set')}")

    # Display video metadata
    if "metadata" in analysis:
        metadata = analysis["metadata"]

        # Display title and channel name spanning full width
        st.header(f"📊 Analysis: {metadata.get('title', 'Unknown Video')}")
        st.subheader(f"📺 {metadata.get('channel_title', 'Unknown Channel')}")

        # Display video metadata in two columns
        col1, col2 = st.columns([1, 3])

        # Left column - Thumbnail
        with col1:
            thumbnail_displayed = False

            if "thumbnail" in analysis:
                # Debug thumbnail data
                print(
                    f"Thumbnail data keys: {analysis['thumbnail'].keys() if analysis['thumbnail'] else 'None'}"
                )

                # First try using the URL
                thumbnail_url = analysis["thumbnail"].get("url")
                video_id = analysis.get("video_id")
                video_url = (
                    f"https://www.youtube.com/watch?v={video_id}" if video_id else None
                )

                if (
                    thumbnail_url
                    and isinstance(thumbnail_url, str)
                    and thumbnail_url.startswith("http")
                ):
                    try:
                        # Create a clickable thumbnail using HTML
                        if video_url:
                            # Use HTML to create a clickable image
                            html = f"""
                            <a href="{video_url}" target="_blank">
                                <img src="{thumbnail_url}" style="width:100%">
                            </a>
                            """
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            # Fallback to regular image if no video URL
                            st.image(
                                thumbnail_url,
                                caption="Thumbnail",
                                use_container_width=True,
                            )
                        thumbnail_displayed = True
                        print(
                            f"Successfully displayed thumbnail from URL: {thumbnail_url}"
                        )
                    except Exception as e:
                        print(f"Error displaying thumbnail from URL: {str(e)}")
                        # Continue to try local path

                # If URL fails, try local path
                if not thumbnail_displayed:
                    local_path = analysis["thumbnail"].get("local_path")
                    if local_path and os.path.exists(local_path):
                        try:
                            if video_url:
                                # For local images, we need to read the file and convert to base64
                                import base64
                                from pathlib import Path

                                # Read image file and convert to base64
                                img_format = (
                                    Path(local_path).suffix.replace(".", "").lower()
                                )
                                with open(local_path, "rb") as img_file:
                                    img_str = base64.b64encode(img_file.read()).decode()

                                # Create HTML with the base64 image
                                html = f"""
                                <a href="{video_url}" target="_blank">
                                    <img src="data:image/{img_format};base64,{img_str}" style="width:100%">
                                </a>
                                """
                                st.markdown(html, unsafe_allow_html=True)
                            else:
                                # Fallback to regular image if no video URL
                                st.image(
                                    local_path,
                                    caption="Thumbnail",
                                    use_container_width=True,
                                )
                            thumbnail_displayed = True
                            print(
                                f"Successfully displayed thumbnail from local path: {local_path}"
                            )
                        except Exception as e2:
                            print(
                                f"Error displaying thumbnail from local path: {str(e2)}"
                            )
                            # Continue to placeholder

            # If no thumbnail was displayed, show a placeholder
            if not thumbnail_displayed:
                print("Using placeholder image for thumbnail")
                import numpy as np
                from PIL import Image as PILImage
                import io
                import base64

                # Create a dark gray placeholder image
                img_array = np.ones((180, 320, 3), dtype=np.uint8) * 64
                placeholder_img = PILImage.fromarray(img_array)

                if video_url:
                    # Convert PIL image to base64
                    buffer = io.BytesIO()
                    placeholder_img.save(buffer, format="PNG")
                    img_str = base64.b64encode(buffer.getvalue()).decode()

                    # Create HTML with the base64 image
                    html = f"""
                    <a href="{video_url}" target="_blank">
                        <img src="data:image/png;base64,{img_str}" style="width:100%">
                    </a>
                    """
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    # Regular non-clickable placeholder
                    st.image(
                        placeholder_img, caption="Thumbnail", use_container_width=True
                    )

        # Right column - Metrics and description
        with col2:
            # Create metrics row
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Views", f"{int(metadata.get('view_count', 0)):,}")
            with metrics_cols[1]:
                st.metric("Likes", f"{int(metadata.get('like_count', 0)):,}")
            with metrics_cols[2]:
                st.metric("Comments", f"{int(metadata.get('comment_count', 0)):,}")
            with metrics_cols[3]:
                st.metric("Duration", metadata.get("formatted_duration", "Unknown"))

            # Show description
            if metadata.get("description"):
                with st.expander("📝 Description"):
                    st.text(metadata.get("description"))

    # Thumbnail analysis is now displayed in the Hook Analysis section
    # Display AI analysis
    if "analysis" in analysis:
        ai_analysis = analysis["analysis"]

        st.subheader("🤖 AI Analysis")

    # Display scores if available
    if "scores" in analysis:
        scores = analysis["scores"]

        # Define the performance categories
        performance_categories = [
            {"name": "Hook", "key": "hook_quality", "emoji": "🎣"},
            {"name": "SEO Optimization", "key": "seo_optimization", "emoji": "🔍"},
            {"name": "Content Quality", "key": "content_quality", "emoji": "📝"},
            {
                "name": "Audience Engagement",
                "key": "engagement_metrics",
                "emoji": "👥",
            },
            {
                "name": "Technical Quality",
                "key": "technical_quality",
                "emoji": "⚙️",
            },
        ]

        # Compute overall score as average of available category scores
        if "factors" in scores:
            factors = scores["factors"]
            available_scores = []

            # Get scores from all factors
            for category in performance_categories:
                factor_key = category["key"]
                if factor_key in factors and "score" in factors[factor_key]:
                    try:
                        score_value = min(5.0, float(factors[factor_key]["score"]))
                        available_scores.append(score_value)
                    except (ValueError, TypeError):
                        pass

            # Calculate overall score
            if available_scores:
                computed_overall_score = sum(available_scores) / len(available_scores)
                overall_score = min(5.0, round(computed_overall_score, 1))
            else:
                overall_score = min(5.0, float(scores.get("overall_score", 0)))

            st.subheader("📈 Performance Scores")
            st.markdown(f"### Overall Score: {overall_score}/5")
            # Ensure progress value is between 0.0 and 1.0
            progress_value = min(1.0, max(0.0, overall_score / 5.0))
            st.progress(progress_value)

            # Factor scores
            if "factors" in scores:
                # Create columns for each factor
                cols = st.columns(len(performance_categories))

                # Display each performance category
                for i, category in enumerate(performance_categories):
                    with cols[i]:
                        factor_key = category["key"]
                        factor_data = factors.get(
                            factor_key, {"score": 0, "description": "No data available"}
                        )

                        # Display score with stars - cap at 5.0
                        score = min(5.0, float(factor_data.get("score", 0)))
                        stars = "⭐" * int(score) + "☆" * (5 - int(score))

                        st.markdown(f"### {category['emoji']} {category['name']}")
                        st.markdown(f"**{stars}** ({score}/5)")

                        # Get rating description based on score
                        rating_description = "No data available"
                        if "description" in factor_data:
                            rating_description = factor_data["description"]
                        else:
                            # Generate description based on score
                            if score >= 4.5:
                                rating_description = "Excellent"
                            elif score >= 3.5:
                                rating_description = "Good"
                            elif score >= 2.5:
                                rating_description = "Average"
                            elif score >= 1.5:
                                rating_description = "Below Average"
                            else:
                                rating_description = "Poor"

                        st.caption(rating_description)

    # Main Hook Analysis section
    st.markdown("## 🎣 Hook Analysis")

    # Get title_analysis from multiple possible locations
    title_analysis = None
    if "title_analysis" in analysis:
        title_analysis = analysis["title_analysis"]
    elif "analysis" in analysis and "title_analysis" in analysis["analysis"]:
        title_analysis = analysis["analysis"]["title_analysis"]
    elif (
        "scores" in analysis
        and "factors" in analysis["scores"]
        and "hook_quality" in analysis["scores"]["factors"]
    ):
        # Try to get from hook_analysis in scores
        hook_data = analysis["scores"]["factors"]["hook_quality"]
        if isinstance(hook_data, dict) and "hook_analysis" in hook_data:
            title_analysis = {
                "analysis": hook_data["hook_analysis"].get("title_analysis", "")
            }

    # Title Analysis subsection
    st.markdown("### 💬 Title Analysis")
    with st.expander("View Title Analysis Details", expanded=False):
        if title_analysis:
            # Check for error in title analysis
            if isinstance(title_analysis, dict) and "error" in title_analysis:
                st.warning(f"Error in title analysis: {title_analysis['error']}")

                # Try to get title information from metadata
                if "metadata" in analysis and "title" in analysis["metadata"]:
                    video_title = analysis["metadata"]["title"]
                    st.info(f"Video title: {video_title}")

                    # Try to get hook quality score from scores
                    if (
                        "scores" in analysis
                        and "factors" in analysis["scores"]
                        and "hook_quality" in analysis["scores"]["factors"]
                    ):
                        hook_data = analysis["scores"]["factors"]["hook_quality"]
                        if isinstance(hook_data, dict) and "score" in hook_data:
                            hook_score = min(5.0, float(hook_data["score"]))
                            stars = "⭐" * int(hook_score) + "☆" * (5 - int(hook_score))
                            st.markdown(
                                f"**Hook Quality Score: {stars} ({hook_score}/5)**"
                            )

                            # Display hook description if available
                            if "description" in hook_data:
                                st.markdown(
                                    f"**Assessment:** {hook_data['description']}"
                                )

                    # Display fallback message
                    st.info(
                        "The detailed title analysis could not be generated. The hook quality score above includes both title and thumbnail assessment."
                    )
            elif isinstance(title_analysis, dict):
                # Calculate title score if provided
                if "score" in title_analysis:
                    try:
                        title_score = min(5.0, float(title_analysis["score"]))
                        stars = "⭐" * int(title_score) + "☆" * (5 - int(title_score))
                        st.markdown(f"**Title Score: {stars} ({title_score}/5)**")
                    except (ValueError, TypeError) as e:
                        st.warning(f"Title score not available: {e}")

                # Display title analysis content with proper formatting
                if "analysis" in title_analysis and title_analysis["analysis"]:
                    st.markdown("**Analysis:**")
                    st.markdown(title_analysis["analysis"])
                elif (
                    "effectiveness" in title_analysis
                    and title_analysis["effectiveness"]
                ):
                    st.markdown("**Effectiveness:**")

                    # Parse the effectiveness content
                    effectiveness_text = title_analysis["effectiveness"]

                    # Ensure effectiveness_text is a string
                    if not isinstance(effectiveness_text, str):
                        effectiveness_text = str(effectiveness_text)

                    # Check if it's in JSON-like format with quotes and curly braces
                    if effectiveness_text.strip().startswith(
                        "{"
                    ) and effectiveness_text.strip().endswith("}"):
                        try:
                            import json
                            import ast

                            # Try to parse as JSON or Python dict
                            try:
                                parsed_content = json.loads(effectiveness_text)
                            except:
                                try:
                                    # Try using ast.literal_eval for Python dict-like strings
                                    parsed_content = ast.literal_eval(
                                        effectiveness_text
                                    )
                                except:
                                    parsed_content = None

                            if parsed_content and isinstance(parsed_content, dict):
                                # Format each key-value pair as a bullet point
                                for key, value in parsed_content.items():
                                    key = key.strip("'").replace("_", " ").title()
                                    st.markdown(f"• **{key}:** {value}")
                            else:
                                # If parsing fails, display as is
                                st.markdown(effectiveness_text)
                        except Exception:
                            # If parsing fails, display as is
                            st.markdown(effectiveness_text)
                    else:
                        # Not JSON format, display as is
                        st.markdown(effectiveness_text)

                # Display recommendations if available
                if (
                    "recommendations" in title_analysis
                    and title_analysis["recommendations"]
                ):
                    st.markdown("**Recommendations:**")
                    if isinstance(title_analysis["recommendations"], list):
                        # Display each recommendation using the helper function
                        for rec in title_analysis["recommendations"]:
                            display_recommendation(rec)
                    elif isinstance(title_analysis["recommendations"], str):
                        # Try to parse JSON string if it looks like JSON
                        try:
                            import json
                            import ast

                            # Try different parsing methods
                            try:
                                recs_list = json.loads(
                                    title_analysis["recommendations"]
                                )
                            except Exception:
                                try:
                                    recs_list = ast.literal_eval(
                                        title_analysis["recommendations"]
                                    )
                                except Exception:
                                    recs_list = None

                            if isinstance(recs_list, list):
                                for rec in recs_list:
                                    display_recommendation(rec)
                            elif recs_list is not None:
                                st.markdown(str(recs_list))
                            else:
                                # If not JSON, try to parse as plain text with bullet points
                                lines = title_analysis["recommendations"].split("\n")
                                for line in lines:
                                    line = line.strip()
                                    if line.startswith(("-", "*", "•")) or (
                                        len(line) > 2
                                        and line[0].isdigit()
                                        and line[1:].startswith(". ")
                                    ):
                                        st.markdown(line)
                                    elif line:  # Only add non-empty lines
                                        st.markdown(f"- {line}")
                        except Exception as e:
                            st.markdown(title_analysis["recommendations"])
                            st.error(f"Error parsing recommendations: {str(e)}")
                    else:
                        st.markdown(
                            "- No recommendations available in the expected format."
                        )
                        raw_rec = title_analysis["recommendations"]
                        if isinstance(
                            raw_rec, (dict, list, str, int, float, bool, type(None))
                        ):
                            st.json(raw_rec)
                        else:
                            st.warning("Raw recommendations data is not displayable.")

    # Thumbnail Analysis section - moved outside of Title Analysis
    st.markdown("### 😋 Thumbnail Analysis")
    with st.expander("View Thumbnail Analysis Details", expanded=False):
        # Debug output - keep this for now to help with troubleshooting
        # st.json({"thumbnail_analysis_data": thumbnail_analysis})

        if thumbnail_analysis:
            # Display thumbnail analysis
            if isinstance(thumbnail_analysis, dict):
                # Display thumbnail score if available
                if "score" in thumbnail_analysis:
                    try:
                        thumbnail_score = min(5.0, float(thumbnail_analysis["score"]))
                        stars = "⭐" * int(thumbnail_score) + "☆" * (
                            5 - int(thumbnail_score)
                        )
                        st.markdown(
                            f"**Thumbnail Score: {stars} ({thumbnail_score:.1f}/5)**"
                        )
                    except (ValueError, TypeError) as e:
                        st.warning(f"Thumbnail score not available: {e}")

                # Helper function to display content
                def display_content(content, prefix=""):
                    if isinstance(content, dict):
                        for key, value in content.items():
                            key_display = key.replace("_", " ").title()
                            if isinstance(value, dict):
                                st.markdown(f"**{key_display}**")
                                display_content(value, prefix + "  ")
                            elif isinstance(value, (list, tuple)):
                                st.markdown(f"**{key_display}:**")
                                for item in value:
                                    if isinstance(item, (dict, list)):
                                        display_content(item, prefix + "  ")
                                    else:
                                        st.markdown(f"{prefix}• {item}")
                            else:
                                st.markdown(f"**{key_display}:** {value}")
                    elif isinstance(content, (list, tuple)):
                        for item in content:
                            if isinstance(item, (dict, list)):
                                display_content(item, prefix + "  ")
                            else:
                                st.markdown(f"{prefix}- {item}")
                    elif content:  # Only display non-empty strings/values
                        st.markdown(content)

                # Display main analysis content
                for key in ["analysis", "effectiveness"]:
                    if key in thumbnail_analysis and thumbnail_analysis[key]:
                        content = thumbnail_analysis[key]
                        if key == "analysis" or (
                            key == "effectiveness"
                            and "analysis" not in thumbnail_analysis
                        ):
                            st.markdown(
                                f"**{key.title()}:**"
                                if key == "effectiveness"
                                else "**Analysis:**"
                            )
                            display_content(content)

                # Display design effectiveness, visual elements, and optimization
                sections = [
                    ("Design Effectiveness", "design_effectiveness"),
                    ("Visual Elements", "visual_elements"),
                    ("Thumbnail Optimization", "thumbnail_optimization"),
                ]

                for section_title, section_key in sections:
                    if (
                        section_key in thumbnail_analysis
                        and thumbnail_analysis[section_key]
                    ):
                        section_data = thumbnail_analysis[section_key]
                        st.markdown(f"**{section_title}:**")

                        # Try to parse string as JSON/dict if needed
                        if isinstance(section_data, str):
                            if section_data.strip().startswith(
                                "{"
                            ) and section_data.strip().endswith("}"):
                                try:
                                    import json
                                    import ast

                                    try:
                                        section_data = json.loads(section_data)
                                    except:
                                        section_data = ast.literal_eval(section_data)
                                except:
                                    pass
                            else:
                                st.markdown(section_data)
                                continue

                        # Display the section content
                        if isinstance(section_data, dict):
                            display_content(section_data)
                        else:
                            st.markdown(section_data)

                # Display any remaining top-level keys that haven't been displayed yet
                displayed_keys = {
                    "score",
                    "analysis",
                    "effectiveness",
                    "design_effectiveness",
                    "visual_elements",
                    "thumbnail_optimization",
                    "recommendations",
                }
                remaining_keys = set(thumbnail_analysis.keys()) - displayed_keys

                # Display recommendations if available
                if (
                    "recommendations" in thumbnail_analysis
                    and thumbnail_analysis["recommendations"]
                ):
                    st.markdown("**Recommendations:**")
                    recs = thumbnail_analysis["recommendations"]
                    if isinstance(recs, str):
                        # Try to parse JSON string if it looks like JSON
                        if recs.strip().startswith("{") and recs.strip().endswith("}"):
                            try:
                                import json
                                import ast

                                try:
                                    recs = json.loads(recs)
                                except:
                                    recs = ast.literal_eval(recs)
                            except:
                                pass

                    if isinstance(recs, list):
                        for rec in recs:
                            if isinstance(rec, dict):
                                display_recommendation(rec)
                            else:
                                st.markdown(f"• {rec}")
                    elif isinstance(recs, dict):
                        display_recommendation(recs)
                    elif recs:  # Only display if not empty string
                        st.markdown(f"• {recs}")
    # Get scores from analysis at the beginning of the function
    scores = analysis.get("scores", {})
    # The scores are directly in the scores dict, not under 'factors' key
    factor_scores = {}
    for factor in [
        "hook_quality",
        "content_quality",
        "seo_optimization",
        "engagement_metrics",
        "technical_quality",
    ]:
        if factor in scores:
            try:
                factor_scores[factor] = (
                    float(scores[factor]) * 5.0
                )  # Convert to 5-point scale
            except Exception:
                pass

    # SEO Optimization
    st.markdown("## 🔍 SEO Optimization")
    ai_analysis = analysis.get("analysis", {})
    factors = analysis.get("scores", {}).get("factors", {})
    if ai_analysis.get("seo_optimization"):
        content = ai_analysis["seo_optimization"]
        seo_score = None
        try:
            seo_score = float(factors.get("seo_optimization", {}).get("score", None))
            seo_score = min(5.0, seo_score)
        except (TypeError, ValueError):
            seo_score = None
        # Define the SEO display function first, so it's available everywhere
        def display_seo_content(content, prefix=""):
            if isinstance(content, dict):
                # First try to extract key fields of interest
                important_fields = [
                    "title_effectiveness", 
                    "description", 
                    "tags", 
                    "thumbnail_clickability"
                ]
                recommendations = None
                
                # Special handling for recommendations
                if "recommendations" in content:
                    recommendations = content["recommendations"]
                
                # Display main fields
                for key in important_fields:
                    if key in content and content[key]:
                        key_display = key.replace("_", " ").title()
                        st.markdown(f"**{key_display}:** {content[key]}")
                
                # Display recommendations last
                if recommendations:
                    st.markdown("**Recommendations:**")
                    if isinstance(recommendations, list):
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.markdown(recommendations)
                        
                # Handle any other keys
                for key, value in content.items():
                    if key not in important_fields and key != "recommendations" and key != "score":
                        key_display = key.replace("_", " ").title()
                        if isinstance(value, (dict, list)):
                            st.markdown(f"**{key_display}:**")
                            display_seo_content(value, prefix + "  ")
                        else:
                            st.markdown(f"**{key_display}:** {value}")
            
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, (dict, list)):
                        display_seo_content(item, prefix + "  ")
                    else:
                        st.markdown(f"{prefix}- {item}")
            else:
                st.markdown(f"{prefix}{content}")
                
        with st.expander("View SEO Optimization Details", expanded=False):
            if seo_score is not None:
                stars = "⭐" * int(seo_score) + "☆" * (5 - int(seo_score))
                st.markdown(f"**SEO Optimization Score: {stars} ({seo_score:.1f}/5)**")
            # Debug statements removed - UI is now clean
            
            if content:
                # Skip string display that contains raw JSON - only use our formatter
                if isinstance(content, (int, float)):
                    st.markdown(str(content))
                elif isinstance(content, str):
                    # Try to parse JSON from string
                    try:
                        # Find potential JSON in the string
                        import json
                        import re
                        
                        # First check if the entire string is valid JSON
                        json_parsed = False
                        try:
                            json_data = json.loads(content)
                            display_seo_content(json_data)
                            json_parsed = True
                        except json.JSONDecodeError:
                            pass
                            
                        # Only continue if we haven't successfully parsed JSON yet
                        if not json_parsed:
                            # Look for JSON object pattern within the string
                            json_pattern = r'\{[\s\S]*\}'
                            match = re.search(json_pattern, content)
                            
                            if match:
                                potential_json = match.group(0)
                                try:
                                    json_data = json.loads(potential_json)
                                    display_seo_content(json_data)
                                    json_parsed = True
                                except json.JSONDecodeError:
                                    pass
                    
                    except Exception as e:
                        st.warning(f"Error parsing SEO data: {str(e)}")
                    
                    # If we get here and haven't parsed JSON successfully, display the cleaned string
                    if not json_parsed:
                        cleaned = re.sub(r"SEO Optimization \([\-\d.]+/5\)\*\*:?:?\s*", "", content)
                        # Remove "SEO Optimization (3/5)" line that might be in the string
                        cleaned = re.sub(r"SEO Optimization \([\d\.]+/5\)\s*\n", "", cleaned)
                        st.markdown(cleaned)
                elif isinstance(content, (dict, list)):
                    # Display structured data nicely instead of raw JSON
                    display_seo_content(content)
                else:
                    st.warning("SEO optimization analysis data is not displayable.")
            else:
                st.markdown("No SEO optimization analysis available.")

    # Content Analysis section (main section)
    st.markdown("## 📝 Content Analysis")
    factors = analysis.get("scores", {}).get("factors", {})
    content_score = None
    try:
        content_score = float(factors.get("content_quality", {}).get("score", None))
        content_score = min(5.0, content_score)
    except (TypeError, ValueError):
        content_score = None

    # Robust extraction, similar to Thumbnail Analysis
    def try_parse_json_block(s):
        import json, re
        # Find the first JSON block inside triple backticks (optionally with 'json')
        match = re.search(r"```(?:json)?\s*({[\s\S]+?})\s*```", s, re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e:
                print("JSON parse error:", e)
        # Fallback: try to parse any JSON-looking substring
        s_strip = s.strip()
        brace_match = re.search(r"({[\s\S]+})", s_strip)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except Exception as e:
                print("JSON parse error (fallback):", e)
        return None

    content_analysis_data = (
        analysis.get("content_quality_analysis")
        or analysis.get("analysis", {}).get("content_quality_detail")
        or analysis.get("analysis", {}).get("content_quality")
        or analysis.get("content_quality")
    )

    if isinstance(content_analysis_data, str):
        parsed = try_parse_json_block(content_analysis_data)
        if parsed:
            content_analysis_data = parsed

    with st.expander("View Content Analysis Details", expanded=False):
        if content_score is not None:
            stars = "⭐" * int(content_score) + "☆" * (5 - int(content_score))
            st.markdown(f"**Content Quality Score: {stars} ({content_score:.1f}/5)**")
        else:
            st.markdown("**Content Quality Score: N/A**")

        if isinstance(content_analysis_data, dict):
            # Show summary first
            summary = content_analysis_data.get("summary")
            if summary:
                st.markdown(f"**Summary:** {summary}")

            # Show all known aspects
            known_aspects = {
                "clarity": "Clarity",
                "depth_of_information": "Depth of Information",
                "structure_and_flow": "Structure and Flow",
                "value_proposition": "Value Proposition",
                "engagement_factors": "Engagement Factors",
                "originality": "Originality",
                "accuracy": "Accuracy",
                "call_to_action_effectiveness": "Call to Action Effectiveness",
                "script_quality": "Script Quality",
                "presentation_style": "Presentation Style",
                "editing_and_pacing": "Editing and Pacing",
            }
            for key, display_name in known_aspects.items():
                aspect = content_analysis_data.get(key)
                if aspect:
                    if isinstance(aspect, dict):
                        rating = aspect.get("rating")
                        comment = aspect.get("comment") or aspect.get("details", "")
                        if rating and comment:
                            st.markdown(f"**{display_name}:** {rating}. {comment}")
                        elif rating:
                            st.markdown(f"**{display_name}:** {rating}")
                        elif comment:
                            st.markdown(f"**{display_name}:** {comment}")
                    else:
                        st.markdown(f"**{display_name}:** {aspect}")

            # Key topics
            key_topics = content_analysis_data.get("key_topics")
            if key_topics:
                st.markdown("**Key Topics Covered:**")
                if isinstance(key_topics, list):
                    for topic in key_topics:
                        st.markdown(f"    - {topic}")
                else:
                    st.markdown(f"    {key_topics}")

            # Strengths
            strengths = content_analysis_data.get("strengths")
            if strengths:
                st.markdown("**Strengths:**")
                if isinstance(strengths, list):
                    for item in strengths:
                        st.markdown(f"    - {item}")
                else:
                    st.markdown(f"    {strengths}")

            # Weaknesses
            weaknesses = content_analysis_data.get("weaknesses") or content_analysis_data.get("areas_for_improvement")
            if weaknesses:
                st.markdown("**Areas for Improvement / Weaknesses:**")
                if isinstance(weaknesses, list):
                    for item in weaknesses:
                        st.markdown(f"    - {item}")
                else:
                    st.markdown(f"    {weaknesses}")

            # Recommendations
            recommendations = content_analysis_data.get("recommendations")
            if recommendations:
                st.markdown("**Recommendations (Content Specific):**")
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        st.markdown(f"    - {rec}")
                else:
                    st.markdown(f"    {recommendations}")
        else:
            st.warning("Content analysis data is not in the expected format.")

    # Audience Engagement section
    if ai_analysis.get("audience_engagement"):
        st.markdown("## 👥 Audience Engagement")
        content = ai_analysis["audience_engagement"]
        factors = analysis.get("scores", {}).get("factors", {})
        audience_score = None
        try:
            audience_score = float(
                factors.get("engagement_metrics", {}).get("score", None)
            )
            audience_score = min(5.0, audience_score)
        except (TypeError, ValueError):
            audience_score = None
        # Define the engagement display function first
        def display_engagement_content(content, prefix=""):
            if isinstance(content, dict):
                # First try to extract key fields of interest
                important_fields = [
                    "hook_strength", 
                    "storytelling", 
                    "ctas", 
                    "community_potential"
                ]
                recommendations = None
                
                # Special handling for recommendations
                if "recommendations" in content:
                    recommendations = content["recommendations"]
                
                # Display main fields
                for key in important_fields:
                    if key in content and content[key]:
                        key_display = key.replace("_", " ").title()
                        st.markdown(f"**{key_display}:** {content[key]}")
                
                # Display recommendations last
                if recommendations:
                    st.markdown("**Recommendations:**")
                    if isinstance(recommendations, list):
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.markdown(recommendations)
                        
                # Handle any other keys
                for key, value in content.items():
                    if key not in important_fields and key != "recommendations" and key != "score":
                        key_display = key.replace("_", " ").title()
                        if isinstance(value, (dict, list)):
                            st.markdown(f"**{key_display}:**")
                            display_engagement_content(value, prefix + "  ")
                        else:
                            st.markdown(f"**{key_display}:** {value}")
            
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, (dict, list)):
                        display_engagement_content(item, prefix + "  ")
                    else:
                        st.markdown(f"{prefix}- {item}")
            else:
                st.markdown(f"{prefix}{content}")

        with st.expander("View Audience Engagement Details", expanded=False):
            if audience_score is not None:
                stars = "⭐" * int(audience_score) + "☆" * (5 - int(audience_score))
                st.markdown(
                    f"**Audience Engagement Score: {stars} ({audience_score:.1f}/5)**"
                )
            else:
                st.markdown("**Audience Engagement Score: N/A**")
            
            # Debug statements removed - UI is now clean
            
            if content:
                # Skip string display that contains raw JSON - only use our formatter
                if isinstance(content, (int, float)):
                    st.markdown(str(content))
                elif isinstance(content, str):
                    # Try to parse JSON from string
                    try:
                        # Find potential JSON in the string
                        import json
                        import re
                        
                        # First check if the entire string is valid JSON
                        json_parsed = False
                        try:
                            json_data = json.loads(content)
                            display_engagement_content(json_data)
                            json_parsed = True
                        except json.JSONDecodeError:
                            pass
                            
                        # Only continue if we haven't successfully parsed JSON yet
                        if not json_parsed:
                            # Look for JSON object pattern within the string
                            json_pattern = r'\{[\s\S]*\}'
                            match = re.search(json_pattern, content)
                            
                            if match:
                                potential_json = match.group(0)
                                try:
                                    json_data = json.loads(potential_json)
                                    display_engagement_content(json_data)
                                    json_parsed = True
                                except json.JSONDecodeError:
                                    pass
                    
                    except Exception as e:
                        st.warning(f"Error parsing Audience Engagement data: {str(e)}")
                    
                    # If we get here and haven't parsed JSON successfully, display the cleaned string
                    if not json_parsed:
                        cleaned = re.sub(r"Audience Engagement \([\-\d.]+/5\)\*\*:?:?\s*", "", content)
                        # Remove "Audience Engagement (3/5)" line that might be in the string
                        cleaned = re.sub(r"Audience Engagement \([\d\.]+/5\)\s*\n", "", cleaned)
                        st.markdown(cleaned)
                elif isinstance(content, (dict, list)):
                    # Display structured data nicely instead of raw JSON
                    display_engagement_content(content)
                else:
                    st.warning("Audience engagement analysis data is not displayable.")
            else:
                st.markdown("No audience engagement analysis available.")

    # Technical Performance section
    if ai_analysis.get("technical_performance"):
        st.markdown("## ⚙️ Technical Performance")
        content = ai_analysis["technical_performance"]
        factors = analysis.get("scores", {}).get("factors", {})
        tech_score = None
        try:
            tech_score = float(factors.get("technical_quality", {}).get("score", None))
            tech_score = min(5.0, tech_score)
        except (TypeError, ValueError):
            tech_score = None
        # Define technical content display function first
        def display_technical_content(content, prefix=""):
            if isinstance(content, dict):
                # First try to extract key fields of interest
                important_fields = [
                    "video_quality", 
                    "audio_quality", 
                    "length_appropriateness", 
                    "accessibility"
                ]
                recommendations = None
                
                # Special handling for recommendations
                if "recommendations" in content:
                    recommendations = content["recommendations"]
                
                # Display main fields
                for key in important_fields:
                    if key in content and content[key]:
                        key_display = key.replace("_", " ").title()
                        st.markdown(f"**{key_display}:** {content[key]}")
                
                # Display recommendations last
                if recommendations:
                    st.markdown("**Recommendations:**")
                    if isinstance(recommendations, list):
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.markdown(recommendations)
                        
                # Handle any other keys
                for key, value in content.items():
                    if key not in important_fields and key != "recommendations" and key != "score":
                        key_display = key.replace("_", " ").title()
                        if isinstance(value, (dict, list)):
                            st.markdown(f"**{key_display}:**")
                            display_technical_content(value, prefix + "  ")
                        else:
                            st.markdown(f"**{key_display}:** {value}")
            
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, (dict, list)):
                        display_technical_content(item, prefix + "  ")
                    else:
                        st.markdown(f"{prefix}- {item}")
            else:
                st.markdown(f"{prefix}{content}")

        with st.expander("View Technical Performance Details", expanded=False):
            if tech_score is not None:
                stars = "⭐" * int(tech_score) + "☆" * (5 - int(tech_score))
                st.markdown(
                    f"**Technical Performance Score: {stars} ({tech_score:.1f}/5)**"
                )
            else:
                st.markdown("**Technical Performance Score: N/A**")
            
            # Debug statements removed - UI is now clean
            
            if content:
                # Skip string display that contains raw JSON - only use our formatter
                if isinstance(content, (int, float)):
                    st.markdown(str(content))
                elif isinstance(content, str):
                    # Try to parse JSON from string
                    try:
                        # Find potential JSON in the string
                        import json
                        import re
                        
                        # First check if the entire string is valid JSON
                        json_parsed = False
                        try:
                            json_data = json.loads(content)
                            display_technical_content(json_data)
                            json_parsed = True
                        except json.JSONDecodeError:
                            pass
                            
                        # Only continue if we haven't successfully parsed JSON yet
                        if not json_parsed:
                            # Look for JSON object pattern within the string
                            json_pattern = r'\{[\s\S]*\}'
                            match = re.search(json_pattern, content)
                            
                            if match:
                                potential_json = match.group(0)
                                try:
                                    json_data = json.loads(potential_json)
                                    display_technical_content(json_data)
                                    json_parsed = True
                                except json.JSONDecodeError:
                                    pass
                    
                    except Exception as e:
                        st.warning(f"Error parsing Technical Performance data: {str(e)}")
                    
                    # If we get here and haven't parsed JSON successfully, display the cleaned string
                    if not json_parsed:
                        cleaned = re.sub(r"Technical Performance \([\-\d.]+/5\)\*\*:?:?\s*", "", content)
                        # Remove "Technical Performance (3/5)" line that might be in the string
                        cleaned = re.sub(r"Technical Performance \([\d\.]+/5\)\s*\n", "", cleaned)
                        st.markdown(cleaned)
                elif isinstance(content, (dict, list)):
                    # Display structured data nicely instead of raw JSON
                    display_technical_content(content)
                else:
                    st.warning("Technical performance analysis data is not displayable.")
            else:
                st.markdown("No technical performance analysis available.")

    # Display Optimization Recommendations in a separate section
    if "recommendations" in analysis and analysis["recommendations"]:
        recommendations = analysis["recommendations"]
        if (
            "overall_recommendations" in recommendations
            and recommendations["overall_recommendations"]
        ):
            st.markdown("## 💡 Optimization Recommendations")

            # Display overall recommendations
            for i, rec in enumerate(recommendations["overall_recommendations"]):
                st.markdown(f"**{i+1}.** {rec}")

            # Factor-specific recommendations
            if "factor_recommendations" in recommendations:
                with st.expander("See detailed recommendations by factor"):
                    for factor_name, factor_data in recommendations[
                        "factor_recommendations"
                    ].items():
                        if (
                            factor_name != "ai_suggestions"
                        ):  # Skip AI suggestions as they're already in overall
                            factor_display_name = factor_name.replace("_", " ").title()
                            st.markdown(
                                f"**{factor_display_name}** ({factor_data.get('description', '')})"
                            )
                            for rec in factor_data.get("recommendations", []):
                                st.markdown(f"- {rec}")

    # Display related videos if available
    if "related_videos" in analysis and analysis["related_videos"]:
        st.subheader("🔁 Related Videos")

        # Create columns for related videos
        cols = st.columns(3)

        for i, video in enumerate(
            analysis["related_videos"][:6]
        ):  # Show up to 6 related videos
            with cols[i % 3]:
                # Create a card-like display for each related video
                st.markdown(
                    f"[**{video.get('title', 'Unknown')}**]({video.get('url', '')})"
                )
                # Format view count with commas
                view_count = video.get("view_count", "0")
                try:
                    view_count = f"{int(view_count):,}"
                except (ValueError, TypeError):
                    pass  # Keep original if not convertible to int
                st.caption(
                    f"👁️ {view_count} views • ⏱️ {video.get('formatted_duration', 'Unknown')}"
                )

                if "thumbnail_url" in video:
                    st.image(video["thumbnail_url"], use_container_width=True)

    # No duplicate sections needed - the recommendations and related videos are already displayed above


def clean_response_text(response, for_chat=True):
    """Extract plain text from a potentially structured response.
    
    Args:
        response: A response that could be plain text, a dict with 'text' key,
                 or a string representation of such a dict
        for_chat: If True, process for chat display (extract text from structures).
                  If False, preserve structured data (for analysis).
                 
    Returns:
        str or dict: Cleaned response - plain text for chat or preserved structure for analysis
    """
    # Don't process structured data if not for chat display
    if not for_chat and isinstance(response, dict):
        print(f"[clean_response_text] Preserving structured data for analysis: {type(response)}")
        return response
    
    # Return as is if None or empty
    if not response:
        return response
        
    # Handle dictionary response for chat display
    if isinstance(response, dict) and "text" in response:
        print(f"[clean_response_text] Extracting text from dict: {response.keys()}")
        return response["text"]
        
    # Handle string representation of dictionary for chat display
    if isinstance(response, str):
        # Check if it looks like a JSON object with a text key
        if response.startswith('{') and response.endswith('}') and '"text"' in response:
            try:
                # import json # Assuming json is imported at the top of the file
                parsed = json.loads(response)
                if isinstance(parsed, dict) and "text" in parsed:
                    print(f"[clean_response_text] Extracted text from JSON string")
                    return parsed["text"]
            except json.JSONDecodeError:
                # Not valid JSON, or JSON doesn't have 'text' key
                print(f"[clean_response_text] Failed to parse as JSON or extract text, trying literal_eval.")
                pass # Fall through to literal_eval or return original

        # Try ast.literal_eval for strings that look like Python dicts/lists
        # (e.g., "{'text': '...'}")
        if response.strip().startswith('{') and response.strip().endswith('}'):
            try:
                # import ast # Assuming ast is imported at the top of the file
                parsed_literal = ast.literal_eval(response)
                if isinstance(parsed_literal, dict) and "text" in parsed_literal:
                    print(f"[clean_response_text] Extracted text using ast.literal_eval from dict-like string.")
                    return parsed_literal["text"]
            except (ValueError, SyntaxError, TypeError) as e:
                # ast.literal_eval failed, it's likely just a plain string
                print(f"[clean_response_text] ast.literal_eval failed ({e}), treating as plain string.")
                pass # Fall through to return original response
    
    # Return original if no transformation needed or possible
    print(f"[clean_response_text] No specific cleaning applied, returning original: {type(response)}")
    return response


def display_chat():
    """Display the chat history from st.session_state.messages."""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            # Content should already be cleaned plain text before being added to messages
            content_to_display = message.get("content", "")
            # However, for robustness, especially with older history, let's clean it again if it's complex.
            # But primarily, we expect it to be simple string content.
            # Always attempt to clean, as a string itself might be a stringified dict.
            # clean_response_text will return plain strings as-is if no cleaning is needed.
            # Ensure the input to clean_response_text is a string if it's not a dict.
            if isinstance(content_to_display, dict):
                content_to_display = clean_response_text(content_to_display, for_chat=True)
            else:
                content_to_display = clean_response_text(str(content_to_display), for_chat=True)
            st.markdown(content_to_display)


def save_current_conversation():
    """Save the current state of the active chat to the conversation history."""
    # Check if saving should be skipped for this chat (important for New Chat flow)
    if st.session_state.get("skip_save_for_current_chat", False):
        logger.info("Skipping save due to skip_save_for_current_chat flag.")
        # Reset the flag after using it
        st.session_state.skip_save_for_current_chat = False
        return
        
    current_chat_id = st.session_state.get("current_chat_id")
    if not current_chat_id:
        logger.warning("save_current_conversation called without current_chat_id.")
        return

    current_messages = st.session_state.get("messages", [])
    current_analysis_data = st.session_state.get("current_analysis")
    current_title = st.session_state.get("current_chat_title", f"Chat {current_chat_id[:8]}")

    # Only save if there's more than the initial assistant message OR if there's analysis data
    # The initial message is from the assistant.
    has_meaningful_content = len(current_messages) > 1 or current_analysis_data is not None

    if not has_meaningful_content:
        logger.info(f"Skipping save for chat {current_chat_id} as it has no meaningful content.")
        return

    current_chat_data = {
        "id": current_chat_id,
        "title": current_title,
        "messages": sanitize_for_json(current_messages),
        "analysis": sanitize_for_json(current_analysis_data),
        "timestamp": datetime.now().isoformat(),
    }

    history = st.session_state.get("conversation_history", [])
    found_in_history = False
    for i, conv_in_history in enumerate(history):
        if conv_in_history.get("id") == current_chat_id:
            history[i] = current_chat_data
            found_in_history = True
            logger.info(f"Updated existing chat in history: {current_chat_id}")
            break
    
    if not found_in_history:
        history.insert(0, current_chat_data) # Prepend to show most recent first
        logger.info(f"Added new chat to history: {current_chat_id}")

    st.session_state.conversation_history = history[:MAX_HISTORY_ITEMS]
    persist_chat_history(st.session_state.conversation_history)


def generate_contextual_response(prompt: str, analysis: Dict[str, Any]) -> str:
    """
    Generate a contextual response based on user input and video analysis data.

    Args:
        prompt: User's query or prompt
        analysis: Dictionary containing video analysis data

    Returns:
        A markdown-formatted response to the user's query
    """
    # Debug output to console only (not visible to users)
    print("\n" + "=" * 50)
    print(f"generate_contextual_response called for prompt: {prompt}")
    print(f"Analysis data available: {list(analysis.keys()) if analysis else 'None'}")
    print("=" * 50)

    # Ensure we have valid analysis data
    if not analysis or not isinstance(analysis, dict):
        return "I don't have enough information about the video to answer that question. Please try analyzing a video first."

    # Extract metadata and analysis data for easier access
    metadata = analysis.get("metadata", {})
    analysis_data = analysis.get("analysis", {})
    scores = analysis.get("scores", {})
    recommendations = analysis.get("recommendations", {})

    # Safely get thumbnail_analysis from the analysis data
    thumbnail_analysis = analysis.get("thumbnail_analysis", {})

    # Use Gemini to generate a comprehensive response
    # Helper function to be run in a thread
    def _call_gemini_and_extract_text(gemini_client_instance, current_comprehensive_prompt, current_generation_config, original_timestamp_for_prompt_id):
        # This function encapsulates the original Gemini call and response processing logic.
        # It's designed to be executed in a separate thread.
        try:
            print(f"THREAD: Sending comprehensive prompt to Gemini with Response ID: {original_timestamp_for_prompt_id}")
            # Original call to Gemini (from line 2004)
            response_obj = gemini_client_instance.generate_content(
                current_comprehensive_prompt, generation_config=current_generation_config
            )

            # Original text extraction logic (lines 2009-2031 of original file)
            try:
                if hasattr(response_obj, "text"):
                    response_text = response_obj.text
                elif hasattr(response_obj, "parts") and len(response_obj.parts) > 0:
                    response_text = response_obj.parts[0].text
                elif hasattr(response_obj, "candidates") and len(response_obj.candidates) > 0:
                    candidate = response_obj.candidates[0]
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and len(candidate.content.parts) > 0:
                        response_text = candidate.content.parts[0].text
                    else:
                        response_text = str(candidate)
                else:
                    response_text = str(response_obj)

                if not response_text or len(response_text.strip()) == 0:
                    response_text = "I couldn't generate a specific response to your question. Please try rephrasing or asking something else about this video."
            except Exception as e_extract:
                print(f"THREAD: Error extracting text from Gemini response: {str(e_extract)}")
                # Raise an error to be caught by future.result() in the main thread
                raise RuntimeError(f"Error extracting text from Gemini response: {str(e_extract)}") from e_extract

            # Original debug logging (lines 2041-2042)
            print(f"THREAD: Received response from Gemini with length: {len(response_text)}")
            print(f"THREAD: Response preview: {response_text[:100]}...")
            
            # Original enhanced parsing (lines 2050-2061)
            print(f"THREAD: Response before cleaning: {response_text[:100]}")
            if isinstance(response_text, str):
                if response_text.strip().startswith('{') and '"text"' in response_text:
                    try:
                        # Ensure json is available (it's imported at the top of streamlit_app.py)
                        parsed_json = json.loads(response_text)
                        if isinstance(parsed_json, dict) and 'text' in parsed_json:
                            response_text = parsed_json['text']
                            print("THREAD: Extracted text from JSON string")
                    except json.JSONDecodeError:
                        print("THREAD: Failed to parse as JSON, using as-is")
            elif isinstance(response_text, dict) and 'text' in response_text:
                response_text = response_text['text']
                print("THREAD: Extracted text from dictionary response")
            
            print(f"THREAD: Response after cleaning: {response_text[:100]}")
            return response_text # Return the final cleaned text
        except Exception as e_main_gemini_call:
            # If Gemini call itself fails (original lines 2064-2068)
            print(f"THREAD: Error generating response with Gemini: {str(e_main_gemini_call)}")
            # Raise an exception to be caught by future.result() in the main thread
            raise RuntimeError(f"Error generating response with Gemini: {str(e_main_gemini_call)}") from e_main_gemini_call

    # New logic using ThreadPoolExecutor
    prompt_key = prompt # Use the original user prompt as the key for the future
    
    # Create timestamp for tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Create the comprehensive prompt for Gemini API
    comprehensive_prompt = f"""
    # YOUTUBE VIDEO ASSISTANT - RESPONSE GENERATION
    Response ID: {timestamp}_query

    ## USER QUERY
    {prompt}

    ## VIDEO INFORMATION
    Title: {metadata.get('title', 'Unknown')}
    Channel: {metadata.get('channel_title', 'Unknown')}
    Duration: {metadata.get('duration', 'Unknown')}
    Views: {metadata.get('views', 'Unknown')}
    
    ## ANALYSIS
    Title Analysis: {analysis_data.get('title_analysis', {})}
    Thumbnail Analysis: {thumbnail_analysis}
    
    ## RECOMMENDATIONS
    {recommendations}

    ## TASK
    Using the video information and analysis provided above, answer the user's query in a helpful, accurate, and engaging way.
    Provide specific information from the analysis and video details.
    DO NOT include markdown code blocks in your response.
    Format your response naturally, as if you are having a conversation.
    When recommending improvements, be constructive and positive.
    """
    
    # Define generation config for consistent response
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    gemini_client_to_use = st.session_state.gemini_client

    if prompt_key in st.session_state.gemini_futures:
        future = st.session_state.gemini_futures[prompt_key]
        if future.done():
            try:
                response_text = future.result() # This will re-raise exceptions from the thread
                del st.session_state.gemini_futures[prompt_key] # Clean up the future
                # Log success to console only
                print(f"ASYNC_RESULT: Received response from Gemini with length: {len(response_text)}")
                print(f"ASYNC_RESULT: Response preview: {response_text[:100]}...")
                return response_text
            except Exception as e:
                del st.session_state.gemini_futures[prompt_key] # Ensure cleanup on error
                print(f"ASYNC_ERROR: Error getting result from Gemini future: {str(e)}", exc_info=True)
                # Return the error message similar to the original error handling (lines 2066-2068)
                return f"I encountered an error processing your request (async). Error details: {str(e)}"
        else:
            print(f"Gemini response for '{prompt_key}' is still pending.")
            return "PENDING_RESPONSE" # Special status to indicate processing
    else:
        print(f"Submitting Gemini call for '{prompt_key}' to thread pool.")
        
        # 'timestamp' (original line 1938) is used in 'comprehensive_prompt'.
        # 'comprehensive_prompt' and 'generation_config' are already defined in the outer scope
        # of generate_contextual_response, before the original 'try' block.
        future = st.session_state.thread_executor.submit(
            _call_gemini_and_extract_text,
            gemini_client_to_use,
            comprehensive_prompt, 
            generation_config,
            timestamp # This is the original timestamp from line 1938, used in the prompt's Response ID
        )
        st.session_state.gemini_futures[prompt_key] = future
        return "PENDING_RESPONSE" # Special status to indicate processing


def process_user_input(new_prompt):
    """Process the user's input after it's been submitted."""
    if new_prompt:
        print(f"[DEBUG_TRACE] Processing new prompt: '{new_prompt}'")
        st.session_state.messages.append({"role": "user", "content": new_prompt})

        if "youtube.com" in new_prompt or "youtu.be" in new_prompt:
            print(f"[DEBUG_TRACE] Prompt is a URL. Calling analyze_video for: {new_prompt}")
            analyze_video(new_prompt) 
            # analyze_video should handle its own state, messages, and reruns.
            # The save_current_conversation and rerun below will catch any state changes it makes if it doesn't rerun itself.
        elif "current_analysis" in st.session_state and st.session_state.current_analysis:
            print("[DEBUG_TRACE] Prompt is a question and current_analysis IS available.")
            st.session_state.current_processing_prompt = new_prompt
            st.session_state.is_generating_response = True
            st.session_state.messages.append({"role": "assistant", "content": "🤔 Thinking...", "type": "thinking_indicator"})
            print("[DEBUG_TRACE] Set flags for response generation, appended 'Thinking...'. Will save and rerun.")
        else: # Question, but no current_analysis
            print("[DEBUG_TRACE] Prompt is a question but current_analysis IS NOT available.")
            no_analysis_response = "No current analysis. Please provide a YouTube URL first so I can answer questions about it."
            st.session_state.messages.append(
                {"role": "assistant", "content": no_analysis_response, "type": "info"}
            )
            logger.debug(f"[DEBUG_TRACE] Appended 'no analysis' message: {no_analysis_response[:100]}...")

        # Common save and rerun for all new prompt scenarios.
        save_current_conversation()
        print(f"[DEBUG_TRACE] First pass: Conversation saved. Rerunning. Messages count: {len(st.session_state.messages)}")
        st.rerun()

def handle_user_input():
    """Handle user input and generate responses."""
    # Part 1: Check for and handle ongoing response generation
    if st.session_state.get("is_generating_response", False) and st.session_state.get("current_processing_prompt"):
        print("[DEBUG_TRACE] Second pass: is_generating_response is True and current_processing_prompt exists.")
        prompt_to_process = st.session_state.current_processing_prompt
        
        response_content = None
        try:
            print(f"[DEBUG_TRACE] Attempting to call generate_contextual_response for: {prompt_to_process}")
            if not st.session_state.current_analysis:
                print("[ERROR_TRACE] current_analysis is missing during second pass of response generation. Aborting response.")
                # Pop the 'Thinking...' message first, if it exists as the last message
                if st.session_state.messages and st.session_state.messages[-1].get("type") == "thinking_indicator":
                    st.session_state.messages.pop()
                    print("[DEBUG_TRACE] Popped 'Thinking...' message before appending error.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "An issue occurred. Could not retrieve video analysis to generate response. Please try re-analyzing the video or asking a new question.", 
                    "type": "error"
                })
                st.session_state.is_generating_response = False # Reset flags
                st.session_state.current_processing_prompt = None
                save_current_conversation()
                print("[DEBUG_TRACE] Second pass: Error due to missing analysis. Conversation saved. Rerunning.")
                st.rerun()
                return # Exit after handling the error

            analysis_data = st.session_state.current_analysis
            response_content = generate_contextual_response(prompt_to_process, analysis_data)
            print(f"[DEBUG_TRACE] Raw response from generate_contextual_response: {str(response_content)[:200]}...")

            if response_content == "PENDING_RESPONSE":
                print("[DEBUG_TRACE] Second pass: Response is PENDING. Rerunning to check later.")
                # The 'Thinking...' message is already displayed via display_chat and current_processing_prompt.
                # No need to change current_processing_prompt or is_generating_response flags here.
                # No need to save conversation yet.
                st.rerun() # Trigger a rerun to re-check the future status
                return     # Exit this execution of handle_user_input
            
            st.session_state.last_gemini_response = response_content # Store actual response if not pending
        except Exception as e:
            print(f"[DEBUG_TRACE] Error during generate_contextual_response: {str(e)}")
            response_content = f"Error generating response: {str(e)}" # This will be cleaned
        
        cleaned_response = ""
        if response_content:
            print(f"[DEBUG_TRACE] Raw response before cleaning: {str(response_content)[:200]}...")
            cleaned_response = clean_response_text(response_content, for_chat=True)
            print(f"[DEBUG_TRACE] Response after cleaning: {str(cleaned_response)[:100]}...")
        else:
            print("[DEBUG_TRACE] No response_content, using default error message.")
            cleaned_response = "Sorry, I couldn't generate a response at this time."

        # Pop the 'Thinking...' message if it's the last one
        if st.session_state.messages and st.session_state.messages[-1].get("type") == "thinking_indicator":
            st.session_state.messages.pop()
            print("[DEBUG_TRACE] Popped 'Thinking...' message.")
        
        st.session_state.messages.append(
            {"role": "assistant", "content": cleaned_response, "type": "contextual_response"}
        )
        logger.debug(f"[DEBUG_TRACE] Appended final assistant message: {cleaned_response[:100]}...")
        
        st.session_state.is_generating_response = False
        st.session_state.current_processing_prompt = None
        print("[DEBUG_TRACE] Reset is_generating_response and current_processing_prompt.")

        save_current_conversation()
        print(f"[DEBUG_TRACE] Second pass: Conversation saved. Rerunning. Messages count: {len(st.session_state.messages)}")
        st.rerun()
        return # Exit after handling the ongoing response generation

    # Part 2: Handle new user input from chat_input (first pass)
    if st.session_state.get("just_created_new_chat", False):
        st.session_state.just_created_new_chat = False # Consume the flag
        print("[DEBUG_TRACE] 'just_created_new_chat' is True. Continuing to render input box but will ignore any carried-over input.")
        # Instead of returning early, we'll continue to render the input box
        # but we'll ignore any inputs that might have carried over
    
    # User's own message is appended here, but displayed by display_chat on rerun.
    if new_prompt := st.chat_input("Paste a YouTube URL or ask a question...", key="chat_input"):
        print(f"[DEBUG_TRACE] First pass: New prompt received: '{new_prompt}'")
        st.session_state.messages.append({"role": "user", "content": new_prompt})

        if "youtube.com" in new_prompt or "youtu.be" in new_prompt:
            print(f"[DEBUG_TRACE] Prompt is a URL. Calling analyze_video for: {new_prompt}")
            analyze_video(new_prompt) 
            # analyze_video should handle its own state, messages, and reruns.
            # The save_current_conversation and rerun below will catch any state changes it makes if it doesn't rerun itself.
        elif "current_analysis" in st.session_state and st.session_state.current_analysis:
            print("[DEBUG_TRACE] Prompt is a question and current_analysis IS available.")
            st.session_state.current_processing_prompt = new_prompt
            st.session_state.is_generating_response = True
            st.session_state.messages.append({"role": "assistant", "content": "🤔 Thinking...", "type": "thinking_indicator"})
            print("[DEBUG_TRACE] Set flags for response generation, appended 'Thinking...'. Will save and rerun.")
        else: # Question, but no current_analysis
            print("[DEBUG_TRACE] Prompt is a question but current_analysis IS NOT available.")
            no_analysis_response = "No current analysis. Please provide a YouTube URL first so I can answer questions about it."
            st.session_state.messages.append(
                {"role": "assistant", "content": no_analysis_response, "type": "info"}
            )
            logger.debug(f"[DEBUG_TRACE] Appended 'no analysis' message: {no_analysis_response[:100]}...")

        # Common save and rerun for all new prompt scenarios.
        save_current_conversation()
        print(f"[DEBUG_TRACE] First pass: Conversation saved. Rerunning. Messages count: {len(st.session_state.messages)}")
        st.rerun()
    else:
        # This else corresponds to `if new_prompt := st.chat_input(...)`
        # It means no new input was submitted in this specific script run.
        # This is normal during the "second pass" after 'Thinking...' is shown,
        # as the script reruns and `st.chat_input` returns None.
        # The logic in "Part 1" handles the ongoing response generation.
        print("[DEBUG_TRACE] No new prompt received from st.chat_input (normal for second pass or no interaction).")


def analyze_video(video_url: str):
    """
    Analyze a YouTube video and display results.

    Args:
        video_url (str): URL of the YouTube video to analyze
    """
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing video..."):
            try:
                # Step 1: Analyze the video using our VideoAnalyzer
                st.text("Step 1/3: Fetching video data and analyzing content...")
                analysis_result = st.session_state.analyzer.analyze_video(
                    video_url, include_related=True, include_thumbnail=True
                )

                # Check for errors
                if "error" in analysis_result:
                    error_msg = f"Error analyzing video: {analysis_result['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"❌ {error_msg}",
                            "video_url": video_url,
                            "type": "error",
                        }
                    )
                    return

                # Step 2: Score the video
                st.text("Step 2/3: Calculating performance scores...")
                score_result = st.session_state.scoring_engine.score_video(
                    analysis_result["metadata"], analysis_result["analysis"]
                )

                # Step 3: Generate recommendations
                st.text("Step 3/3: Generating optimization recommendations...")
                recommendation_result = (
                    st.session_state.recommendation_engine.generate_recommendations(
                        analysis_result["metadata"],
                        analysis_result["analysis"],
                        score_result,
                    )
                )

                # Combine all results
                complete_analysis = {
                    "video_id": analysis_result["video_id"],
                    "metadata": analysis_result["metadata"],
                    "analysis": analysis_result["analysis"],
                    "scores": score_result,
                    "recommendations": recommendation_result,
                }

                # Add thumbnail if available
                if "thumbnail" in analysis_result:
                    complete_analysis["thumbnail"] = analysis_result["thumbnail"]

                # Add title analysis if available
                if "title_analysis" in analysis_result:
                    complete_analysis["title_analysis"] = analysis_result[
                        "title_analysis"
                    ]
                    print("Added title_analysis to complete_analysis")

                # Add thumbnail analysis if available
                if "thumbnail_analysis" in analysis_result:
                    complete_analysis["thumbnail_analysis"] = analysis_result[
                        "thumbnail_analysis"
                    ]
                    print("Added thumbnail_analysis to complete_analysis")

                # Add related videos if available
                if "related_videos" in analysis_result:
                    complete_analysis["related_videos"] = analysis_result[
                        "related_videos"
                    ]

                # Debug the analysis result structure
                print("Analysis result keys:", analysis_result.keys())
                if "thumbnail" in analysis_result:
                    print("Thumbnail data:", analysis_result["thumbnail"])
                if "thumbnail_analysis" in analysis_result:
                    print("Thumbnail analysis available:", True)

                # Store the complete analysis in session state for both display and contextual responses
                # Make sure we don't accidentally clean the structured data
                print(f"Storing analysis with type: {type(complete_analysis)} and keys: {complete_analysis.keys() if hasattr(complete_analysis, 'keys') else 'no keys'}")
                
                # Explicitly protect the analysis from being processed as a chat response
                st.session_state.analysis = copy.deepcopy(complete_analysis)
                st.session_state.current_analysis = copy.deepcopy(complete_analysis)
                
                # Verify the stored analysis maintains its structure
                if 'analysis' in st.session_state:
                    print(f"Verified: Analysis stored with type: {type(st.session_state.analysis)}")
                    if hasattr(st.session_state.analysis, 'keys'):
                        print(f"Verified: Analysis contains keys: {list(st.session_state.analysis.keys())}")
                else:
                    print("WARNING: Analysis not properly stored in session state")

                # Debug the complete analysis structure
                print("Complete analysis keys:", complete_analysis.keys())

                # Add a timestamp to force cache refresh
                st.session_state.last_analysis_time = datetime.now().timestamp()

                # Create a brief summary for the chat interface
                video_title = analysis_result["metadata"].get("title", "Unknown video")
                channel = analysis_result["metadata"].get(
                    "channel_title", "Unknown channel"
                )

                # Format the duration and published date
                duration = format_duration(
                    analysis_result["metadata"].get("duration", "Unknown")
                )
                published_date = format_published_date(
                    analysis_result["metadata"].get("published_at", "Unknown")
                )

                # Get comment count if available
                comment_count = analysis_result["metadata"].get(
                    "comment_count", "Unknown"
                )

                response = f"🎬 Video Analysis Complete\n\n"
                response += f"**Title**: {video_title}\n\n"
                response += f"**Channel**: {channel}\n\n"
                response += f"**Duration**: {duration}\n\n"
                response += f"**Published**: {published_date}\n\n"
                response += f"**Views**: {analysis_result['metadata'].get('view_count', 'Unknown')}\n\n"
                response += f"**Comments**: {comment_count}\n\n"
                response += f"**Overall Score**: {score_result.get('overall_score', 'N/A')}/5\n\n"

                # Add top 3 recommendations
                response += "### 💡 Top Recommendations:\n"
                top_recommendations = recommendation_result.get(
                    "overall_recommendations", []
                )
                for i, rec in enumerate(top_recommendations[:3]):
                    response += f"{i+1}. {rec}\n"

                response += (
                    "\n*View the Analysis tab for complete details and scores.*\n"
                )
                response += "*Ask follow-up questions about this video in the chat!*"

                # Add analysis to chat
                st.markdown(response)

                # Add to messages
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "video_url": video_url,
                        "type": "video_analysis",
                    }
                )

                # Print debug information
                print(
                    f"Analysis data stored in session state with keys: {list(complete_analysis.keys())}"
                )
                print(
                    f"Metadata available: {list(complete_analysis.get('metadata', {}).keys())}"
                )
                print(f"Current analysis set and ready for contextual responses")

                # Don't rerun here - let the display_chat function handle the rerun
                # This prevents double reruns when processing YouTube URLs

            except Exception as e:
                error_msg = f"Error analyzing video: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ {error_msg}",
                        "video_url": video_url,
                        "type": "error",
                    }
                )


def main():
    """Main function to run the Streamlit app."""
    # Setup
    init_session_state()
    setup_ui()

    # Create tabs for chat and analysis
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analysis"])

    # Display chat in the Chat tab
    with tab1:
        display_chat()
        handle_user_input() # Call to render chat input box and handle its logic

    # Add a debug message to verify the app is running properly
    st.sidebar.caption(f"Last app refresh: {datetime.now().strftime('%H:%M:%S')}")
    print(f"App refreshed at {datetime.now().strftime('%H:%M:%S')}")

    # Display analysis in the Analysis tab if available
    with tab2:
        # Add debug information about analysis data
        print(f"Analysis data available: {'analysis' in st.session_state}")
        if 'analysis' in st.session_state:
            print(f"Analysis data type: {type(st.session_state.analysis)}")
            if st.session_state.analysis:
                print(f"Analysis keys: {st.session_state.analysis.keys() if hasattr(st.session_state.analysis, 'keys') else 'No keys method'}")
        
        if st.session_state.analysis:
            # Add a message to confirm we're trying to display analysis
            print("Attempting to display analysis results...")
            display_analysis_results(st.session_state.analysis)
        else:
            st.info(
                "No analysis results available yet. Paste a YouTube URL in the chat to analyze a video."
            )
            st.caption(
                "Once analysis is complete, results will appear here automatically."
            )


if __name__ == "__main__":
    main()
