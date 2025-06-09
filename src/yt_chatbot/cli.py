"""
Command-line interface for the YouTube Video Intelligence Chatbot.
"""

import argparse
import logging
import sys
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Import local modules
from yt_chatbot.api.youtube_client import YouTubeClient
from yt_chatbot.api.gemini_client import GeminiClient
from yt_chatbot.database.supabase_client import DatabaseClient

# Initialize clients
youtube_client = YouTubeClient()
gemini_client = GeminiClient()
db_client = DatabaseClient()


def analyze_video(video_url: str, output_format: str = "text") -> None:
    """Analyze a YouTube video.

    Args:
        video_url: URL of the YouTube video to analyze
        output_format: Output format (text, json, csv)

    Returns:
        None: Outputs results to console
    """
    try:
        logger.info(f"Starting analysis of video: {video_url}")

        # Extract video ID
        video_id = youtube_client.extract_video_id(video_url)
        if not video_id:
            logger.error("Invalid YouTube URL")
            print("Error: Invalid YouTube URL")
            return

        # Get video metadata
        logger.info("Fetching video metadata...")
        metadata = youtube_client.get_video_metadata(video_id)
        if "error" in metadata:
            logger.error(f"Failed to fetch metadata: {metadata['error']}")
            print(f"Error: {metadata['error']}")
            return

        # Generate analysis using Gemini
        logger.info("Generating analysis...")
        prompt = f"Analyze this YouTube video: {metadata.get('title', '')}\n\n"
        prompt += f"Description: {metadata.get('description', '')[:1000]}..."

        analysis = gemini_client.generate_content(prompt)

        # Format output based on requested format
        if output_format == "json":
            import json

            result = {
                "video_id": video_id,
                "title": metadata.get("title"),
                "analysis": analysis,
            }
            print(json.dumps(result, indent=2))
        elif output_format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Field", "Value"])
            writer.writerow(["Video ID", video_id])
            writer.writerow(["Title", metadata.get("title", "")])
            writer.writerow(["Analysis", analysis])
            print(output.getvalue())
        else:  # text format
            print("=" * 80)
            print(f"Analysis for: {metadata.get('title', 'Unknown')}")
            print("=" * 80)
            print(f"Video ID: {video_id}")
            print(f"Channel: {metadata.get('channelTitle', 'Unknown')}")
            print("\n" + "-" * 80)
            print("Analysis:")
            print(analysis)
            print("\n" + "=" * 80)

        logger.info("Analysis completed successfully")

    except Exception as e:
        error_msg = f"Error analyzing video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}")


def list_history(limit: int = 10) -> None:
    """List previous video analyses.

    Args:
        limit: Maximum number of analyses to show

    Returns:
        None: Outputs history to console
    """
    try:
        logger.info(f"Fetching last {limit} analyses from database")

        # Get history from database
        history = db_client.get_video_analyses(limit=limit)

        if not history:
            print("No analysis history found.")
            return

        print("\n" + "=" * 80)
        print(f"Last {len(history)} Video Analyses")
        print("=" * 80)

        for i, analysis in enumerate(history, 1):
            print(f"\n{i}. {analysis.get('video_title', 'Untitled')}")
            print(f"   Video ID: {analysis.get('video_id')}")
            print(f"   Analyzed: {analysis.get('created_at', 'Unknown')}")

        print("\n" + "=" * 80)

    except Exception as e:
        error_msg = f"Error fetching history: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}")


def export_analysis(analysis_id: str, output_file: str) -> None:
    """Export a video analysis to a file.

    Args:
        analysis_id: ID of the analysis to export
        output_file: Path to save the export

    Returns:
        None: Saves analysis to specified file
    """
    try:
        logger.info(f"Exporting analysis {analysis_id} to {output_file}")

        # Get analysis from database
        analysis = db_client.get_video_analysis(analysis_id)
        if not analysis:
            logger.error(f"Analysis with ID {analysis_id} not found")
            print(f"Error: Analysis with ID {analysis_id} not found")
            return

        # Determine output format from file extension
        if output_file.lower().endswith(".json"):
            import json

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        elif output_file.lower().endswith((".csv", ".tsv")):
            import csv

            delimiter = "\t" if output_file.lower().endswith(".tsv") else ","
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter)
                # Write header
                writer.writerow(["Field", "Value"])
                # Write data
                for key, value in analysis.items():
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    writer.writerow([key, value])
        else:  # Default to text
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Analysis Report\n")
                f.write("=" * 80 + "\n\n")
                for key, value in analysis.items():
                    f.write(f"{key}:\n{value}\n\n")

        logger.info(f"Successfully exported analysis to {output_file}")
        print(f"Successfully exported analysis to {output_file}")

    except Exception as e:
        error_msg = f"Error exporting analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Intelligence Chatbot - Analyze and optimize YouTube videos"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a YouTube video")
    analyze_parser.add_argument("video_url", help="URL of the YouTube video to analyze")
    analyze_parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format",
    )

    # History command
    history_parser = subparsers.add_parser("history", help="View analysis history")
    history_parser.add_argument(
        "--limit", type=int, default=10, help="Number of analyses to show"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export an analysis")
    export_parser.add_argument("analysis_id", help="ID of the analysis to export")
    export_parser.add_argument("output_file", help="Path to save the export")

    # Version command
    subparsers.add_parser("version", help="Show version information")

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    try:
        parsed_args = parse_args(args)

        if not parsed_args.command:
            print("No command specified. Use --help for usage information.")
            return 1

        # Initialize database connection
        db_client.connect()

        # Route to appropriate handler
        if parsed_args.command == "analyze":
            analyze_video(parsed_args.video_url, parsed_args.format)
        elif parsed_args.command == "history":
            list_history(parsed_args.limit)
        elif parsed_args.command == "export":
            export_analysis(parsed_args.analysis_id, parsed_args.output_file)
        elif parsed_args.command == "version":
            from yt_chatbot import __version__

            print(f"YouTube Video Intelligence Chatbot v{__version__}")
            return 0
        else:
            print(f"Unknown command: {parsed_args.command}")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for Ctrl+C

    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        print(f"Error: {error_msg}", file=sys.stderr)
        return 1

    finally:
        # Ensure database connection is closed
        if "db_client" in locals():
            db_client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
