# 🎬 YouTube Video Intelligence Chatbot

A powerful AI-powered chatbot that provides in-depth analysis and optimization recommendations for YouTube videos using Google's Gemini AI. This tool helps content creators understand their video performance and get actionable insights.

> **Note**: This is an alpha version. Please report any issues you encounter.

## 🌟 Features

- **Comprehensive Video Analysis**: Get detailed insights on any YouTube video with structured JSON output
- **AI-Powered Recommendations**: Receive personalized optimization suggestions based on Gemini AI analysis
- **Structured Scoring**: 5-point rating system across key performance metrics using Gemini model with guidelines:
  - Hook Score (Title & Thumbnail Effectiveness) - 20%
  - Content Quality - 25%
  - SEO Optimization - 20%
  - Technical Quality - 15%
  - Engagement Metrics - 20%
- **Robust Data Validation**: Pydantic schemas ensure consistent, validated analysis results
- **Enhanced Chat System**: UUID-based conversation tracking with persistent history
- **Responsive Web Interface**: Built with Streamlit for easy access

## 📋 Requirements

- Python 3.8+
- YouTube Data API v3 key
- Google Gemini API key

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/yt-chatbot.git
   cd yt-chatbot
   ```

2. **Set up Python environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys and configuration.

## ⚙️ Configuration

### Required Environment Variables

```bash
# API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# App Settings
DEBUG=False
LOG_LEVEL=INFO
```

### Optional Settings

- `STREAMLIT_SERVER_PORT`: Port for the Streamlit app (default: 8501)
- `MAX_REQUESTS_PER_MINUTE`: API rate limiting (default: 60)
- `CACHE_TTL_HOURS`: Cache duration in hours (default: 24)

## 🚦 Running the Application

### Development Mode

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Production Deployment

For production, consider using:
- Gunicorn with a WSGI server
- Docker containerization
- Serverless deployment (AWS Lambda, Vercel, etc.)

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_thumbnail_analysis.py -v
```

## 🏗️ Project Structure

```
yt-chatbot/
├── docs/                     # Documentation
│   ├── performance_metrics_methodology.md
│   └── youtube_client_usage.md
├── scripts/                  # Utility scripts
│   ├── README.md
│   └── test_setup.py         # Environment setup verification
├── src/                      # Source code
│   ├── analysis/             # Video analysis logic
│   │   ├── recommendation_engine.py
│   │   ├── schemas.py         # Pydantic schemas for data validation
│   │   ├── scoring_engine.py
│   │   └── video_analyzer.py
│   ├── api/                  # API clients and integrations
│   │   ├── gemini_client.py   # Enhanced with robust logging
│   │   ├── gemini_client_guideline_scorer.py
│   │   └── youtube_client.py
│   ├── database/             # Database models and client
│   │   ├── models.py
│   │   └── supabase_client.py
│   ├── utils/                # Utility functions
│   │   ├── config.py
│   │   ├── formatters.py
│   │   └── guideline_loader.py
│   └── yt_chatbot/           # Main package
│       └── cli.py            # Command line interface
├── tests/                    # Test files
│   ├── test_basic.py         # Basic functionality tests
│   ├── test_gemini.py        # Gemini API tests
│   ├── test_schema_validation.py # Schema validation tests
│   ├── test_scoring_engine.py # Scoring engine tests
│   └── test_thumbnail_analysis.py # Thumbnail analysis tests
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
├── PLANNING.md               # Project planning document
├── README.md                 # This file
├── TASK.md                   # Task tracking
├── guideline.md              # YouTube content guidelines
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
└── streamlit_app.py          # Main Streamlit application with UUID-based chat tracking
```

## 🧪 Testing Your Setup

Run the setup test script to verify everything is working correctly:

```bash
python scripts/test_setup.py
```

## 📊 Features in Detail

### Video Analysis
- Content summarization and key points extraction with structured JSON output
- Performance metrics and engagement analysis using Gemini AI
- SEO optimization scoring based on metadata and discoverability
- Technical assessment (video/audio quality, captions, etc.)
- Competitive positioning analysis

### AI-Powered Recommendations
- Title optimization suggestions with specific emphasis points
- Thumbnail design improvements with visual element analysis
- Content structure and pacing recommendations
- Engagement and retention strategies
- Detailed content quality analysis with strengths and weaknesses

### Chat System
- UUID-based conversation tracking
- Persistent chat history with JSON storage
- Multi-chat support with load/save/delete functionality
- Context-aware responses based on video analysis

### User Experience
- Clean, intuitive chat interface
- Conversation history and persistence
- Exportable analysis reports
- Responsive design for all devices

## 🙏 Acknowledgments

- [Google Gemini AI](https://ai.google/)
- [Streamlit](https://streamlit.io/)
- [YouTube Data API](https://developers.google.com/youtube/v3)

---

Made by Tom Yeo