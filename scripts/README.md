# Scripts

This directory contains utility scripts for the YouTube Video Intelligence Chatbot.

## Available Scripts

### `test_setup.py`

Verifies that your development environment is properly set up by testing:

1. Required environment variables
2. YouTube API client connectivity
3. Google Gemini API client connectivity
4. Supabase database connection

**Usage:**
```bash
python scripts/test_setup.py
```

**Prerequisites:**
- Python 3.8+
- Required Python packages (install with `pip install -r requirements.txt`)
- `.env` file with all required environment variables

## Adding New Scripts

1. Create a new Python file in this directory
2. Add a docstring explaining the script's purpose and usage
3. Update this README with information about the new script
4. Make sure to handle errors gracefully and provide helpful feedback

## Best Practices

- Use the `src` package for shared functionality
- Keep scripts focused on a single task
- Include error handling and logging
- Document any required environment variables
- Follow the project's coding style guidelines
