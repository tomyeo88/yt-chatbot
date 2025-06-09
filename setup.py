from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="yt-chatbot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered YouTube video analysis and optimization chatbot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yt-chatbot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    keywords="youtube ai chatbot analysis optimization",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/yt-chatbot/issues",
        "Source": "https://github.com/yourusername/yt-chatbot",
    },
    entry_points={
        "console_scripts": [
            "yt-chatbot=yt_chatbot.cli:main",
        ],
    },
)
