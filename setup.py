#!/usr/bin/env python3
"""
Setup script for Patent Search Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A comprehensive semantic patent search system built with Qdrant vector database."

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    requirements_file = Path(__file__).parent / filename
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]
    return []

# Core requirements
install_requires = read_requirements("requirements.txt")

# Development requirements (optional)
dev_requires = read_requirements("requirements-dev.txt")

# Package version
version = "1.0.0"

setup(
    name="patent-search-tool",
    version=version,
    author="Patent Search Team",
    author_email="patent-search@company.com",
    description="Semantic patent search using Qdrant vector database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-company/patent-search-tool",
    license="MIT",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=install_requires,
    
    # Optional dependencies
    extras_require={
        "dev": dev_requires,
        "api": [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic[email]>=2.4.2"
        ],
        "monitoring": [
            "prometheus-client>=0.18.0",
            "structlog>=23.2.0"
        ],
        "visualization": [
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
            "plotly>=5.17.0"
        ],
        "all": dev_requires + [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "prometheus-client>=0.18.0",
            "matplotlib>=3.7.2"
        ]
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "patent-search=main:cli",
            "patent-search-cli=main:cli",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Database :: Database Engines/Servers",
        "Typing :: Typed",
    ],
    
    # Keywords for discovery
    keywords=[
        "patent", "search", "vector-database", "semantic-search", 
        "qdrant", "nlp", "machine-learning", "embeddings",
        "intellectual-property", "prior-art", "r-and-d"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://docs.company.com/patent-search",
        "Source": "https://github.com/your-company/patent-search-tool",
        "Bug Reports": "https://github.com/your-company/patent-search-tool/issues",
        "Funding": "https://github.com/sponsors/your-company",
    },
    
    # Package data
    package_data={
        "patent_search": [
            "py.typed",  # Indicates package supports type hints
        ],
    },
    
    # Data files
    data_files=[
        ("", ["LICENSE", "README.md"]),
    ],
)
