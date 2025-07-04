"""
Patent Search Tool
A comprehensive semantic patent search system built with Qdrant vector database.
"""

__version__ = "1.0.0"
__author__ = "Patent Search Team"
__email__ = "patent-search@company.com"

from .engine import PatentSearchEngine
from .models import Patent, SearchResult, SearchFilter, CollectionStats
from .processor import PatentDataProcessor
from .loader import PatentDataLoader

__all__ = [
    'PatentSearchEngine',
    'Patent',
    'SearchResult', 
    'SearchFilter',
    'CollectionStats',
    'PatentDataProcessor',
    'PatentDataLoader'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'patent-search-tool',
    'version': __version__,
    'description': 'Semantic patent search using Qdrant vector database',
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/your-company/patent-search-tool',
    'license': 'MIT',
    'keywords': ['patent', 'search', 'vector-database', 'semantic-search', 'qdrant'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
}

# Configuration defaults
DEFAULT_CONFIG = {
    'qdrant': {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'patents'
    },
    'model': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'device': 'cpu'
    },
    'search': {
        'default_limit': 10,
        'min_score': 0.5,
        'batch_size': 100
    }
}

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def get_default_config():
    """Get default configuration"""
    return DEFAULT_CONFIG.copy()
