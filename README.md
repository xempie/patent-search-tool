# Patent Search Tool

A comprehensive semantic patent search system built with Qdrant vector database for R&D teams to efficiently discover relevant patents and prior art.


## Features

- **Semantic Search**: Find patents by meaning, not just keywords
- **Multi-format Support**: XML, JSON, CSV patent data ingestion
- **Advanced Filtering**: Search by technology field, assignees, date ranges
- **Similarity Analysis**: Identify patents similar to existing ones
- **Batch Processing**: Efficiently index thousands of patents
- **Production Ready**: Optimized for large-scale patent databases
- **Real-time Updates**: Incremental patent indexing
- **Technology Classification**: Automatic IPC code-based categorization

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Qdrant server)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-company/patent-search-tool.git
cd patent-search-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Qdrant server**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. **Run the example**
```bash
python patent_search_engine.py
```

### Basic Usage

```python
from patent_search_engine import PatentSearchEngine, PatentDataLoader

# Initialize search engine
search_engine = PatentSearchEngine()

# Load sample data
data_loader = PatentDataLoader()
patents = data_loader.create_sample_patents(100)

# Index patents
indexed_count = search_engine.index_patents_batch(patents)
print(f"Indexed {indexed_count} patents")

# Perform semantic search
results = search_engine.semantic_search("machine learning neural networks", limit=10)
for patent in results:
    print(f"- {patent['title']} (Score: {patent['similarity_score']:.3f})")
```

## Documentation

### Core Components

#### PatentSearchEngine
Main search interface with semantic capabilities:

```python
# Initialize with custom settings
search_engine = PatentSearchEngine(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="patents",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Semantic search
results = search_engine.semantic_search("quantum computing algorithms", limit=5)

# Filtered search
results = search_engine.search_patents(
    query="artificial intelligence",
    technology_field="Machine Learning",
    assignees=["Google", "Microsoft"],
    date_range=("2020-01-01", "2024-12-31"),
    min_score=0.7
)

# Find similar patents
similar = search_engine.get_similar_patents("US10123456", limit=5)
```

#### Patent Data Structure
```python
@dataclass
class Patent:
    patent_id: str
    title: str
    abstract: str
    description: str
    claims: List[str]
    inventors: List[str]
    assignees: List[str]
    filing_date: str
    publication_date: str
    patent_number: str
    classification_codes: List[str]
    technology_field: str
    country: str
    status: str
```

#### Data Loading
```python
data_loader = PatentDataLoader()

# Load from CSV
patents = data_loader.load_from_csv("patents.csv")

# Load from JSON
patents = data_loader.load_from_json("patents.json")

# Create sample data for testing
sample_patents = data_loader.create_sample_patents(50)
```

### Search Types

#### 1. Semantic Search
Find patents by meaning and context:
```python
# Natural language queries
results = search_engine.semantic_search("deep learning computer vision")
results = search_engine.semantic_search("blockchain cryptocurrency distributed ledger")
results = search_engine.semantic_search("quantum error correction algorithms")
```

#### 2. Filtered Search
Combine semantic search with structured filters:
```python
results = search_engine.search_patents(
    query="renewable energy storage",
    technology_field="Electricity",
    date_range=("2020-01-01", "2024-12-31"),
    min_score=0.8
)
```

#### 3. Similar Patents
Find patents similar to a reference patent:
```python
similar_patents = search_engine.get_similar_patents("US10987654", limit=10)
```

### CSV Data Format

For CSV import, use the following column structure:
```csv
patent_id,title,abstract,description,claims,inventors,assignees,filing_date,publication_date,patent_number,classification_codes,technology_field,country,status
US10123456,"AI System","Abstract text","Description text","Claim 1|Claim 2","John Doe|Jane Smith","TechCorp","2023-01-01","2024-01-01","US10123456","G06F15/00|G06N3/00","Machine Learning","US","Published"
```

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Search Engine   │    │  Vector Store   │
│                 │    │                  │    │                 │
│ • XML Patents   │───>│ • Text Processing│───>│ • Qdrant DB     │
│ • JSON Files    │    │ • Vectorization  │    │ • HNSW Index    │
│ • CSV Data      │    │ • Semantic Search│    │ • Cosine Sim    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Vector Database Configuration

- **Distance Metric**: Cosine similarity
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Vector Dimension**: 384 (all-MiniLM-L6-v2)
- **Optimization**: Automatic segment optimization
- **Indexing**: Parallel processing with 4 threads

## Performance

### Benchmarks

| Operation           | Time  | Throughput      |
|---------------------|-------|-----------------|
| Single Patent Index | ~50ms | 20 patents/sec  |
| Batch Index (100)   | ~2s   | 50 patents/sec  |
| Semantic Search     | ~10ms | 100 queries/sec |
| Filtered Search     | ~15ms | 70 queries/sec  |

### Scalability

- **Storage**: Handles millions of patents
- **Memory**: 4GB RAM for 100K patents
- **Search**: Sub-second response times
- **Indexing**: Parallel batch processing

## Configuration

### Environment Variables

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=patents

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_SIZE=384

# Performance Settings
BATCH_SIZE=100
MAX_THREADS=4
```

### Advanced Configuration

```python
# Custom embedding model
search_engine = PatentSearchEngine(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Higher accuracy
)

# Production settings
search_engine = PatentSearchEngine(
    qdrant_host="your-qdrant-cluster.com",
    qdrant_port=6333,
    collection_name="production_patents"
)
```

## Error Handling

The system includes comprehensive error handling:

- **Connection Issues**: Automatic retry with exponential backoff
- **Data Validation**: Schema validation for all patent data
- **Indexing Errors**: Graceful handling of malformed patents
- **Search Failures**: Fallback to exact match search
- **Memory Management**: Streaming for large datasets

## Analytics & Monitoring

### Collection Statistics
```python
stats = search_engine.get_collection_stats()
print(f"Total patents: {stats['total_patents']}")
print(f"Technology distribution: {stats['technology_fields']}")
```

### Performance Monitoring
```python
# Built-in logging
import logging
logging.basicConfig(level=logging.INFO)

# Custom metrics
search_time = time.time()
results = search_engine.semantic_search("query")
print(f"Search took: {time.time() - search_time:.3f}s")
```

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python tests/performance_test.py
```

## Data Pipeline

### Batch Processing
```python
# Process large patent datasets
def process_patent_batch(file_path: str, batch_size: int = 1000):
    loader = PatentDataLoader()
    patents = loader.load_from_csv(file_path)
    
    for i in range(0, len(patents), batch_size):
        batch = patents[i:i + batch_size]
        search_engine.index_patents_batch(batch)
        print(f"Processed batch {i//batch_size + 1}")
```

### Real-time Updates
```python
# Add new patents as they become available
new_patent = Patent(...)
search_engine.index_patent(new_patent)
```

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
```

4. **Run tests**
```bash
pytest
```

5. **Submit a pull request**

## Use Cases

### R&D Teams
- **Prior Art Search**: Find existing patents before filing
- **Technology Landscape**: Analyze competitor patents
- **Innovation Gaps**: Identify under-explored areas
- **Patent Clustering**: Group related technologies

### IP Professionals
- **Patent Analysis**: Semantic similarity analysis
- **Freedom to Operate**: Comprehensive prior art search
- **Portfolio Management**: Organize patent collections
- **Licensing Opportunities**: Find relevant patents

### Researchers
- **Literature Review**: Find related technical patents
- **Technology Trends**: Analyze patent evolution
- **Collaboration**: Identify potential partners
- **Innovation Metrics**: Measure technological progress

## Security

- **Data Privacy**: All processing happens locally
- **Access Control**: Role-based permissions (enterprise)
- **Audit Logging**: Complete search history tracking
- **Encryption**: Data at rest and in transit

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [USPTO](https://www.uspto.gov/) for patent data standards
- Open source community for inspiration
