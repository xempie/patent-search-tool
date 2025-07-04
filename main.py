#!/usr/bin/env python3
"""
Patent Search Tool - Main Entry Point
Command-line interface for the patent search system.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import click
from dotenv import load_dotenv

# Add the package to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patent_search.engine import PatentSearchEngine
from patent_search.loader import PatentDataLoader
from patent_search.models import Patent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatentSearchCLI:
    """Command-line interface for patent search operations"""
    
    def __init__(self):
        self.search_engine = None
        self.data_loader = PatentDataLoader()
    
    def initialize_engine(self, 
                         qdrant_host: str = None,
                         qdrant_port: int = None,
                         collection_name: str = None,
                         model_name: str = None):
        """Initialize the search engine with custom parameters"""
        
        # Use environment variables or defaults
        host = qdrant_host or os.getenv('QDRANT_HOST', 'localhost')
        port = int(qdrant_port or os.getenv('QDRANT_PORT', 6333))
        collection = collection_name or os.getenv('COLLECTION_NAME', 'patents')
        model = model_name or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        
        try:
            self.search_engine = PatentSearchEngine(
                qdrant_host=host,
                qdrant_port=port,
                collection_name=collection,
                model_name=model
            )
            logger.info(f"Initialized search engine: {host}:{port}/{collection}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            return False
    
    def load_data(self, file_path: str, file_format: str = 'auto') -> int:
        """Load patent data from file"""
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 0
        
        # Auto-detect format if not specified
        if file_format == 'auto':
            ext = Path(file_path).suffix.lower()
            if ext == '.csv':
                file_format = 'csv'
            elif ext == '.json':
                file_format = 'json'
            else:
                logger.error(f"Unsupported file format: {ext}")
                return 0
        
        # Load patents
        try:
            if file_format == 'csv':
                patents = self.data_loader.load_from_csv(file_path)
            elif file_format == 'json':
                patents = self.data_loader.load_from_json(file_path)
            else:
                logger.error(f"Unsupported format: {file_format}")
                return 0
            
            logger.info(f"Loaded {len(patents)} patents from {file_path}")
            return len(patents)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 0
    
    def index_data(self, file_path: str, batch_size: int = 100) -> int:
        """Load and index patent data"""
        
        if not self.search_engine:
            if not self.initialize_engine():
                return 0
        
        # Load patents
        patents_count = self.load_data(file_path)
        if patents_count == 0:
            return 0
        
        # Get patents for indexing
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            patents = self.data_loader.load_from_csv(file_path)
        elif ext == '.json':
            patents = self.data_loader.load_from_json(file_path)
        else:
            return 0
        
        # Index patents
        try:
            indexed_count = self.search_engine.index_patents_batch(patents, batch_size)
            logger.info(f"Successfully indexed {indexed_count} patents")
            return indexed_count
        except Exception as e:
            logger.error(f"Error indexing patents: {e}")
            return 0
    
    def search(self, 
               query: str,
               limit: int = 10,
               technology_field: str = None,
               assignees: List[str] = None,
               date_range: tuple = None,
               min_score: float = 0.7,
               output_format: str = 'text') -> List[Dict[str, Any]]:
        """Perform patent search"""
        
        if not self.search_engine:
            if not self.initialize_engine():
                return []
        
        try:
            results = self.search_engine.search_patents(
                query=query,
                limit=limit,
                technology_field=technology_field,
                assignees=assignees,
                date_range=date_range,
                min_score=min_score
            )
            
            self._display_results(results, output_format)
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def semantic_search(self, 
                       query: str, 
                       limit: int = 10,
                       output_format: str = 'text') -> List[Dict[str, Any]]:
        """Perform semantic search"""
        
        if not self.search_engine:
            if not self.initialize_engine():
                return []
        
        try:
            results = self.search_engine.semantic_search(query, limit)
            self._display_results(results, output_format)
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def find_similar(self, 
                    patent_id: str, 
                    limit: int = 5,
                    output_format: str = 'text') -> List[Dict[str, Any]]:
        """Find similar patents"""
        
        if not self.search_engine:
            if not self.initialize_engine():
                return []
        
        try:
            results = self.search_engine.get_similar_patents(patent_id, limit)
            self._display_results(results, output_format)
            return results
        except Exception as e:
            logger.error(f"Similar search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        
        if not self.search_engine:
            if not self.initialize_engine():
                return {}
        
        try:
            stats = self.search_engine.get_collection_stats()
            self._display_stats(stats)
            return stats
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
    
    def create_sample_data(self, count: int = 50, output_file: str = 'sample_patents.json'):
        """Create sample patent data for testing"""
        
        try:
            patents = self.data_loader.create_sample_patents(count)
            
            # Convert to serializable format
            patents_data = [patent.__dict__ for patent in patents]
            
            with open(output_file, 'w') as f:
                json.dump(patents_data, f, indent=2)
            
            logger.info(f"Created {count} sample patents in {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return False
    
    def _display_results(self, results: List[Dict[str, Any]], output_format: str):
        """Display search results"""
        
        if not results:
            click.echo("No results found.")
            return
        
        if output_format == 'json':
            click.echo(json.dumps(results, indent=2, default=str))
            return
        
        # Text format
        click.echo(f"\n Found {len(results)} patents:\n")
        
        for i, patent in enumerate(results, 1):
            score = patent.get('similarity_score', 0)
            click.echo(f"{i}. {patent.get('title', 'Unknown Title')}")
            click.echo(f"   Patent ID: {patent.get('patent_id', 'N/A')}")
            click.echo(f"   Score: {score:.3f}")
            click.echo(f"   Technology: {patent.get('technology_field', 'Unknown')}")
            click.echo(f"   Assignees: {', '.join(patent.get('assignees', []))}")
            click.echo(f"   Filing Date: {patent.get('filing_date', 'N/A')}")
            
            # Show abstract preview
            abstract = patent.get('abstract', '')
            if abstract:
                preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
                click.echo(f"   Abstract: {preview}")
            
            click.echo()
    
    def _display_stats(self, stats: Dict[str, Any]):
        """Display collection statistics"""
        
        click.echo("\n Collection Statistics:")
        click.echo(f"Total Patents: {stats.get('total_patents', 0)}")
        click.echo(f"Vector Size: {stats.get('vector_size', 0)}")
        click.echo(f"Distance Metric: {stats.get('distance_metric', 'Unknown')}")
        
        tech_fields = stats.get('technology_fields', {})
        if tech_fields:
            click.echo("\n Technology Fields:")
            for field, count in sorted(tech_fields.items(), key=lambda x: x[1], reverse=True):
                click.echo(f"  {field}: {count}")
        
        click.echo()

# CLI Commands using Click
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Patent Search Tool - Semantic patent search using Qdrant"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['cli'] = PatentSearchCLI()

@cli.command()
@click.argument('file_path')
@click.option('--batch-size', '-b', default=100, help='Batch size for indexing')
@click.option('--host', default='localhost', help='Qdrant host')
@click.option('--port', default=6333, help='Qdrant port')
@click.pass_context
def index(ctx, file_path, batch_size, host, port):
    """Index patent data from file (CSV or JSON)"""
    
    cli_obj = ctx.obj['cli']
    
    # Initialize with custom parameters
    if not cli_obj.initialize_engine(qdrant_host=host, qdrant_port=port):
        ctx.exit(1)
    
    indexed_count = cli_obj.index_data(file_path, batch_size)
    
    if indexed_count > 0:
        click.echo(f"Successfully indexed {indexed_count} patents")
    else:
        click.echo("Failed to index patents")
        ctx.exit(1)

@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of results')
@click.option('--tech-field', help='Filter by technology field')
@click.option('--assignee', multiple=True, help='Filter by assignee (can specify multiple)')
@click.option('--min-score', default=0.7, help='Minimum similarity score')
@click.option('--output', '-o', default='text', type=click.Choice(['text', 'json']), help='Output format')
@click.pass_context
def search(ctx, query, limit, tech_field, assignee, min_score, output):
    """Search patents with filters"""
    
    cli_obj = ctx.obj['cli']
    
    assignees = list(assignee) if assignee else None
    
    results = cli_obj.search(
        query=query,
        limit=limit,
        technology_field=tech_field,
        assignees=assignees,
        min_score=min_score,
        output_format=output
    )
    
    if not results:
        ctx.exit(1)

@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of results')
@click.option('--output', '-o', default='text', type=click.Choice(['text', 'json']), help='Output format')
@click.pass_context
def semantic(ctx, query, limit, output):
    """Perform semantic search on patents"""
    
    cli_obj = ctx.obj['cli']
    
    results = cli_obj.semantic_search(query, limit, output)
    
    if not results:
        ctx.exit(1)

@cli.command()
@click.argument('patent_id')
@click.option('--limit', '-l', default=5, help='Number of similar patents')
@click.option('--output', '-o', default='text', type=click.Choice(['text', 'json']), help='Output format')
@click.pass_context
def similar(ctx, patent_id, limit, output):
    """Find patents similar to a given patent ID"""
    
    cli_obj = ctx.obj['cli']
    
    results = cli_obj.find_similar(patent_id, limit, output)
    
    if not results:
        ctx.exit(1)

@cli.command()
@click.pass_context
def stats(ctx):
    """Show collection statistics"""
    
    cli_obj = ctx.obj['cli']
    
    stats = cli_obj.get_stats()
    
    if not stats:
        ctx.exit(1)

@cli.command()
@click.option('--count', '-c', default=50, help='Number of sample patents to create')
@click.option('--output', '-o', default='sample_patents.json', help='Output file')
@click.pass_context
def sample(ctx, count, output):
    """Create sample patent data for testing"""
    
    cli_obj = ctx.obj['cli']
    
    if cli_obj.create_sample_data(count, output):
        click.echo(f"Created {count} sample patents in {output}")
    else:
        click.echo("Failed to create sample data")
        ctx.exit(1)

@cli.command()
@click.pass_context
def demo(ctx):
    """Run a complete demo with sample data"""
    
    cli_obj = ctx.obj['cli']
    
    click.echo("Starting Patent Search Demo...")
    
    # Create sample data
    click.echo("Creating sample patents...")
    if not cli_obj.create_sample_data(50, 'demo_patents.json'):
        click.echo("Failed to create sample data")
        ctx.exit(1)
    
    # Initialize search engine
    click.echo("🔧 Initializing search engine...")
    if not cli_obj.initialize_engine():
        click.echo("Failed to initialize search engine")
        ctx.exit(1)
    
    # Index sample data
    click.echo("Indexing patents...")
    indexed_count = cli_obj.index_data('demo_patents.json')
    if indexed_count == 0:
        click.echo("Failed to index patents")
        ctx.exit(1)
    
    click.echo(f"Indexed {indexed_count} patents")
    
    # Show stats
    click.echo("Collection Statistics:")
    cli_obj.get_stats()
    
    # Perform searches
    click.echo(" Example Searches:")
    
    click.echo("\n1. Machine Learning Patents:")
    cli_obj.semantic_search("machine learning algorithms", limit=3)
    
    click.echo("\n2. Blockchain Technology:")
    cli_obj.semantic_search("blockchain distributed ledger", limit=3)
    
    click.echo("\n3. Quantum Computing:")
    cli_obj.semantic_search("quantum computing algorithms", limit=3)
    
    click.echo("\n Demo completed! Use 'python main.py --help' for more commands.")

if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
