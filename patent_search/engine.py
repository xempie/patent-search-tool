"""
Patent Search Tool - Search Engine
Core search engine using Qdrant vector database for semantic patent search.
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range, SearchRequest,
    UpdateCollection, OptimizersConfig, HnswConfig,
    PayloadSchemaType, CreatePayloadIndex
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .models import Patent, SearchResult, SearchFilter, CollectionStats

logger = logging.getLogger(__name__)


class PatentSearchEngine:
    """Main patent search engine using Qdrant vector database"""
    
    def __init__(self, 
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_api_key: Optional[str] = None,
                 collection_name: str = "patents",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        """
        Initialize the patent search engine
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            qdrant_api_key: API key for Qdrant Cloud (optional)
            collection_name: Name of the collection to use
            model_name: Sentence transformer model name
            device: Device for model inference ('cpu' or 'cuda')
        """
        
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize Qdrant client
        if qdrant_api_key:
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key
            )
        else:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {model_name} (dim: {self.vector_size})")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
        
        # Initialize collection
        self._initialize_collection()
        
        logger.info(f"PatentSearchEngine initialized: {qdrant_host}:{qdrant_port}/{collection_name}")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection with optimized settings"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new collection: {self.collection_name}")
                
                # Create collection with optimized settings
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfig(
                            m=32,  # Number of bi-directional links
                            ef_construct=200,  # Size of dynamic candidate list
                            full_scan_threshold=10000,  # Threshold for full scan
                            max_indexing_threads=4  # Parallel indexing
                        )
                    ),
                    optimizers_config=OptimizersConfig(
                        default_segment_number=8,
                        max_segment_size=200000,
                        memmap_threshold=200000,
                        indexing_threshold=50000,
                        flush_interval_sec=30,
                        max_optimization_threads=4
                    )
                )
                
                # Create payload indexes for efficient filtering
                self._create_payload_indexes()
                
                logger.info(f"Created collection '{self.collection_name}' with indexes")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def _create_payload_indexes(self):
        """Create payload indexes for efficient filtering"""
        try:
            # Technology field index
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="technology_field",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Assignees index
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="assignees",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Filing date index
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filing_date",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Country index
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="country",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Status index
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="status",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info("Created payload indexes for efficient filtering")
            
        except Exception as e:
            logger.warning(f"Some payload indexes may already exist: {e}")
    
    def _create_combined_text(self, patent: Patent) -> str:
        """
        Create combined text for vectorization
        
        Args:
            patent: Patent object
            
        Returns:
            Combined text for embedding
        """
        components = [
            f"Title: {patent.title}",
            f"Abstract: {patent.abstract}",
            f"Technology Field: {patent.technology_field}",
            f"Claims: {' '.join(patent.claims[:3])}",  # First 3 claims
            f"Description: {patent.description[:1000]}"  # First 1000 chars
        ]
        
        # Add inventor and assignee information
        if patent.inventors:
            components.append(f"Inventors: {', '.join(patent.inventors[:3])}")
        
        if patent.assignees:
            components.append(f"Assignees: {', '.join(patent.assignees[:2])}")
        
        return " ".join(filter(None, components))
    
    def _generate_vector(self, text: str) -> List[float]:
        """
        Generate vector embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as list of floats
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating vector: {e}")
            return [0.0] * self.vector_size
    
    def _create_point_id(self, patent: Patent) -> str:
        """
        Create unique point ID for patent
        
        Args:
            patent: Patent object
            
        Returns:
            Unique point ID
        """
        # Create hash from patent ID and title for uniqueness
        content = f"{patent.patent_id}_{patent.title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def index_patent(self, patent: Patent) -> bool:
        """
        Index a single patent
        
        Args:
            patent: Patent object to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create combined text for vectorization
            combined_text = self._create_combined_text(patent)
            
            # Generate vector
            vector = self._generate_vector(combined_text)
            
            # Create unique ID
            point_id = self._create_point_id(patent)
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=patent.to_dict()
            )
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Indexed patent: {patent.patent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing patent {patent.patent_id}: {e}")
            return False
    
    def index_patents_batch(self, patents: List[Patent], batch_size: int = 100) -> int:
        """
        Index multiple patents in batches
        
        Args:
            patents: List of Patent objects
            batch_size: Size of each batch
            
        Returns:
            Number of successfully indexed patents
        """
        indexed_count = 0
        total_patents = len(patents)
        
        logger.info(f"Starting batch indexing of {total_patents} patents")
        
        for i in range(0, total_patents, batch_size):
            batch = patents[i:i + batch_size]
            points = []
            
            for patent in batch:
                try:
                    combined_text = self._create_combined_text(patent)
                    vector = self._generate_vector(combined_text)
                    point_id = self._create_point_id(patent)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=patent.to_dict()
                    )
                    
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Error processing patent {patent.patent_id}: {e}")
                    continue
            
            if points:
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    indexed_count += len(points)
                    logger.info(f"Indexed batch {i//batch_size + 1}/{(total_patents-1)//batch_size + 1}: {len(points)} patents")
                    
                except Exception as e:
                    logger.error(f"Error indexing batch: {e}")
        
        logger.info(f"Batch indexing completed: {indexed_count}/{total_patents} patents indexed")
        return indexed_count
    
    def search_patents(self, 
                      query: str, 
                      limit: int = 10,
                      technology_field: Optional[str] = None,
                      assignees: Optional[List[str]] = None,
                      inventors: Optional[List[str]] = None,
                      countries: Optional[List[str]] = None,
                      date_range: Optional[Tuple[str, str]] = None,
                      classification_codes: Optional[List[str]] = None,
                      status: Optional[List[str]] = None,
                      min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search patents with filters
        
        Args:
            query: Search query text
            limit: Maximum number of results
            technology_field: Filter by technology field
            assignees: Filter by assignees
            inventors: Filter by inventors
            countries: Filter by countries
            date_range: Filter by filing date range (start_date, end_date)
            classification_codes: Filter by classification codes
            status: Filter by patent status
            min_score: Minimum similarity score
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate query vector
            query_vector = self._generate_vector(query)
            
            # Build filters
            filters = []
            
            if technology_field:
                filters.append(
                    FieldCondition(
                        key="technology_field",
                        match=MatchValue(value=technology_field)
                    )
                )
            
            if assignees:
                assignee_filters = []
                for assignee in assignees:
                    assignee_filters.append(
                        FieldCondition(
                            key="assignees",
                            match=MatchValue(value=assignee)
                        )
                    )
                filters.extend(assignee_filters)
            
            if inventors:
                inventor_filters = []
                for inventor in inventors:
                    inventor_filters.append(
                        FieldCondition(
                            key="inventors",
                            match=MatchValue(value=inventor)
                        )
                    )
                filters.extend(inventor_filters)
            
            if countries:
                country_filters = []
                for country in countries:
                    country_filters.append(
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country)
                        )
                    )
                filters.extend(country_filters)
            
            if status:
                status_filters = []
                for stat in status:
                    status_filters.append(
                        FieldCondition(
                            key="status",
                            match=MatchValue(value=stat)
                        )
                    )
                filters.extend(status_filters)
            
            if date_range:
                start_date, end_date = date_range
                if start_date or end_date:
                    range_filter = {}
                    if start_date:
                        range_filter['gte'] = start_date
                    if end_date:
                        range_filter['lte'] = end_date
                    
                    filters.append(
                        FieldCondition(
                            key="filing_date",
                            range=Range(**range_filter)
                        )
                    )
            
            if classification_codes:
                code_filters = []
                for code in classification_codes:
                    code_filters.append(
                        FieldCondition(
                            key="classification_codes",
                            match=MatchValue(value=code)
                        )
                    )
                filters.extend(code_filters)
            
            # Combine filters
            search_filter = Filter(must=filters) if filters else None
            
            # Perform search
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score
            )
            
            # Format results
            results = []
            for rank, result in enumerate(search_results, 1):
                patent_data = result.payload.copy()
                patent_data['similarity_score'] = result.score
                patent_data['search_query'] = query
                patent_data['rank'] = rank
                results.append(patent_data)
            
            logger.info(f"Search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching patents: {e}")
            return []
    
    def semantic_search(self, query: str, limit: int = 10, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform semantic search on patents
        
        Args:
            query: Search query text
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        return self.search_patents(query, limit=limit, min_score=min_score)
    
    def get_similar_patents(self, patent_id: str, limit: int = 5, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find patents similar to a given patent
        
        Args:
            patent_id: Patent ID to find similar patents for
            limit: Maximum number of similar patents
            min_score: Minimum similarity score
            
        Returns:
            List of similar patents
        """
        try:
            # First, retrieve the patent
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="patent_id",
                            match=MatchValue(value=patent_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            if not search_results[0]:
                logger.warning(f"Patent not found: {patent_id}")
                return []
            
            # Get the patent's vector
            patent_point = search_results[0][0]
            patent_vector = patent_point.vector
            
            # Search for similar patents
            similar_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=patent_vector,
                limit=limit + 1,  # +1 to exclude the original patent
                score_threshold=min_score
            )
            
            # Filter out the original patent and format results
            results = []
            for rank, result in enumerate(similar_results, 1):
                if result.payload.get('patent_id') != patent_id:
                    patent_data = result.payload.copy()
                    patent_data['similarity_score'] = result.score
                    patent_data['reference_patent_id'] = patent_id
                    patent_data['rank'] = rank
                    results.append(patent_data)
            
            # Limit to requested number
            results = results[:limit]
            
            logger.info(f"Found {len(results)} similar patents for {patent_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar patents for {patent_id}: {e}")
            return []
    
    def get_patent_by_id(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a patent by its ID
        
        Args:
            patent_id: Patent ID to retrieve
            
        Returns:
            Patent data or None if not found
        """
        try:
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="patent_id",
                            match=MatchValue(value=patent_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_results[0]:
                return search_results[0][0].payload
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving patent {patent_id}: {e}")
            return None
    
    def delete_patent(self, patent_id: str) -> bool:
        """
        Delete a patent from the index
        
        Args:
            patent_id: Patent ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the point ID
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="patent_id",
                            match=MatchValue(value=patent_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if not search_results[0]:
                logger.warning(f"Patent not found for deletion: {patent_id}")
                return False
            
            point_id = search_results[0][0].id
            
            # Delete the point
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            
            logger.info(f"Deleted patent: {patent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting patent {patent_id}: {e}")
            return False
    
    def update_patent(self, patent: Patent) -> bool:
        """
        Update an existing patent in the index
        
        Args:
            patent: Updated Patent object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing and re-index
            self.delete_patent(patent.patent_id)
            return self.index_patent(patent)
            
        except Exception as e:
            logger.error(f"Error updating patent {patent.patent_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get technology field distribution
            tech_fields = {}
            assignees = {}
            countries = {}
            filing_years = {}
            
            # Scroll through all points to gather statistics
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on collection size
                with_payload=["technology_field", "assignees", "country", "filing_date"]
            )
            
            for point in scroll_result[0]:
                payload = point.payload
                
                # Technology fields
                field = payload.get('technology_field', 'Unknown')
                tech_fields[field] = tech_fields.get(field, 0) + 1
                
                # Countries
                country = payload.get('country', 'Unknown')
                countries[country] = countries.get(country, 0) + 1
                
                # Assignees
                assignee_list = payload.get('assignees', [])
                if isinstance(assignee_list, list):
                    for assignee in assignee_list[:1]:  # Count primary assignee
                        assignees[assignee] = assignees.get(assignee, 0) + 1
                
                # Filing years
                filing_date = payload.get('filing_date', '')
                if filing_date and len(filing_date) >= 4:
                    year = filing_date[:4]
                    filing_years[year] = filing_years.get(year, 0) + 1
            
            stats = {
                'total_patents': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance.value,
                'technology_fields': dict(sorted(tech_fields.items(), key=lambda x: x[1], reverse=True)[:10]),
                'top_assignees': dict(sorted(assignees.items(), key=lambda x: x[1], reverse=True)[:10]),
                'countries': dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10]),
                'filing_years': dict(sorted(filing_years.items(), key=lambda x: x[0], reverse=True)[:10]),
                'indexed_fields': ['technology_field', 'assignees', 'filing_date', 'country', 'status']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all patents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.qdrant_client.delete_collection(self.collection_name)
            
            # Recreate it
            self._initialize_collection()
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def export_patents(self, output_format: str = 'json', limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Export all patents from the collection
        
        Args:
            output_format: Format for export ('json', 'csv')
            limit: Maximum number of patents to export
            
        Returns:
            List of patent dictionaries
        """
        try:
            patents = []
            offset = None
            batch_size = 1000
            
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    patents.append(point.payload)
                    
                    if limit and len(patents) >= limit:
                        return patents[:limit]
                
                offset = next_offset
                if not next_offset:
                    break
            
            logger.info(f"Exported {len(patents)} patents")
            return patents
            
        except Exception as e:
            logger.error(f"Error exporting patents: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the search engine
        
        Returns:
            Health status information
        """
        try:
            # Check Qdrant connection
            collections = self.qdrant_client.get_collections()
            
            # Check if our collection exists
            collection_exists = self.collection_name in [col.name for col in collections.collections]
            
            # Get collection info if it exists
            collection_info = None
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            health = {
                'status': 'healthy',
                'qdrant_connection': True,
                'collection_exists': collection_exists,
                'collection_name': self.collection_name,
                'model_name': self.model_name,
                'vector_size': self.vector_size,
                'total_points': collection_info.points_count if collection_info else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            return health
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'qdrant_connection': False,
                'collection_exists': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_embedding_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model
        
        Returns:
            Model information
        """
        return {
            'model_name': self.model_name,
            'vector_size': self.vector_size,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'device': str(self.model.device),
        }
