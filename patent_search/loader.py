"""
Patent Search Tool - Data Loading
Load patent data from various sources and formats.
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles

from .models import Patent, validate_patent_data, create_sample_patent
from .processor import PatentDataProcessor

logger = logging.getLogger(__name__)


class PatentDataLoader:
    """Load patent data from various sources and formats"""
    
    def __init__(self):
        self.processor = PatentDataProcessor()
        self.supported_formats = ['csv', 'json', 'xml', 'xlsx']
    
    def load_from_csv(self, csv_path: str, encoding: str = 'utf-8') -> List[Patent]:
        """
        Load patents from CSV file
        
        Args:
            csv_path: Path to CSV file
            encoding: File encoding
            
        Returns:
            List of Patent objects
        """
        try:
            patents = []
            
            # Read CSV with pandas for better handling
            df = pd.read_csv(csv_path, encoding=encoding)
            
            logger.info(f"Loading {len(df)} patents from {csv_path}")
            
            for index, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    patent_data = row.to_dict()
                    
                    # Handle NaN values
                    for key, value in patent_data.items():
                        if pd.isna(value):
                            patent_data[key] = "" if isinstance(value, str) else []
                    
                    # Validate and clean data
                    validated_data = validate_patent_data(patent_data)
                    
                    # Create Patent object
                    patent = Patent.from_dict(validated_data)
                    
                    # Enhance with processor
                    patent = self.processor.enhance_patent_data(patent)
                    
                    patents.append(patent)
                    
                except Exception as e:
                    logger.warning(f"Error processing row {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(patents)} patents from CSV")
            return patents
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return []
    
    def load_from_json(self, json_path: str, encoding: str = 'utf-8') -> List[Patent]:
        """
        Load patents from JSON file
        
        Args:
            json_path: Path to JSON file
            encoding: File encoding
            
        Returns:
            List of Patent objects
        """
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            patents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                patent_list = data
            elif isinstance(data, dict):
                # Try common top-level keys
                patent_list = (data.get('patents') or 
                             data.get('data') or 
                             data.get('results') or 
                             [data])  # Single patent
            else:
                raise ValueError("Invalid JSON structure")
            
            logger.info(f"Loading {len(patent_list)} patents from {json_path}")
            
            for index, patent_data in enumerate(patent_list):
                try:
                    if not isinstance(patent_data, dict):
                        logger.warning(f"Invalid patent data at index {index}")
                        continue
                    
                    # Validate and clean data
                    validated_data = validate_patent_data(patent_data)
                    
                    # Create Patent object
                    patent = Patent.from_dict(validated_data)
                    
                    # Enhance with processor
                    patent = self.processor.enhance_patent_data(patent)
                    
                    patents.append(patent)
                    
                except Exception as e:
                    logger.warning(f"Error processing patent {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(patents)} patents from JSON")
            return patents
            
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
            return []
    
    def load_from_xml(self, xml_path: str, encoding: str = 'utf-8') -> List[Patent]:
        """
        Load patents from XML file
        
        Args:
            xml_path: Path to XML file
            encoding: File encoding
            
        Returns:
            List of Patent objects
        """
        try:
            with open(xml_path, 'r', encoding=encoding) as f:
                xml_content = f.read()
            
            patents = []
            
            # Try to parse as a collection of patents or single patent
            try:
                # Parse as single document
                root = ET.fromstring(xml_content)
                
                # Check if it's a collection
                patent_elements = root.findall('.//patent-document') or root.findall('.//patent')
                
                if not patent_elements:
                    # Try parsing the entire document as a single patent
                    patent = self.processor.parse_patent_xml(xml_content)
                    if patent:
                        patents.append(patent)
                else:
                    # Process each patent in the collection
                    for patent_elem in patent_elements:
                        patent_xml = ET.tostring(patent_elem, encoding='unicode')
                        patent = self.processor.parse_patent_xml(patent_xml)
                        if patent:
                            patents.append(patent)
                
            except ET.ParseError:
                # Try splitting by patent document boundaries
                patent_docs = xml_content.split('<?xml')
                for doc in patent_docs[1:]:  # Skip first empty split
                    doc = '<?xml' + doc
                    patent = self.processor.parse_patent_xml(doc)
                    if patent:
                        patents.append(patent)
            
            logger.info(f"Successfully loaded {len(patents)} patents from XML")
            return patents
            
        except Exception as e:
            logger.error(f"Error loading XML file {xml_path}: {e}")
            return []
    
    def load_from_xlsx(self, xlsx_path: str) -> List[Patent]:
        """
        Load patents from Excel file
        
        Args:
            xlsx_path: Path to Excel file
            
        Returns:
            List of Patent objects
        """
        try:
            # Read Excel file
            df = pd.read_excel(xlsx_path)
            
            patents = []
            
            logger.info(f"Loading {len(df)} patents from {xlsx_path}")
            
            for index, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    patent_data = row.to_dict()
                    
                    # Handle NaN values
                    for key, value in patent_data.items():
                        if pd.isna(value):
                            patent_data[key] = "" if key in ['patent_id', 'title', 'abstract', 'description'] else []
                    
                    # Validate and clean data
                    validated_data = validate_patent_data(patent_data)
                    
                    # Create Patent object
                    patent = Patent.from_dict(validated_data)
                    
                    # Enhance with processor
                    patent = self.processor.enhance_patent_data(patent)
                    
                    patents.append(patent)
                    
                except Exception as e:
                    logger.warning(f"Error processing row {index}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(patents)} patents from Excel")
            return patents
            
        except Exception as e:
            logger.error(f"Error loading Excel file {xlsx_path}: {e}")
            return []
    
    def load_from_directory(self, 
                          directory_path: str, 
                          file_pattern: str = "*",
                          max_workers: int = 4) -> List[Patent]:
        """
        Load patents from multiple files in a directory
        
        Args:
            directory_path: Path to directory containing patent files
            file_pattern: File pattern to match (e.g., "*.json", "*.xml")
            max_workers: Number of parallel workers
            
        Returns:
            List of Patent objects
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory_path}")
                return []
            
            # Find matching files
            files = list(directory.glob(file_pattern))
            
            if not files:
                logger.warning(f"No files found matching pattern: {file_pattern}")
                return []
            
            logger.info(f"Loading patents from {len(files)} files")
            
            all_patents = []
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit loading tasks
                future_to_file = {}
                for file_path in files:
                    future = executor.submit(self._load_single_file, str(file_path))
                    future_to_file[future] = file_path
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        patents = future.result()
                        all_patents.extend(patents)
                        logger.info(f"Loaded {len(patents)} patents from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
            
            logger.info(f"Successfully loaded {len(all_patents)} patents from directory")
            return all_patents
            
        except Exception as e:
            logger.error(f"Error loading from directory {directory_path}: {e}")
            return []
    
    def _load_single_file(self, file_path: str) -> List[Patent]:
        """
        Load patents from a single file (internal method)
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Patent objects
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return self.load_from_csv(file_path)
        elif file_ext == '.json':
            return self.load_from_json(file_path)
        elif file_ext in ['.xml', '.rdf']:
            return self.load_from_xml(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_from_xlsx(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return []
    
    async def load_from_csv_async(self, csv_path: str, encoding: str = 'utf-8') -> List[Patent]:
        """
        Asynchronously load patents from CSV file
        
        Args:
            csv_path: Path to CSV file
            encoding: File encoding
            
        Returns:
            List of Patent objects
        """
        try:
            async with aiofiles.open(csv_path, 'r', encoding=encoding) as f:
                content = await f.read()
            
            # Process in chunks for large files
            patents = []
            lines = content.strip().split('\n')
            
            if not lines:
                return patents
            
            # Parse header
            header = next(csv.reader([lines[0]]))
            
            # Process data rows
            for line_num, line in enumerate(lines[1:], 2):
                try:
                    row = next(csv.reader([line]))
                    if len(row) != len(header):
                        continue
                    
                    # Create patent data dictionary
                    patent_data = dict(zip(header, row))
                    
                    # Validate and create patent
                    validated_data = validate_patent_data(patent_data)
                    patent = Patent.from_dict(validated_data)
                    patent = self.processor.enhance_patent_data(patent)
                    
                    patents.append(patent)
                    
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
            
            logger.info(f"Async loaded {len(patents)} patents from CSV")
            return patents
            
        except Exception as e:
            logger.error(f"Error async loading CSV file {csv_path}: {e}")
            return []
    
    def stream_from_file(self, file_path: str, chunk_size: int = 1000) -> Iterator[List[Patent]]:
        """
        Stream patents from file in chunks (memory efficient)
        
        Args:
            file_path: Path to file
            chunk_size: Number of patents per chunk
            
        Yields:
            Chunks of Patent objects
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            yield from self._stream_from_csv(file_path, chunk_size)
        elif file_ext == '.json':
            # For JSON, load all then yield in chunks
            patents = self.load_from_json(file_path)
            for i in range(0, len(patents), chunk_size):
                yield patents[i:i + chunk_size]
        else:
            # For other formats, load all then yield in chunks
            patents = self._load_single_file(file_path)
            for i in range(0, len(patents), chunk_size):
                yield patents[i:i + chunk_size]
    
    def _stream_from_csv(self, csv_path: str, chunk_size: int) -> Iterator[List[Patent]]:
        """
        Stream patents from CSV file in chunks
        
        Args:
            csv_path: Path to CSV file
            chunk_size: Number of patents per chunk
            
        Yields:
            Chunks of Patent objects
        """
        try:
            chunk_patents = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader):
                    try:
                        # Validate and create patent
                        validated_data = validate_patent_data(row)
                        patent = Patent.from_dict(validated_data)
                        patent = self.processor.enhance_patent_data(patent)
                        
                        chunk_patents.append(patent)
                        
                        # Yield chunk when it reaches the desired size
                        if len(chunk_patents) >= chunk_size:
                            yield chunk_patents
                            chunk_patents = []
                            
                    except Exception as e:
                        logger.warning(f"Error processing row {row_num + 1}: {e}")
                        continue
                
                # Yield remaining patents
                if chunk_patents:
                    yield chunk_patents
                    
        except Exception as e:
            logger.error(f"Error streaming from CSV {csv_path}: {e}")
    
    def create_sample_patents(self, count: int = 50) -> List[Patent]:
        """
        Create sample patents for testing and demonstration
        
        Args:
            count: Number of sample patents to create
            
        Returns:
            List of sample Patent objects
        """
        try:
            logger.info(f"Creating {count} sample patents")
            
            patents = []
            for i in range(count):
                patent = create_sample_patent(i)
                patents.append(patent)
            
            logger.info(f"Successfully created {len(patents)} sample patents")
            return patents
            
        except Exception as e:
            logger.error(f"Error creating sample patents: {e}")
            return []
    
    def save_to_csv(self, patents: List[Patent], csv_path: str) -> bool:
        """
        Save patents to CSV file
        
        Args:
            patents: List of Patent objects
            csv_path: Output CSV file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not patents:
                logger.warning("No patents to save")
                return False
            
            # Convert patents to dictionaries
            patent_dicts = [patent.to_dict() for patent in patents]
            
            # Create DataFrame
            df = pd.DataFrame(patent_dicts)
            
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"Saved {len(patents)} patents to {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving patents to CSV {csv_path}: {e}")
            return False
    
    def save_to_json(self, patents: List[Patent], json_path: str) -> bool:
        """
        Save patents to JSON file
        
        Args:
            patents: List of Patent objects
            json_path: Output JSON file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not patents:
                logger.warning("No patents to save")
                return False
            
            # Convert patents to dictionaries
            patent_dicts = [patent.to_dict() for patent in patents]
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(patent_dicts, f, indent=2, default=str)
            
            logger.info(f"Saved {len(patents)} patents to {json_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving patents to JSON {json_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a patent data file
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'error': 'File does not exist'}
            
            info = {
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower(),
                'supported': file_path.suffix.lower().lstrip('.') in self.supported_formats
            }
            
            # Try to get record count for different formats
            try:
                if info['file_extension'] == '.csv':
                    df = pd.read_csv(file_path, nrows=0)  # Just header
                    info['columns'] = list(df.columns)
                    
                    # Count rows (approximate for large files)
                    with open(file_path, 'r') as f:
                        row_count = sum(1 for _ in f) - 1  # Exclude header
                    info['estimated_records'] = row_count
                    
                elif info['file_extension'] == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        info['estimated_records'] = len(data)
                    elif isinstance(data, dict):
                        # Check common structure keys
                        records = (data.get('patents') or 
                                 data.get('data') or 
                                 data.get('results'))
                        if isinstance(records, list):
                            info['estimated_records'] = len(records)
                        else:
                            info['estimated_records'] = 1
                    
                elif info['file_extension'] in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, nrows=0)
                    info['columns'] = list(df.columns)
                    
                    # For Excel, we need to read to count (can be expensive)
                    df_full = pd.read_excel(file_path)
                    info['estimated_records'] = len(df_full)
                    
            except Exception as e:
                info['analysis_error'] = str(e)
            
            return info
            
        except Exception as e:
            return {'error': f'Error analyzing file: {e}'}
    
    def validate_file_format(self, file_path: str) -> Dict[str, Any]:
        """
        Validate patent data file format and structure
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation results
        """
        try:
            file_path = Path(file_path)
            
            validation = {
                'is_valid': False,
                'format': file_path.suffix.lower().lstrip('.'),
                'errors': [],
                'warnings': [],
                'suggestions': []
            }
            
            if not file_path.exists():
                validation['errors'].append('File does not exist')
                return validation
            
            # Format-specific validation
            if validation['format'] == 'csv':
                validation.update(self._validate_csv_format(file_path))
            elif validation['format'] == 'json':
                validation.update(self._validate_json_format(file_path))
            elif validation['format'] in ['xlsx', 'xls']:
                validation.update(self._validate_excel_format(file_path))
            elif validation['format'] in ['xml', 'rdf']:
                validation.update(self._validate_xml_format(file_path))
            else:
                validation['errors'].append(f'Unsupported format: {validation["format"]}')
            
            validation['is_valid'] = len(validation['errors']) == 0
            
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'format': 'unknown',
                'errors': [f'Validation error: {e}'],
                'warnings': [],
                'suggestions': []
            }
    
    def _validate_csv_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate CSV file format"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        try:
            # Check if file can be read
            df = pd.read_csv(file_path, nrows=5)
            
            # Check for required columns
            required_cols = ['patent_id', 'title']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                result['errors'].append(f'Missing required columns: {missing_required}')
            
            # Check for recommended columns
            recommended_cols = ['abstract', 'inventors', 'assignees', 'filing_date']
            missing_recommended = [col for col in recommended_cols if col not in df.columns]
            
            if missing_recommended:
                result['warnings'].append(f'Missing recommended columns: {missing_recommended}')
                result['suggestions'].append('Consider adding columns for better search results')
            
            # Check data quality
            if 'patent_id' in df.columns:
                if df['patent_id'].isnull().any():
                    result['warnings'].append('Some patent IDs are missing')
                
                if df['patent_id'].duplicated().any():
                    result['warnings'].append('Duplicate patent IDs found')
            
            return result
            
        except Exception as e:
            return {'errors': [f'CSV parsing error: {e}'], 'warnings': [], 'suggestions': []}
    
    def _validate_json_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate JSON file format"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check structure
            if isinstance(data, list):
                if not data:
                    result['errors'].append('Empty patent list')
                    return result
                
                sample = data[0]
            elif isinstance(data, dict):
                # Look for patent arrays
                patents = (data.get('patents') or data.get('data') or 
                          data.get('results') or [data])
                
                if not patents:
                    result['errors'].append('No patent data found')
                    return result
                
                sample = patents[0] if isinstance(patents, list) else patents
            else:
                result['errors'].append('Invalid JSON structure')
                return result
            
            # Validate sample patent
            if not isinstance(sample, dict):
                result['errors'].append('Patents must be objects/dictionaries')
                return result
            
            # Check required fields
            if 'patent_id' not in sample:
                result['errors'].append('Missing patent_id field')
            
            if 'title' not in sample:
                result['errors'].append('Missing title field')
            
            return result
            
        except json.JSONDecodeError as e:
            return {'errors': [f'Invalid JSON: {e}'], 'warnings': [], 'suggestions': []}
        except Exception as e:
            return {'errors': [f'JSON validation error: {e}'], 'warnings': [], 'suggestions': []}
    
    def _validate_excel_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate Excel file format"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        try:
            # Similar to CSV validation
            df = pd.read_excel(file_path, nrows=5)
            
            required_cols = ['patent_id', 'title']
            missing_required = [col for col in required_cols if col not in df.columns]
            
            if missing_required:
                result['errors'].append(f'Missing required columns: {missing_required}')
            
            return result
            
        except Exception as e:
            return {'errors': [f'Excel parsing error: {e}'], 'warnings': [], 'suggestions': []}
    
    def _validate_xml_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate XML file format"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to parse XML
            ET.fromstring(content)
            
            # Check for patent elements
            if '<patent' not in content.lower():
                result['warnings'].append('No patent elements found in XML')
            
            return result
            
        except ET.ParseError as e:
            return {'errors': [f'Invalid XML: {e}'], 'warnings': [], 'suggestions': []}
        except Exception as e:
            return {'errors': [f'XML validation error: {e}'], 'warnings': [], 'suggestions': []}
