"""
Patent Search Tool - Data Models
Data structures and models for patent information.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class Patent:
    """Patent data structure with comprehensive metadata"""
    
    # Core identification
    patent_id: str
    patent_number: str
    title: str
    
    # Content
    abstract: str
    description: str
    claims: List[str] = field(default_factory=list)
    
    # People and organizations
    inventors: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    
    # Dates
    filing_date: str = ""
    publication_date: str = ""
    priority_date: Optional[str] = None
    
    # Classification and categorization
    classification_codes: List[str] = field(default_factory=list)
    technology_field: str = "Unknown"
    
    # Geographical and legal
    country: str = "US"
    status: str = "Published"
    
    # Relationships
    family_id: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)
    
    # Additional metadata
    application_number: Optional[str] = None
    priority_number: Optional[str] = None
    legal_status: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and cleanup"""
        
        # Ensure lists are not None
        if self.claims is None:
            self.claims = []
        if self.inventors is None:
            self.inventors = []
        if self.assignees is None:
            self.assignees = []
        if self.classification_codes is None:
            self.classification_codes = []
        if self.citations is None:
            self.citations = []
        if self.cited_by is None:
            self.cited_by = []
        
        # Clean string fields
        self.title = self.title.strip()
        self.abstract = self.abstract.strip()
        self.description = self.description.strip()
        
        # Validate patent_id
        if not self.patent_id:
            raise ValueError("Patent ID is required")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patent to dictionary for serialization"""
        return {
            'patent_id': self.patent_id,
            'patent_number': self.patent_number,
            'title': self.title,
            'abstract': self.abstract,
            'description': self.description,
            'claims': self.claims,
            'inventors': self.inventors,
            'assignees': self.assignees,
            'filing_date': self.filing_date,
            'publication_date': self.publication_date,
            'priority_date': self.priority_date,
            'classification_codes': self.classification_codes,
            'technology_field': self.technology_field,
            'country': self.country,
            'status': self.status,
            'family_id': self.family_id,
            'citations': self.citations,
            'cited_by': self.cited_by,
            'application_number': self.application_number,
            'priority_number': self.priority_number,
            'legal_status': self.legal_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Patent':
        """Create Patent from dictionary"""
        
        # Handle missing fields with defaults
        defaults = {
            'claims': [],
            'inventors': [],
            'assignees': [],
            'filing_date': '',
            'publication_date': '',
            'priority_date': None,
            'classification_codes': [],
            'technology_field': 'Unknown',
            'country': 'US',
            'status': 'Published',
            'family_id': None,
            'citations': [],
            'cited_by': [],
            'application_number': None,
            'priority_number': None,
            'legal_status': None
        }
        
        # Merge with defaults
        patent_data = {**defaults, **data}
        
        return cls(**patent_data)
    
    def to_json(self) -> str:
        """Convert patent to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Patent':
        """Create Patent from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_searchable_text(self) -> str:
        """Get combined text for search indexing"""
        components = [
            f"Title: {self.title}",
            f"Abstract: {self.abstract}",
            f"Technology: {self.technology_field}",
            f"Claims: {' '.join(self.claims[:3])}",  # First 3 claims
            f"Description: {self.description[:1000]}"  # First 1000 chars
        ]
        
        return " ".join(filter(None, components))
    
    def get_metadata_text(self) -> str:
        """Get metadata as searchable text"""
        metadata_parts = [
            f"Inventors: {', '.join(self.inventors)}",
            f"Assignees: {', '.join(self.assignees)}",
            f"Classification: {', '.join(self.classification_codes)}",
            f"Country: {self.country}",
            f"Status: {self.status}"
        ]
        
        return " ".join(filter(None, metadata_parts))
    
    def has_assignee(self, assignee: str) -> bool:
        """Check if patent has specific assignee"""
        assignee_lower = assignee.lower()
        return any(assignee_lower in a.lower() for a in self.assignees)
    
    def is_in_technology_field(self, field: str) -> bool:
        """Check if patent belongs to technology field"""
        return field.lower() in self.technology_field.lower()
    
    def is_filed_after(self, date: str) -> bool:
        """Check if patent was filed after given date"""
        try:
            if not self.filing_date:
                return False
            
            # Simple date comparison (assumes YYYY-MM-DD format)
            return self.filing_date >= date
        except:
            return False
    
    def is_filed_before(self, date: str) -> bool:
        """Check if patent was filed before given date"""
        try:
            if not self.filing_date:
                return False
            
            return self.filing_date <= date
        except:
            return False
    
    def get_main_classification(self) -> Optional[str]:
        """Get the main IPC classification code"""
        if self.classification_codes:
            return self.classification_codes[0]
        return None
    
    def get_ipc_section(self) -> Optional[str]:
        """Get the IPC section (first letter of classification)"""
        main_class = self.get_main_classification()
        if main_class and len(main_class) > 0:
            return main_class[0]
        return None
    
    def __str__(self) -> str:
        """String representation of patent"""
        return f"Patent({self.patent_id}: {self.title})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"Patent(id='{self.patent_id}', title='{self.title[:50]}...', "
                f"tech_field='{self.technology_field}', "
                f"assignees={len(self.assignees)}, claims={len(self.claims)})")


@dataclass
class SearchResult:
    """Search result with similarity score and metadata"""
    
    patent: Patent
    similarity_score: float
    search_query: str
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary"""
        result_dict = self.patent.to_dict()
        result_dict.update({
            'similarity_score': self.similarity_score,
            'search_query': self.search_query,
            'rank': self.rank,
            'metadata': self.metadata
        })
        return result_dict


@dataclass
class SearchFilter:
    """Search filter parameters"""
    
    technology_fields: Optional[List[str]] = None
    assignees: Optional[List[str]] = None
    inventors: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    date_range: Optional[tuple] = None  # (start_date, end_date)
    classification_codes: Optional[List[str]] = None
    status: Optional[List[str]] = None
    min_score: float = 0.0
    
    def is_empty(self) -> bool:
        """Check if filter has no constraints"""
        return all([
            not self.technology_fields,
            not self.assignees,
            not self.inventors,
            not self.countries,
            not self.date_range,
            not self.classification_codes,
            not self.status,
            self.min_score == 0.0
        ])
    
    def matches_patent(self, patent: Patent) -> bool:
        """Check if patent matches filter criteria"""
        
        # Technology field filter
        if self.technology_fields:
            if not any(patent.is_in_technology_field(field) for field in self.technology_fields):
                return False
        
        # Assignee filter
        if self.assignees:
            if not any(patent.has_assignee(assignee) for assignee in self.assignees):
                return False
        
        # Inventor filter
        if self.inventors:
            inventor_match = False
            for inventor in self.inventors:
                if any(inventor.lower() in inv.lower() for inv in patent.inventors):
                    inventor_match = True
                    break
            if not inventor_match:
                return False
        
        # Country filter
        if self.countries:
            if patent.country not in self.countries:
                return False
        
        # Date range filter
        if self.date_range:
            start_date, end_date = self.date_range
            if start_date and not patent.is_filed_after(start_date):
                return False
            if end_date and not patent.is_filed_before(end_date):
                return False
        
        # Classification filter
        if self.classification_codes:
            code_match = False
            for code in self.classification_codes:
                if any(code in patent_code for patent_code in patent.classification_codes):
                    code_match = True
                    break
            if not code_match:
                return False
        
        # Status filter
        if self.status:
            if patent.status not in self.status:
                return False
        
        return True


@dataclass
class CollectionStats:
    """Statistics about the patent collection"""
    
    total_patents: int = 0
    technology_fields: Dict[str, int] = field(default_factory=dict)
    assignees: Dict[str, int] = field(default_factory=dict)
    countries: Dict[str, int] = field(default_factory=dict)
    filing_years: Dict[str, int] = field(default_factory=dict)
    vector_size: int = 0
    distance_metric: str = ""
    indexed_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'total_patents': self.total_patents,
            'technology_fields': self.technology_fields,
            'assignees': self.assignees,
            'countries': self.countries,
            'filing_years': self.filing_years,
            'vector_size': self.vector_size,
            'distance_metric': self.distance_metric,
            'indexed_fields': self.indexed_fields
        }


# Technology field mapping based on IPC classification
IPC_TECHNOLOGY_MAPPING = {
    'A': 'Human Necessities',
    'B': 'Performing Operations/Transporting',
    'C': 'Chemistry/Metallurgy',
    'D': 'Textiles/Paper',
    'E': 'Fixed Constructions',
    'F': 'Mechanical Engineering',
    'G': 'Physics',
    'H': 'Electricity'
}

# Common technology fields for modern patents
MODERN_TECHNOLOGY_FIELDS = [
    'Artificial Intelligence',
    'Machine Learning',
    'Deep Learning',
    'Computer Vision',
    'Natural Language Processing',
    'Robotics',
    'Internet of Things',
    'Blockchain',
    'Cryptocurrency',
    'Quantum Computing',
    'Biotechnology',
    'Nanotechnology',
    'Renewable Energy',
    'Electric Vehicles',
    'Autonomous Vehicles',
    'Cybersecurity',
    'Cloud Computing',
    'Edge Computing',
    'Augmented Reality',
    'Virtual Reality',
    'Mixed Reality',
    'Voice Recognition',
    'Image Recognition',
    'Medical Devices',
    'Drug Discovery',
    'Gene Therapy',
    'CRISPR',
    'Solar Energy',
    'Wind Energy',
    'Battery Technology',
    'Wireless Communication',
    '5G Technology',
    'Semiconductor',
    'Microprocessors',
    'Memory Devices',
    'Data Storage',
    'Database Technology',
    'Software Engineering',
    'Mobile Applications',
    'Web Technology',
    'E-commerce',
    'Financial Technology',
    'Digital Payments',
    'Smart Cities',
    'Environmental Technology',
    'Clean Technology'
]

# Patent status options
PATENT_STATUS_OPTIONS = [
    'Published',
    'Granted',
    'Pending',
    'Abandoned',
    'Expired',
    'Withdrawn',
    'Rejected',
    'Under Review',
    'Allowed',
    'Final Rejection',
    'Appeal Filed',
    'Interference Declared',
    'Reissue',
    'Continuation',
    'Divisional'
]

# Country codes for patents
PATENT_COUNTRY_CODES = {
    'US': 'United States',
    'EP': 'European Patent Office',
    'JP': 'Japan',
    'CN': 'China',
    'KR': 'South Korea',
    'DE': 'Germany',
    'GB': 'United Kingdom',
    'FR': 'France',
    'CA': 'Canada',
    'AU': 'Australia',
    'IN': 'India',
    'BR': 'Brazil',
    'RU': 'Russia',
    'MX': 'Mexico',
    'IL': 'Israel',
    'SG': 'Singapore',
    'TW': 'Taiwan',
    'NZ': 'New Zealand',
    'ZA': 'South Africa',
    'MY': 'Malaysia',
    'TH': 'Thailand',
    'PH': 'Philippines',
    'VN': 'Vietnam',
    'ID': 'Indonesia',
    'CL': 'Chile',
    'AR': 'Argentina',
    'CO': 'Colombia',
    'PE': 'Peru',
    'UY': 'Uruguay',
    'EC': 'Ecuador'
}


def classify_technology_field(classification_codes: List[str]) -> str:
    """
    Classify technology field based on IPC codes
    
    Args:
        classification_codes: List of IPC classification codes
        
    Returns:
        Technology field name
    """
    if not classification_codes:
        return "Unknown"
    
    # Get the main section from the first classification code
    main_code = classification_codes[0]
    if not main_code:
        return "Unknown"
    
    # Extract the section (first character)
    section = main_code[0].upper()
    
    # Map to technology field
    return IPC_TECHNOLOGY_MAPPING.get(section, "Unknown")


def validate_patent_data(patent_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean patent data dictionary
    
    Args:
        patent_dict: Raw patent data dictionary
        
    Returns:
        Cleaned and validated patent data
    """
    
    # Required fields
    required_fields = ['patent_id', 'title']
    for field in required_fields:
        if field not in patent_dict or not patent_dict[field]:
            raise ValueError(f"Required field '{field}' is missing or empty")
    
    # Default values for optional fields
    defaults = {
        'patent_number': patent_dict.get('patent_id', ''),
        'abstract': '',
        'description': '',
        'claims': [],
        'inventors': [],
        'assignees': [],
        'filing_date': '',
        'publication_date': '',
        'priority_date': None,
        'classification_codes': [],
        'technology_field': 'Unknown',
        'country': 'US',
        'status': 'Published',
        'family_id': None,
        'citations': [],
        'cited_by': [],
        'application_number': None,
        'priority_number': None,
        'legal_status': None
    }
    
    # Merge with defaults
    validated_data = {**defaults, **patent_dict}
    
    # Clean string fields
    string_fields = ['patent_id', 'patent_number', 'title', 'abstract', 'description', 
                    'filing_date', 'publication_date', 'technology_field', 'country', 'status']
    
    for field in string_fields:
        if isinstance(validated_data[field], str):
            validated_data[field] = validated_data[field].strip()
    
    # Ensure list fields are lists
    list_fields = ['claims', 'inventors', 'assignees', 'classification_codes', 'citations', 'cited_by']
    
    for field in list_fields:
        if not isinstance(validated_data[field], list):
            if isinstance(validated_data[field], str):
                # Split string by common delimiters
                validated_data[field] = [item.strip() for item in 
                                       validated_data[field].split('|') if item.strip()]
            else:
                validated_data[field] = []
    
    # Auto-classify technology field if not provided or unknown
    if (validated_data['technology_field'] == 'Unknown' and 
        validated_data['classification_codes']):
        validated_data['technology_field'] = classify_technology_field(
            validated_data['classification_codes']
        )
    
    return validated_data


def create_sample_patent(index: int = 0) -> Patent:
    """
    Create a sample patent for testing
    
    Args:
        index: Index number for generating unique data
        
    Returns:
        Sample Patent object
    """
    
    tech_fields = [
        "Machine Learning", "Blockchain", "Internet of Things", 
        "Quantum Computing", "Biotechnology", "Renewable Energy",
        "Artificial Intelligence", "Robotics", "Nanotechnology",
        "Medical Devices"
    ]
    
    companies = [
        "TechCorp Industries", "Innovation Labs Inc", "Future Systems LLC",
        "Advanced Technologies", "Digital Solutions Group", "Smart Devices Co",
        "Quantum Innovations", "BioTech Solutions", "Green Energy Systems",
        "AI Research Institute"
    ]
    
    inventors_pool = [
        "Dr. Alice Johnson", "Prof. Bob Smith", "Dr. Carol Brown", "Dr. David Wilson",
        "Dr. Emily Davis", "Prof. Frank Miller", "Dr. Grace Lee", "Dr. Henry Taylor",
        "Dr. Irene Anderson", "Prof. Jack Thompson", "Dr. Karen White", "Dr. Louis Garcia"
    ]
    
    field_index = index % len(tech_fields)
    tech_field = tech_fields[field_index]
    
    # Generate IPC codes based on technology field
    ipc_mapping = {
        "Machine Learning": ["G06N3/00", "G06F15/18"],
        "Blockchain": ["G06Q20/00", "H04L9/00"],
        "Internet of Things": ["H04W4/00", "G01D21/00"],
        "Quantum Computing": ["G06N10/00", "H03K19/00"],
        "Biotechnology": ["C12N15/00", "A61K38/00"],
        "Renewable Energy": ["H01L31/00", "F03D1/00"],
        "Artificial Intelligence": ["G06N5/00", "G06F17/00"],
        "Robotics": ["B25J9/00", "G05B19/00"],
        "Nanotechnology": ["B82Y10/00", "H01L29/00"],
        "Medical Devices": ["A61B5/00", "A61M25/00"]
    }
    
    classification_codes = ipc_mapping.get(tech_field, ["G06F15/00"])
    
    patent = Patent(
        patent_id=f"US{10000000 + index}",
        patent_number=f"US{10000000 + index}",
        title=f"Advanced {tech_field} System and Method {index + 1}",
        abstract=f"This patent describes an innovative approach to {tech_field.lower()} "
                f"technology. The invention provides significant improvements over existing "
                f"solutions by implementing novel algorithms and architectures. The system "
                f"offers enhanced performance, reliability, and scalability for {tech_field.lower()} "
                f"applications in various industries.",
        description=f"The present invention relates to {tech_field.lower()} technology and "
                   f"specifically to systems and methods for implementing advanced {tech_field.lower()} "
                   f"solutions. The invention addresses current limitations in the field by "
                   f"providing a novel architecture that combines multiple innovative approaches. "
                   f"The system includes processors, memory, and specialized components designed "
                   f"for optimal {tech_field.lower()} performance. Implementation details include "
                   f"specific algorithms, data structures, and processing methodologies that "
                   f"enable significant improvements over prior art.",
        claims=[
            f"A method for implementing {tech_field.lower()} technology comprising: "
            f"providing a processing system with specialized components; executing algorithms "
            f"optimized for {tech_field.lower()} applications; and generating improved results.",
            f"The method of claim 1, wherein the processing system includes multiple processors "
            f"configured for parallel {tech_field.lower()} operations.",
            f"A system comprising: processors configured for {tech_field.lower()} computations; "
            f"memory storing {tech_field.lower()} algorithms; and interfaces for data input/output.",
            f"The system of claim 3, further comprising specialized hardware accelerators for "
            f"{tech_field.lower()} processing.",
            f"A computer-readable medium storing instructions that, when executed, cause a "
            f"processor to perform {tech_field.lower()} operations according to the method of claim 1."
        ],
        inventors=[
            inventors_pool[index % len(inventors_pool)],
            inventors_pool[(index + 1) % len(inventors_pool)]
        ],
        assignees=[companies[index % len(companies)]],
        filing_date=f"202{3 - (index % 4)}-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}",
        publication_date=f"202{4 - (index % 4)}-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}",
        classification_codes=classification_codes,
        technology_field=tech_field,
        country="US",
        status="Published" if index % 5 != 0 else "Granted",
        application_number=f"US{16000000 + index}",
        legal_status="Active"
    )
    
    return patent
