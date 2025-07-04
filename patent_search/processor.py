"""
Patent Search Tool - Text Processing
Text processing and cleaning utilities for patent data.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
import logging
from bs4 import BeautifulSoup
import unicodedata

from .models import Patent, classify_technology_field

logger = logging.getLogger(__name__)


class PatentDataProcessor:
    """Process and clean patent data from various sources"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_char_pattern = re.compile(r'[^\w\s\.,\-\(\)\/\:]')
        self.claim_number_pattern = re.compile(r'\n?\s*(\d+)\.\s*')
        self.reference_pattern = re.compile(r'\[\d+\]|\(\d+\)')
        
        # Common abbreviations and expansions
        self.abbreviations = {
            'et al.': 'and others',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so forth',
            'vs.': 'versus',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'LLC': 'Limited Liability Company',
            'Ltd.': 'Limited'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove HTML/XML tags
        text = self.html_tag_pattern.sub('', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Expand common abbreviations
        for abbrev, expansion in self.abbreviations.items():
            text = text.replace(abbrev, expansion)
        
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove most special characters but keep important punctuation
        text = self.special_char_pattern.sub(' ', text)
        
        # Clean up multiple spaces again
        text = self.whitespace_pattern.sub(' ', text.strip())
        
        return text
    
    def extract_claims(self, claims_text: str) -> List[str]:
        """
        Extract individual claims from claims text
        
        Args:
            claims_text: Raw claims text with numbered claims
            
        Returns:
            List of individual claim texts
        """
        if not claims_text:
            return []
        
        # Clean the text first
        claims_text = self.clean_text(claims_text)
        
        # Split on claim numbers (1., 2., etc.)
        claim_parts = self.claim_number_pattern.split(claims_text)
        
        claims = []
        for i in range(1, len(claim_parts), 2):  # Skip empty parts and numbers
            if i + 1 < len(claim_parts):
                claim_text = claim_parts[i + 1].strip()
                
                # Filter out very short claims (likely parsing errors)
                if len(claim_text) > 20:
                    # Remove reference numbers
                    claim_text = self.reference_pattern.sub('', claim_text)
                    
                    # Clean up the claim text
                    claim_text = self.clean_text(claim_text)
                    
                    if claim_text:
                        claims.append(claim_text)
        
        return claims
    
    def extract_inventors(self, inventor_data: Any) -> List[str]:
        """
        Extract inventor names from various data formats
        
        Args:
            inventor_data: Inventor data (string, list, or dict)
            
        Returns:
            List of inventor names
        """
        inventors = []
        
        if isinstance(inventor_data, str):
            # Split by common delimiters
            inventors = [name.strip() for name in re.split(r'[;,|]', inventor_data) if name.strip()]
        elif isinstance(inventor_data, list):
            for item in inventor_data:
                if isinstance(item, str):
                    inventors.append(item.strip())
                elif isinstance(item, dict):
                    # Try common name fields
                    name = item.get('name') or item.get('full_name') or item.get('inventor_name')
                    if name:
                        inventors.append(str(name).strip())
        elif isinstance(inventor_data, dict):
            name = inventor_data.get('name') or inventor_data.get('full_name')
            if name:
                inventors.append(str(name).strip())
        
        # Clean inventor names
        cleaned_inventors = []
        for inventor in inventors:
            # Remove titles and suffixes
            inventor = re.sub(r'\b(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.|Jr\.|Sr\.|PhD|MD)\b', '', inventor)
            inventor = self.clean_text(inventor)
            if inventor:
                cleaned_inventors.append(inventor)
        
        return cleaned_inventors
    
    def extract_assignees(self, assignee_data: Any) -> List[str]:
        """
        Extract assignee names from various data formats
        
        Args:
            assignee_data: Assignee data (string, list, or dict)
            
        Returns:
            List of assignee names
        """
        assignees = []
        
        if isinstance(assignee_data, str):
            # Split by common delimiters
            assignees = [name.strip() for name in re.split(r'[;,|]', assignee_data) if name.strip()]
        elif isinstance(assignee_data, list):
            for item in assignee_data:
                if isinstance(item, str):
                    assignees.append(item.strip())
                elif isinstance(item, dict):
                    # Try common name fields
                    name = (item.get('name') or item.get('assignee_name') or 
                           item.get('organization') or item.get('company'))
                    if name:
                        assignees.append(str(name).strip())
        elif isinstance(assignee_data, dict):
            name = (assignee_data.get('name') or assignee_data.get('assignee_name') or
                   assignee_data.get('organization'))
            if name:
                assignees.append(str(name).strip())
        
        # Clean assignee names
        cleaned_assignees = []
        for assignee in assignees:
            assignee = self.clean_text(assignee)
            if assignee:
                cleaned_assignees.append(assignee)
        
        return cleaned_assignees
    
    def extract_classification_codes(self, classification_data: Any) -> List[str]:
        """
        Extract classification codes from various data formats
        
        Args:
            classification_data: Classification data
            
        Returns:
            List of classification codes
        """
        codes = []
        
        if isinstance(classification_data, str):
            # Split by common delimiters
            codes = [code.strip() for code in re.split(r'[;,|]', classification_data) if code.strip()]
        elif isinstance(classification_data, list):
            for item in classification_data:
                if isinstance(item, str):
                    codes.append(item.strip())
                elif isinstance(item, dict):
                    # Try common code fields
                    code = (item.get('code') or item.get('classification_code') or 
                           item.get('ipc_code') or item.get('class'))
                    if code:
                        codes.append(str(code).strip())
        elif isinstance(classification_data, dict):
            code = (classification_data.get('code') or classification_data.get('classification_code') or
                   classification_data.get('ipc_code'))
            if code:
                codes.append(str(code).strip())
        
        # Clean and validate codes
        cleaned_codes = []
        for code in codes:
            # Basic IPC code pattern validation
            code = code.strip().upper()
            if re.match(r'^[A-H]\d{2}[A-Z]\d+/\d+', code):
                cleaned_codes.append(code)
            elif len(code) >= 3:  # Accept other classification schemes
                cleaned_codes.append(code)
        
        return cleaned_codes
    
    def parse_patent_xml(self, xml_content: str) -> Optional[Patent]:
        """
        Parse patent data from XML format (USPTO style)
        
        Args:
            xml_content: XML content string
            
        Returns:
            Patent object or None if parsing fails
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract basic information
            patent_id_elem = root.find('.//publication-reference/document-id/doc-number')
            patent_id = patent_id_elem.text if patent_id_elem is not None else ""
            
            title_elem = root.find('.//invention-title')
            title = self.clean_text(title_elem.text) if title_elem is not None else ""
            
            abstract_elem = root.find('.//abstract')
            abstract = ""
            if abstract_elem is not None:
                abstract_text = ET.tostring(abstract_elem, encoding='unicode', method='text')
                abstract = self.clean_text(abstract_text)
            
            # Extract description
            description_elem = root.find('.//detailed-description')
            description = ""
            if description_elem is not None:
                desc_text = ET.tostring(description_elem, encoding='unicode', method='text')
                description = self.clean_text(desc_text)
            
            # Extract claims
            claims_section = root.find('.//claims')
            claims = []
            if claims_section is not None:
                claims_text = ET.tostring(claims_section, encoding='unicode', method='text')
                claims = self.extract_claims(claims_text)
            
            # Extract inventors
            inventors = []
            inventor_elements = root.findall('.//inventors/inventor')
            for inventor_elem in inventor_elements:
                name_elem = inventor_elem.find('.//name')
                if name_elem is not None:
                    # Try to get full name or construct from parts
                    full_name = name_elem.text
                    if not full_name:
                        first = inventor_elem.find('.//first-name')
                        last = inventor_elem.find('.//last-name')
                        if first is not None and last is not None:
                            full_name = f"{first.text} {last.text}"
                    
                    if full_name:
                        inventors.append(self.clean_text(full_name))
            
            # Extract assignees
            assignees = []
            assignee_elements = root.findall('.//assignees/assignee')
            for assignee_elem in assignee_elements:
                name_elem = assignee_elem.find('.//orgname')
                if name_elem is None:
                    name_elem = assignee_elem.find('.//name')
                
                if name_elem is not None and name_elem.text:
                    assignees.append(self.clean_text(name_elem.text))
            
            # Extract dates
            filing_date_elem = root.find('.//application-reference/document-id/date')
            filing_date = filing_date_elem.text if filing_date_elem is not None else ""
            
            publication_date_elem = root.find('.//publication-reference/document-id/date')
            publication_date = publication_date_elem.text if publication_date_elem is not None else ""
            
            # Extract classification codes
            classification_codes = []
            ipc_elements = root.findall('.//classification-ipc/main-classification')
            for ipc_elem in ipc_elements:
                if ipc_elem.text:
                    classification_codes.append(ipc_elem.text.strip())
            
            # Also check for other classification systems
            cpc_elements = root.findall('.//classification-cpc/main-cpc/classification-cpc')
            for cpc_elem in cpc_elements:
                code_elem = cpc_elem.find('.//cpc-version-indicator/date')
                if code_elem is not None and code_elem.text:
                    classification_codes.append(code_elem.text.strip())
            
            # Determine technology field
            technology_field = classify_technology_field(classification_codes)
            
            # Extract country (usually from document ID)
            country = "US"  # Default
            country_elem = root.find('.//publication-reference/document-id/country')
            if country_elem is not None and country_elem.text:
                country = country_elem.text.strip()
            
            # Create Patent object
            patent = Patent(
                patent_id=patent_id,
                patent_number=patent_id,
                title=title,
                abstract=abstract,
                description=description,
                claims=claims,
                inventors=inventors,
                assignees=assignees,
                filing_date=self._format_date(filing_date),
                publication_date=self._format_date(publication_date),
                classification_codes=classification_codes,
                technology_field=technology_field,
                country=country,
                status="Published"
            )
            
            return patent
            
        except Exception as e:
            logger.error(f"Error parsing patent XML: {e}")
            return None
    
    def parse_patent_html(self, html_content: str) -> Optional[Patent]:
        """
        Parse patent data from HTML format
        
        Args:
            html_content: HTML content string
            
        Returns:
            Patent object or None if parsing fails
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_elem = soup.find(['h1', 'title', '[class*="title"]'])
            title = self.clean_text(title_elem.get_text()) if title_elem else ""
            
            # Extract patent number/ID
            patent_id = ""
            id_patterns = [
                r'US\d{7,}[A-Z]?\d*',
                r'Patent\s+No\.?\s*(\d+)',
                r'Publication\s+No\.?\s*(US\d+)'
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    patent_id = match.group(1) if match.groups() else match.group()
                    break
            
            # Extract abstract
            abstract_elem = soup.find(['div', 'section'], class_=re.compile(r'abstract', re.I))
            abstract = ""
            if abstract_elem:
                abstract = self.clean_text(abstract_elem.get_text())
            
            # Extract description
            desc_elem = soup.find(['div', 'section'], class_=re.compile(r'description|detail', re.I))
            description = ""
            if desc_elem:
                description = self.clean_text(desc_elem.get_text())
            
            # Extract claims
            claims_elem = soup.find(['div', 'section'], class_=re.compile(r'claims?', re.I))
            claims = []
            if claims_elem:
                claims_text = claims_elem.get_text()
                claims = self.extract_claims(claims_text)
            
            # Extract inventors and assignees from text
            inventors = []
            assignees = []
            
            # Look for inventor patterns
            inventor_patterns = [
                r'Inventor[s]?:?\s*([^,\n]+(?:,[^,\n]+)*)',
                r'By:?\s*([^,\n]+(?:,[^,\n]+)*)'
            ]
            
            for pattern in inventor_patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    inventor_text = match.group(1)
                    inventors = self.extract_inventors(inventor_text)
                    break
            
            # Look for assignee patterns
            assignee_patterns = [
                r'Assignee[s]?:?\s*([^\n]+)',
                r'Applicant[s]?:?\s*([^\n]+)'
            ]
            
            for pattern in assignee_patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    assignee_text = match.group(1)
                    assignees = self.extract_assignees(assignee_text)
                    break
            
            # Create basic patent object
            patent = Patent(
                patent_id=patent_id or "UNKNOWN",
                patent_number=patent_id or "UNKNOWN",
                title=title,
                abstract=abstract,
                description=description,
                claims=claims,
                inventors=inventors,
                assignees=assignees,
                filing_date="",
                publication_date="",
                classification_codes=[],
                technology_field="Unknown",
                country="US",
                status="Published"
            )
            
            return patent
            
        except Exception as e:
            logger.error(f"Error parsing patent HTML: {e}")
            return None
    
    def _format_date(self, date_str: str) -> str:
        """
        Format date string to standard format (YYYY-MM-DD)
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Formatted date string
        """
        if not date_str:
            return ""
        
        # Remove any non-digit characters except hyphens
        date_str = re.sub(r'[^\d\-]', '', date_str)
        
        # Handle different date formats
        if len(date_str) == 8:  # YYYYMMDD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif len(date_str) == 6:  # YYMMDD
            year = int(date_str[:2])
            year = 2000 + year if year < 50 else 1900 + year
            return f"{year}-{date_str[2:4]}-{date_str[4:6]}"
        elif '-' in date_str and len(date_str) == 10:  # Already formatted
            return date_str
        
        return date_str
    
    def enhance_patent_data(self, patent: Patent) -> Patent:
        """
        Enhance patent data with additional processing
        
        Args:
            patent: Patent object to enhance
            
        Returns:
            Enhanced Patent object
        """
        
        # Auto-classify technology field if unknown
        if patent.technology_field == "Unknown" and patent.classification_codes:
            patent.technology_field = classify_technology_field(patent.classification_codes)
        
        # Enhance technology field based on content analysis
        if patent.technology_field == "Unknown":
            patent.technology_field = self._classify_from_content(patent)
        
        # Clean and validate data
        patent.title = self.clean_text(patent.title)
        patent.abstract = self.clean_text(patent.abstract)
        patent.description = self.clean_text(patent.description)
        
        # Clean claims
        cleaned_claims = []
        for claim in patent.claims:
            cleaned_claim = self.clean_text(claim)
            if len(cleaned_claim) > 10:  # Filter very short claims
                cleaned_claims.append(cleaned_claim)
        patent.claims = cleaned_claims
        
        return patent
    
    def _classify_from_content(self, patent: Patent) -> str:
        """
        Classify technology field based on patent content
        
        Args:
            patent: Patent object
            
        Returns:
            Technology field classification
        """
        
        # Combine all text content
        content = f"{patent.title} {patent.abstract} {' '.join(patent.claims[:3])}"
        content = content.lower()
        
        # Technology keyword mapping
        tech_keywords = {
            'Machine Learning': ['machine learning', 'neural network', 'deep learning', 'artificial intelligence', 'ai'],
            'Blockchain': ['blockchain', 'distributed ledger', 'cryptocurrency', 'bitcoin', 'smart contract'],
            'Internet of Things': ['iot', 'internet of things', 'sensor network', 'connected device', 'smart device'],
            'Quantum Computing': ['quantum', 'qubit', 'quantum computing', 'quantum algorithm'],
            'Biotechnology': ['biotech', 'dna', 'gene', 'protein', 'genomic', 'biological'],
            'Renewable Energy': ['solar', 'wind energy', 'renewable', 'photovoltaic', 'clean energy'],
            'Medical Devices': ['medical device', 'diagnostic', 'therapeutic', 'implant', 'surgical'],
            'Cybersecurity': ['cybersecurity', 'encryption', 'security', 'authentication', 'firewall'],
            'Robotics': ['robot', 'robotic', 'automation', 'actuator', 'servo'],
            'Nanotechnology': ['nano', 'nanoparticle', 'nanoscale', 'molecular'],
            'Telecommunications': ['telecommunication', 'wireless', 'cellular', '5g', 'antenna'],
            'Semiconductor': ['semiconductor', 'microprocessor', 'integrated circuit', 'chip', 'transistor'],
            'Software': ['software', 'algorithm', 'computer program', 'application', 'code'],
            'Database': ['database', 'data storage', 'data management', 'sql', 'nosql'],
            'Cloud Computing': ['cloud', 'distributed computing', 'virtualization', 'saas'],
            'Automotive': ['vehicle', 'automotive', 'car', 'autonomous', 'electric vehicle']
        }
        
        # Count keyword matches
        field_scores = {}
        for field, keywords in tech_keywords.items():
            score = sum(content.count(keyword) for keyword in keywords)
            if score > 0:
                field_scores[field] = score
        
        # Return field with highest score
        if field_scores:
            return max(field_scores, key=field_scores.get)
        
        return "Unknown"
