"""
Basic text extractor for plain text and HTML files.
"""

import logging
from typing import List, Dict, Any, Tuple
import re

from ..schema import Section

logger = logging.getLogger(__name__)


class TextExtractor:
    """Simple text extractor for text and HTML files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text content from text/HTML files.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Clean HTML if needed
            if file_path.lower().endswith(('.html', '.htm')):
                content = self._strip_html_tags(content)
            
            layout_elements = [{
                'content': content,
                'type': 'text',
                'bbox': (0, 0, 612, 792),  # Standard letter size
                'page': 1,
                'page_width': 612.0,
                'page_height': 792.0
            }]
            
            metadata = {
                'file_encoding': 'utf-8',
                'character_count': len(content),
                'line_count': len(content.splitlines())
            }
            
            return content, layout_elements, metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _strip_html_tags(self, html_content: str) -> str:
        """Remove HTML tags and return plain text."""
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', html_content)
        
        # Replace common HTML entities
        replacements = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'"
        }
        
        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)
        
        return text.strip()
    
    def convert_to_sections(self, layout_elements: List[Dict[str, Any]]) -> List[Section]:
        """Convert layout elements to document sections."""
        sections = []
        
        for element in layout_elements:
            if element.get('type') == 'text' and element.get('content'):
                # Split content into paragraphs
                paragraphs = [p.strip() for p in element['content'].split('\n\n') if p.strip()]
                
                if len(paragraphs) == 1:
                    # Single section
                    section = Section(
                        title="Document Content",
                        content=element['content'],
                        level=1
                    )
                    sections.append(section)
                else:
                    # Multiple sections for each paragraph
                    for i, para in enumerate(paragraphs):
                        section = Section(
                            title=f"Section {i+1}",
                            content=para,
                            level=1
                        )
                        sections.append(section)
        
        return sections