"""
Basic PDF extractor using pdfplumber for text and layout extraction.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pdfplumber

from ..schema import Section, BoundingBox

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Simple PDF extractor using pdfplumber."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text and basic layout from PDF.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        full_text = ""
        layout_elements = []
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata['page_count'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                    
                    # Create simple layout elements
                    if page_text.strip():
                        layout_elements.append({
                            'content': page_text.strip(),
                            'type': 'text',
                            'bbox': (0, 0, page.width, page.height),
                            'page': page_num,
                            'page_width': page.width,
                            'page_height': page.height
                        })
                        
        except Exception as e:
            self.logger.error(f"Error extracting PDF {file_path}: {e}")
            raise
        
        return full_text, layout_elements, metadata
    
    def convert_to_sections(self, layout_elements: List[Dict[str, Any]]) -> List[Section]:
        """Convert layout elements to document sections."""
        sections = []
        
        for i, element in enumerate(layout_elements):
            if element.get('type') == 'text' and element.get('content'):
                bbox = None
                if element.get('bbox'):
                    bbox = BoundingBox(
                        x1=element['bbox'][0],
                        y1=element['bbox'][1],
                        x2=element['bbox'][2],
                        y2=element['bbox'][3],
                        page=element.get('page', 1)
                    )
                
                section = Section(
                    title=f"Page {element.get('page', i+1)}",
                    content=element['content'],
                    level=1,
                    bbox=bbox
                )
                sections.append(section)
        
        return sections