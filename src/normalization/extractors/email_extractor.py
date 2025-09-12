"""
Basic email extractor for .eml files.
"""

import email
import logging
from typing import List, Dict, Any, Tuple
import re

from ..schema import Section, BoundingBox

logger = logging.getLogger(__name__)


class EmailExtractor:
    """Simple email extractor for .eml files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract content from email files.
        
        Returns:
            Tuple of (full_text, layout_elements, metadata)
        """
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            # Extract headers
            headers = self._extract_headers(msg)
            
            # Extract body
            body = self._extract_body(msg)
            
            # Create full text
            full_text = self._create_full_text(headers, body)
            
            # Create layout elements
            layout_elements = self._create_layout_elements(headers, body)
            
            # Create metadata
            metadata = {
                'message_type': 'email',
                'is_multipart': msg.is_multipart(),
                'content_type': msg.get_content_type()
            }
            
            return full_text, layout_elements, metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting email {file_path}: {e}")
            raise
    
    def _extract_headers(self, msg) -> Dict[str, str]:
        """Extract email headers."""
        headers = {}
        
        header_fields = ['From', 'To', 'Subject', 'Date', 'Cc', 'Bcc']
        
        for field in header_fields:
            value = msg.get(field)
            if value:
                headers[field.lower()] = str(value)
        
        return headers
    
    def _extract_body(self, msg) -> str:
        """Extract email body content."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body += payload.decode(charset, errors='ignore')
                    except Exception:
                        pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='ignore')
            except Exception:
                pass
        
        return body.strip()
    
    def _create_full_text(self, headers: Dict[str, str], body: str) -> str:
        """Create full text representation."""
        lines = []
        
        for field in ['from', 'to', 'subject', 'date']:
            if field in headers:
                lines.append(f"{field.title()}: {headers[field]}")
        
        lines.append("")  # Empty line
        
        if body:
            lines.append(body)
        
        return "\n".join(lines)
    
    def _create_layout_elements(self, headers: Dict[str, str], body: str) -> List[Dict[str, Any]]:
        """Create layout elements from email parts."""
        elements = []
        y_pos = 0
        line_height = 20
        page_width = 800.0
        
        # Headers
        for field in ['from', 'to', 'subject', 'date']:
            if field in headers:
                element = {
                    'content': f"{field.title()}: {headers[field]}",
                    'type': 'header',
                    'bbox': (10, y_pos, page_width - 10, y_pos + line_height),
                    'page': 1,
                    'page_width': page_width,
                    'page_height': 600.0
                }
                elements.append(element)
                y_pos += line_height + 5
        
        # Separator
        y_pos += 10
        
        # Body
        if body:
            body_height = max(100, len(body) // 80 * line_height)
            element = {
                'content': body,
                'type': 'text',
                'bbox': (10, y_pos, page_width - 10, y_pos + body_height),
                'page': 1,
                'page_width': page_width,
                'page_height': 600.0
            }
            elements.append(element)
        
        return elements
    
    def convert_to_sections(self, layout_elements: List[Dict[str, Any]]) -> List[Section]:
        """Convert layout elements to document sections."""
        sections = []
        
        headers = [e for e in layout_elements if e.get('type') == 'header']
        text_elements = [e for e in layout_elements if e.get('type') == 'text']
        
        # Headers section
        if headers:
            header_content = '\n'.join(e['content'] for e in headers)
            bbox = BoundingBox(
                x1=headers[0]['bbox'][0],
                y1=headers[0]['bbox'][1],
                x2=headers[-1]['bbox'][2],
                y2=headers[-1]['bbox'][3],
                page=1
            )
            sections.append(Section(
                title="Email Headers",
                content=header_content,
                level=1,
                bbox=bbox
            ))
        
        # Body section
        if text_elements:
            body_content = '\n'.join(e['content'] for e in text_elements)
            bbox = BoundingBox(
                x1=text_elements[0]['bbox'][0],
                y1=text_elements[0]['bbox'][1],
                x2=text_elements[-1]['bbox'][2],
                y2=text_elements[-1]['bbox'][3],
                page=1
            )
            sections.append(Section(
                title="Email Body",
                content=body_content,
                level=1,
                bbox=bbox
            ))
        
        return sections