"""
Normalized Document Schema for Multi-Format Document Parser.

This module defines the core data structures for representing normalized documents
with consistent schema across different input formats (PDF, text, email, etc.).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import uuid


@dataclass
class BoundingBox:
    """Represents a bounding box for layout elements."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 1


@dataclass
class KeyValue:
    """Represents an extracted key-value pair."""
    key: str
    value: Any
    confidence: float
    extraction_method: str  # "rule", "model", "manual"
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """Represents a document section."""
    title: str
    content: str
    level: int = 1
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """Represents a table in the document."""
    rows: List[List[str]]
    headers: List[str] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a text chunk for RAG processing."""
    content: str
    chunk_id: str
    start_page: int
    end_page: int
    tokens: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestMetadata:
    """Metadata about document ingestion."""
    filename: str
    file_size: int
    file_type: str
    uploaded_at: str
    content_hash: str
    page_count: int = 1
    processing_time_seconds: float = 0.0


@dataclass
class ProcessingMeta:
    """Metadata about document processing."""
    pipeline_version: str = "1.0.0"
    signature_id: Optional[str] = None
    signature_match_score: float = 0.0
    rules_applied: List[str] = field(default_factory=list)
    model_calls_made: int = 0
    total_cost_usd: float = 0.0
    coverage_stats: Dict[str, Any] = field(default_factory=dict)
    model_call_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NormalizedDocument:
    """Main normalized document structure."""
    doc_id: str
    ingest_metadata: IngestMetadata
    sections: List[Section] = field(default_factory=list)
    key_values: List[KeyValue] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    processing_meta: ProcessingMeta = field(default_factory=ProcessingMeta)
    interpretation_log_path: Optional[str] = None

    @classmethod
    def create(cls, filename: str, file_size: int, file_type: str, content: bytes) -> 'NormalizedDocument':
        """Create a new normalized document."""
        # Generate document ID
        doc_id = str(uuid.uuid4())[:8]
        
        # Calculate content hash
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Create ingest metadata
        ingest_metadata = IngestMetadata(
            filename=filename,
            file_size=file_size,
            file_type=file_type,
            uploaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            content_hash=content_hash
        )
        
        return cls(
            doc_id=doc_id,
            ingest_metadata=ingest_metadata,
            processing_meta=ProcessingMeta()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            else:
                return obj
        
        return convert_dataclass(self)


def calculate_coverage_stats(document: NormalizedDocument) -> Dict[str, Any]:
    """Calculate coverage statistics for a document."""
    total_fields = len(document.key_values)
    rule_based = len([kv for kv in document.key_values if kv.extraction_method == "rule"])
    model_based = len([kv for kv in document.key_values if kv.extraction_method == "model"])
    
    return {
        "required_fields_total": total_fields,
        "extracted": total_fields,
        "rule_based": rule_based,
        "model_based": model_based,
        "rule_coverage": rule_based / total_fields if total_fields > 0 else 0.0,
        "model_coverage": model_based / total_fields if total_fields > 0 else 0.0
    }