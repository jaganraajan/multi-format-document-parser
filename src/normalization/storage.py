"""
Document storage and repository management.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .schema import NormalizedDocument

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Simple document repository with JSON storage."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir
        self.index_file = os.path.join(outputs_dir, "document_index.json")
        self._ensure_outputs_dir()
        self.index = self._load_index()
    
    def _ensure_outputs_dir(self):
        """Ensure outputs directory exists."""
        os.makedirs(self.outputs_dir, exist_ok=True)
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load document index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading document index: {e}")
        
        return {}
    
    def _save_index(self):
        """Save document index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving document index: {e}")
    
    def save_document(self, document: NormalizedDocument) -> str:
        """
        Save normalized document to repository.
        
        Returns:
            File path where document was saved
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{document.doc_id}_{timestamp}.json"
            filepath = os.path.join(self.outputs_dir, filename)
            
            # Save document
            with open(filepath, 'w') as f:
                json.dump(document.to_dict(), f, indent=2, default=str)
            
            # Update index
            self.index[document.doc_id] = {
                "doc_id": document.doc_id,
                "filename": document.ingest_metadata.filename,
                "file_type": document.ingest_metadata.file_type,
                "file_size": document.ingest_metadata.file_size,
                "uploaded_at": document.ingest_metadata.uploaded_at,
                "content_hash": document.ingest_metadata.content_hash,
                "signature_id": document.processing_meta.signature_id,
                "signature_match_score": document.processing_meta.signature_match_score,
                "total_cost_usd": document.processing_meta.total_cost_usd,
                "model_calls_made": document.processing_meta.model_calls_made,
                "coverage_stats": document.processing_meta.coverage_stats,
                "saved_at": datetime.now().isoformat(),
                "filepath": filepath,
                "sections_count": len(document.sections),
                "key_values_count": len(document.key_values),
                "tables_count": len(document.tables),
                "chunks_count": len(document.chunks)
            }
            
            self._save_index()
            
            logger.info(f"Saved document {document.doc_id} to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving document {document.doc_id}: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        """Retrieve a document by ID."""
        if doc_id not in self.index:
            return None
        
        try:
            filepath = self.index[doc_id]["filepath"]
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Note: This is a simplified reconstruction
            # In a full implementation, you'd properly reconstruct all dataclass objects
            return data
            
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None
    
    def search_documents(self, **filters) -> List[Dict[str, Any]]:
        """
        Search documents with filters.
        
        Supported filters:
        - file_type: Filter by file type
        - min_coverage: Minimum rule coverage
        - max_cost: Maximum cost
        """
        results = []
        
        for doc_info in self.index.values():
            # Apply filters
            if "file_type" in filters:
                if doc_info.get("file_type") != filters["file_type"]:
                    continue
            
            if "min_coverage" in filters:
                coverage_stats = doc_info.get("coverage_stats", {})
                rule_coverage = coverage_stats.get("rule_coverage", 0)
                if rule_coverage < filters["min_coverage"]:
                    continue
            
            if "max_cost" in filters:
                if doc_info.get("total_cost_usd", 0) > filters["max_cost"]:
                    continue
            
            results.append(doc_info)
        
        # Sort by upload date (newest first)
        results.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if not self.index:
            return {
                "total_documents": 0,
                "total_size_bytes": 0,
                "file_types": {},
                "avg_cost_per_doc": 0.0,
                "total_cost": 0.0
            }
        
        total_documents = len(self.index)
        total_size = sum(doc.get("file_size", 0) for doc in self.index.values())
        total_cost = sum(doc.get("total_cost_usd", 0) for doc in self.index.values())
        
        # Count file types
        file_types = {}
        for doc in self.index.values():
            file_type = doc.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_size_bytes": total_size,
            "file_types": file_types,
            "avg_cost_per_doc": total_cost / total_documents if total_documents > 0 else 0.0,
            "total_cost": total_cost,
            "avg_size_bytes": total_size / total_documents if total_documents > 0 else 0
        }
    
    def cleanup_old_documents(self, days_to_keep: int = 30):
        """Remove documents older than specified days."""
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        removed_count = 0
        
        for doc_id, doc_info in list(self.index.items()):
            try:
                uploaded_at = datetime.fromisoformat(doc_info["uploaded_at"].replace('Z', '+00:00'))
                if uploaded_at.timestamp() < cutoff_time:
                    # Remove file
                    filepath = doc_info["filepath"]
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    
                    # Remove from index
                    del self.index[doc_id]
                    removed_count += 1
                    
            except Exception as e:
                logger.error(f"Error cleaning up document {doc_id}: {e}")
        
        if removed_count > 0:
            self._save_index()
            logger.info(f"Cleaned up {removed_count} old documents")
        
        return removed_count