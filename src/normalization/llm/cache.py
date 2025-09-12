"""
SHA256-based cache for LLM responses to avoid duplicate API calls.

Caches responses based on a combination of signature_id, content_hash,
and the set of missing fields being requested.
"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """File-based cache for LLM responses."""
    
    def __init__(self, cache_dir: str = "llm_cache", enabled: bool = True):
        """
        Initialize LLM response cache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            logger.debug(f"LLM cache initialized at: {self.cache_dir}")
    
    def _generate_cache_key(self, signature_id: str, content_hash: str, 
                           missing_fields: List[str]) -> str:
        """
        Generate cache key from inputs.
        
        Args:
            signature_id: Document signature ID
            content_hash: SHA256 hash of document content
            missing_fields: List of missing field names (sorted)
            
        Returns:
            SHA256 hash to use as cache key
        """
        # Sort missing fields for consistent cache keys
        sorted_fields = sorted(missing_fields)
        
        # Create cache key components
        key_components = [
            signature_id or "no_signature",
            content_hash,
            ",".join(sorted_fields)
        ]
        
        # Generate SHA256 hash
        key_string = "|".join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()
        
        return cache_key
    
    def get(self, signature_id: str, content_hash: str, 
            missing_fields: List[str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if available.
        
        Args:
            signature_id: Document signature ID
            content_hash: SHA256 hash of document content
            missing_fields: List of missing field names
            
        Returns:
            Cached response data or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_cache_key(signature_id, content_hash, missing_fields)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
        
        return None
    
    def set(self, signature_id: str, content_hash: str, missing_fields: List[str], 
            response_data: Dict[str, Any]) -> bool:
        """
        Store response in cache.
        
        Args:
            signature_id: Document signature ID
            content_hash: SHA256 hash of document content
            missing_fields: List of missing field names
            response_data: LLM response data to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._generate_cache_key(signature_id, content_hash, missing_fields)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Add metadata to cached response
            cached_data = {
                "cache_key": cache_key,
                "signature_id": signature_id,
                "content_hash": content_hash,
                "missing_fields": sorted(missing_fields),
                "cached_at": str(Path(__file__).stat().st_mtime),  # Simple timestamp
                "response_data": response_data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
            
            logger.debug(f"Cached response for key: {cache_key[:8]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")
            return False
    
    def clear(self) -> int:
        """
        Clear all cached responses.
        
        Returns:
            Number of cache files removed
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0
        
        removed_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                removed_count += 1
            
            logger.info(f"Cleared {removed_count} cache files")
            
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False, "cache_files": 0, "total_size_bytes": 0}
        
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "enabled": True,
                "cache_files": len(cache_files),
                "total_size_bytes": total_size,
                "cache_dir": str(self.cache_dir)
            }
            
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"enabled": True, "cache_files": 0, "total_size_bytes": 0, "error": str(e)}