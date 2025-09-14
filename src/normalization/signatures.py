"""
Document layout signature learning and matching system.
"""

import json
import os
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class SignatureToken:
    """Represents a quantized structural token."""
    page: int
    element_type: str
    bbox_bucket: Tuple[int, int, int, int]  # Quantized coordinates
    token_count: int
    content_hash: str


SAME_VERSION_THRESHOLD = 0.90  # similarity to treat as same version
NEW_VERSION_THRESHOLD = 0.70   # similarity to create a new version within family

def _normalize_family_id(vendor_name: Optional[str]) -> Optional[str]:
    if not vendor_name:
        return None
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in vendor_name)
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    cleaned = cleaned.strip('_')
    return cleaned or None

@dataclass
class DocumentSignature:
    """Represents a document layout signature with optional cached field payload and versioning."""
    signature_id: str
    tokens: List[SignatureToken]
    hash_value: str
    created_at: str
    version: str = "1.0"  # schema version (not layout)
    document_count: int = 1
    sample_filenames: List[str] = None
    # Cache fields for reuse (optional)
    cached_fields: Optional[Dict[str, Any]] = None  # {key: {"value":..., "confidence":...}}
    cached_confidence: Optional[float] = None
    cached_updated_at: Optional[str] = None
    # Versioning / family fields
    family_id: Optional[str] = None          # stable family identifier (e.g. normalized vendor name)
    active_version: int = 1                  # current active layout version
    layout_versions: List[Dict[str, Any]] = field(default_factory=list)  # list of version records
    _last_version_event: Optional[str] = None  # runtime only, not persisted intentionally

    def __post_init__(self):
        if self.sample_filenames is None:
            self.sample_filenames = []


class SignatureManager:
    """Manages document layout signatures."""
    
    def __init__(self, signatures_dir: str = "signatures"):
        self.signatures_dir = signatures_dir
        self.signatures = {}
        self.jaccard_threshold = 0.85
        self._ensure_signatures_dir()
        self.load_signatures()
    
    def _ensure_signatures_dir(self):
        """Ensure signatures directory exists."""
        os.makedirs(self.signatures_dir, exist_ok=True)
    
    def load_signatures(self):
        """Load existing signatures from disk."""
        self.signatures = {}
        
        if not os.path.exists(self.signatures_dir):
            return
        
        for filename in os.listdir(self.signatures_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.signatures_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                    # Convert token data back to SignatureToken objects, ensuring bbox_bucket is a tuple
                    tokens = []
                    for token_data in data['tokens']:
                        bbox_bucket = token_data.get('bbox_bucket')
                        if isinstance(bbox_bucket, list):
                            token_data['bbox_bucket'] = tuple(bbox_bucket)
                        tokens.append(SignatureToken(**token_data))
                    
                    signature = DocumentSignature(
                        signature_id=data['signature_id'],
                        tokens=tokens,
                        hash_value=data['hash_value'],
                        created_at=data['created_at'],
                        version=data.get('version', '1.0'),
                        document_count=data.get('document_count', 1),
                        sample_filenames=data.get('sample_filenames', []),
                        cached_fields=data.get('cached_fields'),
                        cached_confidence=data.get('cached_confidence'),
                        cached_updated_at=data.get('cached_updated_at'),
                        family_id=data.get('family_id'),
                        active_version=data.get('active_version', 1),
                        layout_versions=data.get('layout_versions', [])
                    )
                    # Migration: ensure layout_versions exists
                    if not signature.layout_versions:
                        signature.layout_versions = [{
                            "version": 1,
                            "hash": signature.hash_value,
                            "token_count": len(signature.tokens),
                            "created_at": signature.created_at,
                            "document_count": signature.document_count,
                            "min_similarity": 1.0,
                            "max_similarity": 1.0
                        }]
                    
                    self.signatures[signature.signature_id] = signature
                    
                except Exception as e:
                    logger.error(f"Error loading signature {filename}: {e}")
        
        logger.info(f"Loaded {len(self.signatures)} signatures")
    
    def create_or_match_signature(self, layout_elements: List[Dict[str, Any]], filename: str) -> Tuple[DocumentSignature, float]:
        """Create or match signature with version-aware logic.

        Returns (signature, similarity_used_for_decision)
        """
        tokens = self._generate_tokens(layout_elements)
        best_match, similarity = self._find_best_match(tokens)
        if best_match:
            # Ensure version record exists
            if not best_match.layout_versions:
                best_match.layout_versions = [{
                    "version": 1,
                    "hash": best_match.hash_value,
                    "token_count": len(best_match.tokens),
                    "created_at": best_match.created_at,
                    "document_count": best_match.document_count,
                    "min_similarity": 1.0,
                    "max_similarity": 1.0
                }]
            active_version = best_match.active_version
            active_record = next((r for r in best_match.layout_versions if r["version"] == active_version), None)
            active_similarity = similarity  # proxy vs active tokens
            if active_similarity >= SAME_VERSION_THRESHOLD:
                best_match.document_count += 1
                active_record["document_count"] += 1
                active_record["min_similarity"] = min(active_record["min_similarity"], active_similarity)
                active_record["max_similarity"] = max(active_record["max_similarity"], active_similarity)
                if filename not in best_match.sample_filenames:
                    best_match.sample_filenames.append(filename)
                    if len(best_match.sample_filenames) > 5:
                        best_match.sample_filenames = best_match.sample_filenames[-5:]
                best_match._last_version_event = f"same_version(v{active_version})"
                self._save_signature(best_match)
                return best_match, active_similarity
            elif active_similarity >= NEW_VERSION_THRESHOLD:
                new_version = max(r["version"] for r in best_match.layout_versions) + 1
                version_record = {
                    "version": new_version,
                    "hash": hashlib.sha1(json.dumps([asdict(t) for t in tokens], sort_keys=True).encode()).hexdigest(),
                    "token_count": len(tokens),
                    "created_at": datetime.now().isoformat(),
                    "document_count": 1,
                    "min_similarity": active_similarity,
                    "max_similarity": active_similarity
                }
                best_match.layout_versions.append(version_record)
                best_match.active_version = new_version
                best_match.document_count += 1
                if filename not in best_match.sample_filenames:
                    best_match.sample_filenames.append(filename)
                    if len(best_match.sample_filenames) > 5:
                        best_match.sample_filenames = best_match.sample_filenames[-5:]
                best_match._last_version_event = f"new_version(v{new_version})"
                self._save_signature(best_match)
                return best_match, active_similarity
            # Too different -> create new family signature
        # New signature
        new_signature = self._create_new_signature(tokens, filename)
        new_signature._last_version_event = "new_family"
        self.signatures[new_signature.signature_id] = new_signature
        self._save_signature(new_signature)
        return new_signature, 1.0
    
    def _generate_tokens(self, layout_elements: List[Dict[str, Any]]) -> List[SignatureToken]:
        """Generate quantized structural tokens from layout elements."""
        tokens = []
        
        for element in layout_elements:
            # Quantize bounding box to 0-1000 grid
            bbox = element.get('bbox', (0, 0, 612, 792))
            page_width = element.get('page_width', 612)
            page_height = element.get('page_height', 792)
            
            # Normalize and quantize coordinates
            x1 = int((bbox[0] / page_width) * 1000) if page_width > 0 else 0
            y1 = int((bbox[1] / page_height) * 1000) if page_height > 0 else 0
            x2 = int((bbox[2] / page_width) * 1000) if page_width > 0 else 1000
            y2 = int((bbox[3] / page_height) * 1000) if page_height > 0 else 1000
            
            bbox_bucket = (x1, y1, x2, y2)
            
            # Calculate content hash
            content = element.get('content', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            
            # Create token
            token = SignatureToken(
                page=element.get('page', 1),
                element_type=element.get('type', 'text'),
                bbox_bucket=bbox_bucket,
                token_count=len(content.split()) if content else 0,
                content_hash=content_hash
            )
            
            tokens.append(token)
        
        return tokens
    
    def _find_best_match(self, tokens: List[SignatureToken]) -> Tuple[DocumentSignature, float]:
        """Find the best matching signature using Jaccard similarity."""
        best_signature = None
        best_similarity = 0.0
        
        for signature in self.signatures.values():
            similarity = self._calculate_jaccard_similarity(tokens, signature.tokens)
            if similarity > best_similarity:
                best_similarity = similarity
                best_signature = signature
        
        return best_signature, best_similarity
    
    def _calculate_jaccard_similarity(self, tokens1: List[SignatureToken], 
                                    tokens2: List[SignatureToken]) -> float:
        """Calculate Jaccard similarity between two token sets."""
        # Convert tokens to comparable tuples
        set1 = set()
        set2 = set()
        
        for token in tokens1:
            # Ensure bbox_bucket is hashable tuple
            bbox_bucket = tuple(token.bbox_bucket) if isinstance(token.bbox_bucket, (list, tuple)) else token.bbox_bucket
            token_tuple = (token.page, token.element_type, bbox_bucket, token.token_count)
            set1.add(token_tuple)

        for token in tokens2:
            bbox_bucket = tuple(token.bbox_bucket) if isinstance(token.bbox_bucket, (list, tuple)) else token.bbox_bucket
            token_tuple = (token.page, token.element_type, bbox_bucket, token.token_count)
            set2.add(token_tuple)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_new_signature(self, tokens: List[SignatureToken], filename: str) -> DocumentSignature:
        """Create a new document signature."""
        # Generate signature ID
        tokens_str = json.dumps([asdict(token) for token in tokens], sort_keys=True)
        hash_value = hashlib.sha1(tokens_str.encode()).hexdigest()
        signature_id = hash_value[:12]
        
        signature = DocumentSignature(
            signature_id=signature_id,
            tokens=tokens,
            hash_value=hash_value,
            created_at=datetime.now().isoformat(),
            sample_filenames=[filename]
        )
        # Initialize version record
        signature.layout_versions = [{
            "version": 1,
            "hash": hash_value,
            "token_count": len(tokens),
            "created_at": signature.created_at,
            "document_count": 1,
            "min_similarity": 1.0,
            "max_similarity": 1.0
        }]
        
        return signature
    
    def _save_signature(self, signature: DocumentSignature):
        """Save signature to disk."""
        try:
            filepath = os.path.join(self.signatures_dir, f"{signature.signature_id}.json")
            
            # Convert to serializable format
            data = asdict(signature)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving signature {signature.signature_id}: {e}")

    # --- Cache helpers ---
    def update_signature_cache(self, signature: DocumentSignature, fields: List[Any]):
        """Persist extracted fields into signature cache.

        fields: list of KeyValue-like objects (must have key, value, confidence attributes).
        """
        try:
            if not fields:
                return
            cache: Dict[str, Any] = {}
            confidences = []
            for f in fields:
                try:
                    cache[f.key] = {"value": f.value, "confidence": getattr(f, 'confidence', None)}
                    if getattr(f, 'confidence', None) is not None:
                        confidences.append(f.confidence)
                except Exception:
                    continue
            if not cache:
                return
            signature.cached_fields = cache
            signature.cached_confidence = sum(confidences) / len(confidences) if confidences else None
            signature.cached_updated_at = datetime.now().isoformat()
            self._save_signature(signature)
            logger.info(
                "Updated signature cache %s (%d fields, avg_conf=%.2f)",
                signature.signature_id,
                len(cache),
                signature.cached_confidence if signature.cached_confidence is not None else -1.0
            )
        except Exception as e:
            logger.error(f"Failed updating signature cache {signature.signature_id}: {e}")
    
    def get_signature_stats(self) -> Dict[str, Any]:
        """Get signature statistics."""
        total_documents = sum(sig.document_count for sig in self.signatures.values())
        
        return {
            "total_signatures": len(self.signatures),
            "total_documents": total_documents,
            "avg_documents_per_signature": total_documents / len(self.signatures) if self.signatures else 0,
            "jaccard_threshold": self.jaccard_threshold,
            "version_thresholds": {
                "same_version": SAME_VERSION_THRESHOLD,
                "new_version": NEW_VERSION_THRESHOLD
            }
        }

    # --- Backfill helper ---
    def backfill_cached_fields(self, signature_id: str, fields: List[Any]) -> bool:
        """Manually backfill cached_fields for a signature from given KeyValue-like objects.

        Returns True if updated, False otherwise.
        """
        sig = self.signatures.get(signature_id)
        if not sig:
            return False
        try:
            cache: Dict[str, Any] = {}
            confidences = []
            for f in fields:
                k = getattr(f, 'key', None)
                v = getattr(f, 'value', None)
                c = getattr(f, 'confidence', None)
                if k is None:
                    continue
                cache[k] = {"value": v, "confidence": c}
                if c is not None:
                    confidences.append(c)
            if not cache:
                return False
            sig.cached_fields = cache
            sig.cached_confidence = sum(confidences)/len(confidences) if confidences else None
            sig.cached_updated_at = datetime.now().isoformat()
            self._save_signature(sig)
            logger.info("Backfilled cached_fields for signature %s (%d fields)", signature_id, len(cache))
            return True
        except Exception as e:
            logger.error("Failed backfilling cached_fields for %s: %s", signature_id, e)
            return False

    # --- Family ID update ---
    def update_family_id(self, signature: DocumentSignature, vendor_name: Optional[str]):
        """Assign a family_id to a signature if not already set."""
        if signature.family_id:
            return
        fid = _normalize_family_id(vendor_name)
        if not fid:
            return
        signature.family_id = fid
        # Persist
        self._save_signature(signature)

    # --- Family lookup / merge helpers ---
    def find_signatures_by_family(self, family_id: str) -> List[DocumentSignature]:
        return [s for s in self.signatures.values() if s.family_id == family_id]

    def _calculate_similarity_between_signatures(self, sig_a: DocumentSignature, sig_b: DocumentSignature) -> float:
        try:
            return self._calculate_jaccard_similarity(sig_a.tokens, sig_b.tokens)
        except Exception:
            return 0.0

    def merge_signature_into_family(self, source: DocumentSignature, target: DocumentSignature) -> Optional[DocumentSignature]:
        """Merge a newly created 'family' signature into an existing family as a new version.

        Creates a new version record on target using source tokens, updates counts, migrates sample filenames,
        then deletes source signature file and removes it from in-memory map.
        Returns updated target or None if merge not performed.
        """
        if source.signature_id == target.signature_id:
            return target
        if not target.layout_versions:
            target.layout_versions = [{
                "version": 1,
                "hash": target.hash_value,
                "token_count": len(target.tokens),
                "created_at": target.created_at,
                "document_count": target.document_count,
                "min_similarity": 1.0,
                "max_similarity": 1.0
            }]
        similarity = self._calculate_similarity_between_signatures(source, target)
        new_version = max(v["version"] for v in target.layout_versions) + 1
        version_record = {
            "version": new_version,
            "hash": source.hash_value,
            "token_count": len(source.tokens),
            "created_at": datetime.now().isoformat(),
            "document_count": source.document_count,
            "min_similarity": similarity,
            "max_similarity": similarity
        }
        target.layout_versions.append(version_record)
        target.active_version = new_version
        target.document_count += source.document_count
        # Merge sample filenames
        for fn in source.sample_filenames:
            if fn not in target.sample_filenames:
                target.sample_filenames.append(fn)
        if len(target.sample_filenames) > 5:
            target.sample_filenames = target.sample_filenames[-5:]
        # If target lacks cached_fields but source has them, adopt
        if not target.cached_fields and source.cached_fields:
            target.cached_fields = source.cached_fields
            target.cached_confidence = source.cached_confidence
            target.cached_updated_at = source.cached_updated_at
        # Persist target
        self._save_signature(target)
        # Delete source signature file
        try:
            src_path = os.path.join(self.signatures_dir, f"{source.signature_id}.json")
            if os.path.exists(src_path):
                os.remove(src_path)
        except Exception as e:
            logger.warning("Failed deleting merged signature file %s: %s", source.signature_id, e)
        # Remove from map
        self.signatures.pop(source.signature_id, None)
        logger.info(
            "Merged signature %s into family %s as version %d (similarity %.2f)",
            source.signature_id,
            target.signature_id,
            new_version,
            similarity
        )
        target._last_version_event = f"merged_into_family(v{new_version})"
        return target