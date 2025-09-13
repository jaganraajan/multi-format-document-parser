"""
Document layout signature learning and matching system.
"""

import json
import os
import logging
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SignatureToken:
    """Represents a quantized structural token."""
    page: int
    element_type: str
    bbox_bucket: Tuple[int, int, int, int]  # Quantized coordinates
    token_count: int
    content_hash: str


@dataclass
class DocumentSignature:
    """Represents a document layout signature."""
    signature_id: str
    tokens: List[SignatureToken]
    hash_value: str
    created_at: str
    version: str = "1.0"
    document_count: int = 1
    sample_filenames: List[str] = None
    
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
                        sample_filenames=data.get('sample_filenames', [])
                    )
                    
                    self.signatures[signature.signature_id] = signature
                    
                except Exception as e:
                    logger.error(f"Error loading signature {filename}: {e}")
        
        logger.info(f"Loaded {len(self.signatures)} signatures")
    
    def create_or_match_signature(self, layout_elements: List[Dict[str, Any]], 
                                filename: str) -> Tuple[DocumentSignature, float]:
        """
        Create or match a document signature.
        
        Returns:
            Tuple of (signature, similarity_score)
        """
        # Generate tokens from layout elements
        tokens = self._generate_tokens(layout_elements)
        
        # Try to match existing signatures
        best_match, similarity = self._find_best_match(tokens)
        
        if best_match and similarity >= self.jaccard_threshold:
            # Update existing signature
            best_match.document_count += 1
            if filename not in best_match.sample_filenames:
                best_match.sample_filenames.append(filename)
                # Keep only last 5 filenames
                if len(best_match.sample_filenames) > 5:
                    best_match.sample_filenames = best_match.sample_filenames[-5:]
            
            self._save_signature(best_match)
            return best_match, similarity
        
        else:
            # Create new signature
            new_signature = self._create_new_signature(tokens, filename)
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
    
    def get_signature_stats(self) -> Dict[str, Any]:
        """Get signature statistics."""
        total_documents = sum(sig.document_count for sig in self.signatures.values())
        
        return {
            "total_signatures": len(self.signatures),
            "total_documents": total_documents,
            "avg_documents_per_signature": total_documents / len(self.signatures) if self.signatures else 0,
            "jaccard_threshold": self.jaccard_threshold
        }