"""
Main document processing pipeline.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .schema import NormalizedDocument, Section, Chunk, calculate_coverage_stats
from .signatures import SignatureManager
from .rules_engine import RulesEngine
from .llm_extractor import AzureOpenAILLMExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.email_extractor import EmailExtractor
from .storage import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Main document processing pipeline (LLM extraction focused)."""

    def __init__(self,
                 rules_dir: str = "rules",
                 signatures_dir: str = "signatures",
                 outputs_dir: str = "outputs",
                 use_rules: bool = False):
        """Initialize the pipeline.

        Args:
            rules_dir: Directory containing rule files
            signatures_dir: Directory for signature storage
            outputs_dir: Directory for normalized document outputs
            use_rules: Whether to also run legacy regex rules (future hybrid)
        """
        self.rules_dir = rules_dir
        self.signatures_dir = signatures_dir
        self.outputs_dir = outputs_dir
        self.use_rules = use_rules

        # Ensure directories exist
        os.makedirs(self.rules_dir, exist_ok=True)
        os.makedirs(self.signatures_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Core components
        self.signature_manager = SignatureManager(self.signatures_dir)
        self.rules_engine = RulesEngine(self.rules_dir)
        self.llm_extractor = AzureOpenAILLMExtractor()
        self.repository = DocumentRepository(self.outputs_dir)

        # File extractors
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextExtractor()
        self.email_extractor = EmailExtractor()

        logger.info("Pipeline initialized (LLM mode %s)" % ("+ rules" if self.use_rules else "only"))

    def process_document(self, file_path: str) -> Tuple[NormalizedDocument, str]:
        """Process a single document end-to-end."""
        start_time = time.time()
        log_lines: List[str] = []

        # --- Load file ---
        with open(file_path, 'rb') as f:
            content_bytes = f.read()
        file_size = len(content_bytes)
        filename = os.path.basename(file_path)
        file_type = self._detect_file_type(filename)
        log_lines.append(f"Processing {filename} ({file_type}, {file_size} bytes)")

        # --- Create normalized doc container ---
        document = NormalizedDocument.create(filename, file_size, file_type, content_bytes)

        # --- Content extraction ---
        text, layout_elements = self._extract_content(file_path, file_type)
        log_lines.append(f"Extracted {len(text)} chars from {len(layout_elements)} layout elements")

        # --- Signature processing ---
        signature, similarity = self._process_signature(layout_elements, filename)
        document.processing_meta.signature_id = signature.signature_id
        document.processing_meta.signature_match_score = similarity
        log_lines.append(f"Signature {signature.signature_id} (similarity {similarity:.2f})")

        # --- LLM Extraction (primary) ---
        kvs, llm_meta = self.llm_extractor.extract(text)
        document.key_values.extend(kvs)
        document.processing_meta.model_calls_made += 1 if kvs else 0
        log_lines.append(
            f"LLM fields: {len(kvs)} model={llm_meta.get('model_used')} error={llm_meta.get('error')}"
        )

        # Optional legacy rules (disabled by default)
        if self.use_rules:
            rule_kvs, applied = self._apply_rules(text, signature.signature_id)
            document.key_values.extend(rule_kvs)
            document.processing_meta.rules_applied = applied
            log_lines.append(f"Rules added {len(rule_kvs)} fields: {applied}")
        else:
            document.processing_meta.rules_applied = []

        # --- Sections & chunks ---
        self._finalize_document(document, text, layout_elements, file_type)

        # --- Stats ---
        processing_time = time.time() - start_time
        document.ingest_metadata.processing_time_seconds = processing_time
        document.processing_meta.coverage_stats = calculate_coverage_stats(document)

        save_path = self.repository.save_document(document)
        log_lines.append(f"Saved to {save_path}")
        log_lines.append(f"Completed in {processing_time:.2f}s")
        logger.info(f"Processed document {document.doc_id} in {processing_time:.2f}s")
        return document, "\n".join(log_lines)
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from filename."""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        type_mapping = {
            'pdf': 'pdf',
            'txt': 'text',
            'html': 'html',
            'htm': 'html',
            'eml': 'email',
            'msg': 'email'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    def _extract_content(self, file_path: str, file_type: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text and layout elements based on file type."""
        if file_type == 'pdf':
            text, layout_elements, metadata = self.pdf_extractor.extract_content(file_path)
            return text, layout_elements
        
        elif file_type == 'email':
            text, layout_elements, metadata = self.email_extractor.extract_content(file_path)
            return text, layout_elements
        
        elif file_type in ['text', 'html']:
            text, layout_elements, metadata = self.text_extractor.extract_content(file_path)
            return text, layout_elements
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_signature(self, layout_elements: List[Dict[str, Any]], filename: str) -> Tuple[Any, float]:
        """Process layout signature matching."""
        signature, similarity_score = self.signature_manager.create_or_match_signature(
            layout_elements, filename
        )
        return signature, similarity_score
    
    def _apply_rules(self, text: str, signature_id: str) -> Tuple[List[Any], List[str]]:
        """Apply rule-based extraction."""
        extracted_values, rules_applied = self.rules_engine.apply_rules(text, signature_id)
        return extracted_values, rules_applied
    
    def _finalize_document(self, document: NormalizedDocument, text: str, 
                          layout_elements: List[Dict[str, Any]], file_type: str):
        """Finalize document with sections and chunks."""
        # Create sections based on file type
        if file_type == 'pdf':
            document.sections = self.pdf_extractor.convert_to_sections(layout_elements)
        elif file_type == 'email':
            document.sections = self.email_extractor.convert_to_sections(layout_elements)
        elif file_type in ['text', 'html']:
            document.sections = self.text_extractor.convert_to_sections(layout_elements)
        else:
            # Default single section
            document.sections = [Section(
                title="Document Content",
                content=text,
                level=1
            )]
        
        # Create simple chunks
        document.chunks = self._create_chunks(text, document.doc_id)
        
        # Update page count
        document.ingest_metadata.page_count = max(
            elem.get('page', 1) for elem in layout_elements
        ) if layout_elements else 1
    
    def _create_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        """Create document chunks for potential RAG usage."""
        # Simple chunking by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            chunk = Chunk(
                content=paragraph,
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                start_page=1,
                end_page=1,
                tokens=len(paragraph.split())
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_batch(self, file_paths: List[str]) -> List[Tuple[str, Optional[str]]]:
        """
        Process multiple documents in batch.
        
        Returns:
            List of (doc_id, error_message) tuples
        """
        results = []
        
        for file_path in file_paths:
            try:
                document, log = self.process_document(file_path)
                results.append((document.doc_id, None))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append((None, str(e)))
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "signatures": self.signature_manager.get_signature_stats(),
            "rules": self.rules_engine.get_stats(),
            "repository": self.repository.get_statistics()
        }