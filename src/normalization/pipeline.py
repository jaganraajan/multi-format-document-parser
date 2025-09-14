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
from .azure_di_extractor import AzureDocumentIntelligenceExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.email_extractor import EmailExtractor
from .storage import DocumentRepository
from .usage_tracker import UsageTracker

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Main document processing pipeline (LLM extraction focused with signature reuse)."""

    def __init__(self,
                 rules_dir: str = "rules",
                 signatures_dir: str = "signatures",
                 outputs_dir: str = "outputs",
                 use_rules: bool = False,
                 enable_llm: bool = True,
                 enable_di: bool = True,
                 auto_enable_rules_local: bool = True,
                 signature_reuse_threshold: float = 0.90,
                 signature_cache_min_fields: int = 3,
                 enable_signature_reuse: bool = True):
        """Initialize the pipeline.

        Args:
            rules_dir: Directory containing rule files
            signatures_dir: Directory for signature storage
            outputs_dir: Directory for normalized document outputs
            use_rules: Whether to also run legacy regex rules (future hybrid)
            enable_llm: Whether to enable Azure OpenAI LLM extraction
            enable_di: Whether to enable Azure Document Intelligence fallback
            auto_enable_rules_local: Auto turn on rules when both external methods disabled
            signature_reuse_threshold: Similarity required to reuse cached fields
            signature_cache_min_fields: Minimum fields before caching a signature
            enable_signature_reuse: Master toggle for signature-based gating
        """
        self.rules_dir = rules_dir
        self.signatures_dir = signatures_dir
        self.outputs_dir = outputs_dir
        self.use_rules = use_rules
        self.enable_llm = enable_llm
        self.enable_di = enable_di
        self.auto_rules_enabled = False
        self.signature_reuse_threshold = signature_reuse_threshold
        self.signature_cache_min_fields = signature_cache_min_fields
        self.enable_signature_reuse = enable_signature_reuse
        # In-memory field cache for current process/session (signature_id -> {key: {value, confidence}})
        self._signature_field_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Auto-enable rules for pure local mode OR when rules exist and not explicitly disabled
        global_rules_path = os.path.join(self.rules_dir, 'global_rules.yml')
        if (
            (not self.enable_llm and not self.enable_di and not self.use_rules and auto_enable_rules_local)
            or (os.path.exists(global_rules_path) and not self.use_rules and os.getenv('DISABLE_AUTO_RULES', 'false').lower() != 'true')
        ):
            self.use_rules = True
            self.auto_rules_enabled = True

        # Ensure directories exist
        os.makedirs(self.rules_dir, exist_ok=True)
        os.makedirs(self.signatures_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Core components
        self.signature_manager = SignatureManager(self.signatures_dir)
        self.rules_engine = RulesEngine(self.rules_dir)
        self.llm_extractor = AzureOpenAILLMExtractor()
        self.di_extractor = AzureDocumentIntelligenceExtractor()
        self.repository = DocumentRepository(self.outputs_dir)
        self.usage_tracker = UsageTracker()

        # File extractors
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextExtractor()
        self.email_extractor = EmailExtractor()

        logger.info("Pipeline initialized (LLM mode %s)" % ("+ rules" if self.use_rules else "only"))
        if self.auto_rules_enabled:
            logger.info("Rules auto-enabled for pure local mode (LLM & DI both disabled)")
        if self.enable_signature_reuse:
            logger.info(
                "Signature reuse enabled (threshold=%.2f, min_fields=%d)",
                self.signature_reuse_threshold,
                self.signature_cache_min_fields,
            )

        if self.enable_llm:
            logger.info("LLM extraction enabled (user toggle: ON)")
        else:
            logger.info("LLM extraction disabled (user toggle: OFF)")

        if self.di_extractor.enabled and self.enable_di:
            logger.info("Azure Document Intelligence fallback enabled (model %s)", self.di_extractor.model_id)
        elif self.enable_di and not self.di_extractor.enabled:
            logger.info("Azure Document Intelligence selected but env vars missing")
        else:
            logger.info("Azure Document Intelligence fallback disabled (user toggle: OFF)")

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
        text, layout_elements, extraction_metadata = self._extract_content(file_path, file_type)
        log_lines.append(f"Extracted {len(text)} chars from {len(layout_elements)} layout elements")

        # --- Signature processing ---
        signature, similarity = self._process_signature(layout_elements, filename)
        document.processing_meta.signature_id = signature.signature_id
        document.processing_meta.signature_match_score = similarity
        version_event = getattr(signature, '_last_version_event', None)
        family_part = f" family={signature.family_id}" if getattr(signature, 'family_id', None) else ''
        version_part = f" v{getattr(signature, 'active_version', 1)}" if hasattr(signature, 'active_version') else ''
        event_part = f" event={version_event}" if version_event else ''
        log_lines.append(
            f"Signature {signature.signature_id}{family_part}{version_part} (similarity {similarity:.2f}){event_part}"
        )

        # --- Signature-based reuse gating ---
        if (
            self.enable_signature_reuse
            and similarity >= self.signature_reuse_threshold
            and getattr(signature, 'cached_fields', None)
        ):
            from .schema import KeyValue
            reused = [
                KeyValue(
                    key=k,
                    value=meta.get('value'),
                    confidence=meta.get('confidence', 0.75),
                    extraction_method='signature_cache'
                )
                for k, meta in signature.cached_fields.items()  # type: ignore
            ]
            document.key_values.extend(reused)
            log_lines.append(
                f"Signature reuse hit: similarity {similarity:.2f} ≥ {self.signature_reuse_threshold:.2f}; reused {len(reused)} cached fields and skipped model/rules"
            )
            self._finalize_document(document, text, layout_elements, file_type)
            processing_time = time.time() - start_time
            document.ingest_metadata.processing_time_seconds = processing_time
            document.processing_meta.coverage_stats = calculate_coverage_stats(document)
            
            # Track document processing completion (signature cache path)
            self.usage_tracker.record_document_time(processing_time)
            
            save_path = self.repository.save_document(document)
            log_lines.append(f"Saved to {save_path}")
            log_lines.append(f"Completed in {processing_time:.2f}s (signature cache)")
            logger.info(
                "Processed document %s via signature cache in %.2fs", document.doc_id, processing_time
            )
            return document, "\n".join(log_lines)
        else:
            if self.enable_signature_reuse:
                if similarity < self.signature_reuse_threshold:
                    log_lines.append(
                        f"Signature reuse skipped: similarity {similarity:.2f} < threshold {self.signature_reuse_threshold:.2f}"
                    )
                elif not getattr(signature, 'cached_fields', None):
                    log_lines.append(
                        "Signature reuse skipped: no cached_fields present on signature (first qualifying run will create cache)"
                    )

        # --- Rules first ---
        rule_kvs: List[Any] = []
        if self.use_rules:
            rule_kvs, applied = self._apply_rules(text, signature.signature_id)
            document.key_values.extend(rule_kvs)
            document.processing_meta.rules_applied = applied
            self.usage_tracker.record_rules_hit(len(rule_kvs))
            # Interim coverage stats after rules only
            try:
                interim_cov = calculate_coverage_stats(document)
                document.processing_meta.coverage_stats = interim_cov
                rule_cov_pct = interim_cov.get('rule_coverage', 0.0) * 100
                log_lines.append(f"Rules added {len(rule_kvs)} fields (rule coverage {rule_cov_pct:.1f}%)")
            except Exception as e:
                log_lines.append(f"Rule coverage calculation failed: {e}")
        else:
            document.processing_meta.rules_applied = []

        # --- Session cache reuse (if no persisted cached_fields yet) ---
        session_cache_hit = False
        if (
            self.enable_signature_reuse
            and similarity >= self.signature_reuse_threshold
            and not getattr(signature, 'cached_fields', None)
        ):
            cached_session = self._signature_field_cache.get(signature.signature_id)
            if cached_session:
                from .schema import KeyValue
                reused = [
                    KeyValue(key=k, value=v.get('value'), confidence=v.get('confidence', 0.7), extraction_method='session_cache')
                    for k, v in cached_session.items()
                ]
                # avoid duplicates
                existing_keys = {kv.key for kv in document.key_values}
                reused = [kv for kv in reused if kv.key not in existing_keys]
                if reused:
                    document.key_values.extend(reused)
                    session_cache_hit = True
                    log_lines.append(
                        f"Session cache reuse hit: similarity {similarity:.2f} reused {len(reused)} fields (no disk cached_fields yet)"
                    )

        # --- LLM Extraction (gated) ---
        kvs: List[Any] = []
        if not session_cache_hit:  # only consider LLM if we didn't reuse from session
            if self.enable_llm and self.llm_extractor.enabled:
                # Gate on confidence of existing fields (rules)
                confidences = [kv.confidence for kv in document.key_values if getattr(kv, 'confidence', None) is not None]
                avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
                need_llm = (len(document.key_values) == 0) or (avg_conf < 0.70)
                if need_llm:
                    kvs, llm_meta = self.llm_extractor.extract(text)
                    document.key_values.extend(kvs)
                    document.processing_meta.model_calls_made += 1 if kvs else 0
                    if kvs or llm_meta.get('error') != 'llm_disabled':
                        self.usage_tracker.record_llm_call(llm_meta)
                    log_lines.append(
                        f"LLM fields: {len(kvs)} avg_conf_before={avg_conf:.2f} model={llm_meta.get('model_used')}"
                    )
                    # Assign family ID if possible
                    if kvs:
                        vendor_value = None
                        for kv in kvs:
                            if kv.key in ("vendor_name", "supplier_name") and isinstance(kv.value, str):
                                vendor_value = kv.value
                                break
                        if vendor_value:
                            try:
                                self.signature_manager.update_family_id(signature, vendor_value)
                            except Exception:
                                pass
                else:
                    log_lines.append(f"LLM skipped avg_conf={avg_conf:.2f} >= 0.70")
            elif self.enable_llm and not self.llm_extractor.enabled:
                kvs = []
                log_lines.append("LLM selected but Azure env not configured – skipping")
            else:
                kvs = []
                log_lines.append("LLM disabled by user toggle")
        else:
            log_lines.append("LLM skipped due to session cache reuse")

        # --- Azure Document Intelligence Fallback (PDF only) ---
        di_triggered = False
        if file_type == 'pdf' and len(kvs) == 0:
            if self.enable_di and self.di_extractor.enabled:
                di_kvs, di_meta = self.di_extractor.extract(file_path)
                if di_kvs:
                    document.key_values.extend(di_kvs)
                    di_triggered = True
                    log_lines.append(f"DI fallback fields: {len(di_kvs)} model={di_meta.get('model_used')}")
                    
                    # Track DI usage with page count
                    page_count = extraction_metadata.get('page_count', 1)
                    self.usage_tracker.record_di_call(page_count)
                else:
                    log_lines.append(f"DI fallback produced 0 fields error={di_meta.get('error')}")
            elif self.enable_di and not self.di_extractor.enabled:
                log_lines.append("DI selected but Azure env not configured – skipping")
            else:
                log_lines.append("DI disabled by user toggle")

        # Populate session cache after extraction if we have fields and none persisted yet
        if (
            self.enable_signature_reuse
            and len(document.key_values) >= self.signature_cache_min_fields
            and not getattr(signature, 'cached_fields', None)
        ):
            try:
                cache_payload: Dict[str, Dict[str, Any]] = {}
                for kv in document.key_values:
                    if kv.key not in cache_payload:
                        cache_payload[kv.key] = {"value": kv.value, "confidence": getattr(kv, 'confidence', None)}
                self._signature_field_cache[signature.signature_id] = cache_payload
                log_lines.append(f"Session cache stored {len(cache_payload)} fields for signature {signature.signature_id}")
            except Exception as e:
                log_lines.append(f"Session cache store failed: {e}")

        # --- Sections & chunks ---
        self._finalize_document(document, text, layout_elements, file_type)

        # --- Cache update (post-extraction) ---
        try:
            if (
                self.enable_signature_reuse
                and len(document.key_values) >= self.signature_cache_min_fields
                and not any(kv.extraction_method == 'signature_cache' for kv in document.key_values)
            ):
                self.signature_manager.update_signature_cache(signature, document.key_values)
        except Exception as e:
            log_lines.append(f"Cache update failed: {e}")

        # --- Stats ---
        processing_time = time.time() - start_time
        document.ingest_metadata.processing_time_seconds = processing_time
        document.processing_meta.coverage_stats = calculate_coverage_stats(document)
        
        # Track document processing completion
        self.usage_tracker.record_document_time(processing_time)

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
    
    def _extract_content(self, file_path: str, file_type: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Extract text and layout elements based on file type."""
        if file_type == 'pdf':
            text, layout_elements, metadata = self.pdf_extractor.extract_content(file_path)
            return text, layout_elements, metadata
        
        elif file_type == 'email':
            text, layout_elements, metadata = self.email_extractor.extract_content(file_path)
            return text, layout_elements, metadata
        
        elif file_type in ['text', 'html']:
            text, layout_elements, metadata = self.text_extractor.extract_content(file_path)
            return text, layout_elements, metadata
        
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
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage statistics and cost estimates.
        
        Returns:
            Dictionary with usage stats, averages, and cost breakdown
        """
        return self.usage_tracker.snapshot()