"""
Main document processing pipeline.
"""

import os
import time
import logging
import random
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

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Main document processing pipeline (LLM extraction focused)."""

    def __init__(self,
                 rules_dir: str = "rules",
                 signatures_dir: str = "signatures",
                 outputs_dir: str = "outputs",
                 use_rules: bool = False,
                 enable_llm: bool = True,
                 enable_di: bool = True,
                 high_conf_threshold: float = 0.85,
                 min_required_coverage: float = 0.75,
                 borderline_sample_rate: float = 0.30):
        """Initialize the pipeline.

        Args:
            rules_dir: Directory containing rule files
            signatures_dir: Directory for signature storage
            outputs_dir: Directory for normalized document outputs
            use_rules: Whether to also run legacy regex rules (future hybrid)
            enable_llm: Whether to enable Azure OpenAI LLM extraction
            enable_di: Whether to enable Azure Document Intelligence fallback
            high_conf_threshold: Confidence threshold for skipping LLM (0.85)
            min_required_coverage: Minimum coverage for required fields (0.75)
            borderline_sample_rate: Sampling rate for borderline cases (0.30)
        """
        self.rules_dir = rules_dir
        self.signatures_dir = signatures_dir
        self.outputs_dir = outputs_dir
        self.use_rules = use_rules
        self.enable_llm = enable_llm
        self.enable_di = enable_di
        self.high_conf_threshold = high_conf_threshold
        self.min_required_coverage = min_required_coverage
        self.borderline_sample_rate = borderline_sample_rate

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

        # File extractors
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextExtractor()
        self.email_extractor = EmailExtractor()

        logger.info("Pipeline initialized (LLM mode %s)" % ("+ rules" if self.use_rules else "only"))
        
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
        """Process a single document end-to-end with hybrid gating logic."""
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

        # --- Rules FIRST (always run for gating) ---
        rule_kvs, applied = self._apply_rules(text, signature.signature_id)
        document.key_values.extend(rule_kvs)
        document.processing_meta.rules_applied = applied
        log_lines.append(f"Rules extracted {len(rule_kvs)} fields: {applied}")

        # --- Compute required fields and confidence ---
        required_fields = self.rules_engine.get_required_fields(signature.signature_id)
        document.processing_meta.required_fields_total = len(required_fields)
        
        # Calculate per-field dynamic confidence and document metrics
        confidence_breakdown, doc_confidence, coverage_ratio, required_present = self._calculate_confidence_metrics(
            document.key_values, required_fields, similarity
        )
        
        document.processing_meta.confidence_breakdown = confidence_breakdown
        document.processing_meta.document_confidence = doc_confidence
        document.processing_meta.coverage_ratio = coverage_ratio
        document.processing_meta.required_fields_present = required_present
        
        log_lines.append(f"Coverage: {required_present}/{len(required_fields)} required fields ({coverage_ratio:.2f})")
        log_lines.append(f"Document confidence: {doc_confidence:.2f}")

        # --- Gating Decision Logic ---
        gating_decision = self._make_gating_decision(
            required_fields, coverage_ratio, doc_confidence, required_present
        )
        document.processing_meta.gating_decision = gating_decision
        log_lines.append(f"Gating decision: {gating_decision}")

        # --- Conditional LLM Extraction based on gating ---
        llm_called = False
        if gating_decision.startswith("call_llm") and self.enable_llm:
            if self.llm_extractor.enabled:
                kvs, llm_meta = self.llm_extractor.extract(text)
                if kvs:
                    # Merge LLM results with rule results
                    self._merge_extraction_results(document.key_values, kvs)
                    llm_called = True
                    document.processing_meta.model_calls_made += 1
                log_lines.append(
                    f"LLM called: {len(kvs)} fields model={llm_meta.get('model_used')} error={llm_meta.get('error')}"
                )
            else:
                log_lines.append("LLM needed but Azure env not configured – skipping")
        elif gating_decision.startswith("call_llm"):
            log_lines.append("LLM needed but disabled by user toggle")
        else:
            log_lines.append(f"LLM skipped due to gating: {gating_decision}")

        document.processing_meta.ai_used = llm_called

        # --- Azure Document Intelligence Fallback (conditional) ---
        di_called = False
        # DI as secondary fallback: after gating failed/empty LLM results (PDF only) OR insufficient coverage
        should_call_di = (
            file_type == 'pdf' and 
            self.enable_di and 
            self.di_extractor.enabled and
            (
                (llm_called and len([kv for kv in document.key_values if kv.extraction_method == "model"]) == 0) or
                (coverage_ratio < self.min_required_coverage and not llm_called)
            )
        )
        
        if should_call_di:
            di_kvs, di_meta = self.di_extractor.extract(file_path)
            if di_kvs:
                self._merge_extraction_results(document.key_values, di_kvs)
                di_called = True
                log_lines.append(f"DI fallback: {len(di_kvs)} fields model={di_meta.get('model_used')}")
            else:
                log_lines.append(f"DI fallback produced 0 fields error={di_meta.get('error')}")
        elif file_type == 'pdf' and self.enable_di and not self.di_extractor.enabled:
            log_lines.append("DI needed but Azure env not configured")
        else:
            log_lines.append("DI not triggered")

        document.processing_meta.di_used = di_called

        # --- Recalculate final metrics after LLM/DI ---
        if llm_called or di_called:
            confidence_breakdown, doc_confidence, coverage_ratio, required_present = self._calculate_confidence_metrics(
                document.key_values, required_fields, similarity
            )
            document.processing_meta.confidence_breakdown = confidence_breakdown
            document.processing_meta.document_confidence = doc_confidence
            document.processing_meta.coverage_ratio = coverage_ratio
            document.processing_meta.required_fields_present = required_present
            log_lines.append(f"Final metrics - Coverage: {required_present}/{len(required_fields)} ({coverage_ratio:.2f}), Confidence: {doc_confidence:.2f}")

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
    
    def _calculate_confidence_metrics(self, key_values: List[Any], required_fields: List[str], 
                                    signature_similarity: float) -> Tuple[Dict[str, Any], float, float, int]:
        """Calculate per-field confidence and document-level metrics."""
        confidence_breakdown = {}
        required_present = 0
        total_confidence = 0.0
        total_fields = 0
        
        # Create a lookup for extracted fields
        extracted_fields = {kv.key: kv for kv in key_values}
        
        # Process each extracted field
        for kv in key_values:
            # Start with rule confidence
            dynamic_confidence = kv.confidence
            
            # Optional heuristic booster: high signature similarity adds confidence
            if signature_similarity >= 0.9:
                dynamic_confidence = min(0.99, dynamic_confidence + 0.05)
            
            confidence_breakdown[kv.key] = {
                "static_confidence": kv.confidence,
                "dynamic_confidence": dynamic_confidence,
                "extraction_method": kv.extraction_method,
                "is_required": kv.key in required_fields
            }
            
            # Update kv confidence with dynamic value
            kv.confidence = dynamic_confidence
            
            # Count for document confidence
            total_confidence += dynamic_confidence
            total_fields += 1
            
            # Count required fields present
            if kv.key in required_fields:
                required_present += 1
        
        # Calculate coverage ratio
        coverage_ratio = required_present / len(required_fields) if required_fields else 1.0
        
        # Calculate document confidence
        if len(required_fields) > 0:
            # Average confidence of required fields present (0 if none present)
            required_field_confidences = [
                confidence_breakdown[field]["dynamic_confidence"] 
                for field in required_fields if field in extracted_fields
            ]
            doc_confidence = sum(required_field_confidences) / len(required_field_confidences) if required_field_confidences else 0.0
        else:
            # If no required fields, use average of all extracted fields (or 0)
            doc_confidence = total_confidence / total_fields if total_fields > 0 else 0.0
        
        return confidence_breakdown, doc_confidence, coverage_ratio, required_present
    
    def _make_gating_decision(self, required_fields: List[str], coverage_ratio: float, 
                            doc_confidence: float, required_present: int) -> str:
        """Make gating decision based on confidence and coverage metrics."""
        required_total = len(required_fields)
        
        # If required fields exist and all are present with high confidence
        if required_total > 0 and coverage_ratio == 1.0 and doc_confidence >= self.high_conf_threshold:
            return "skip_llm_high_conf"
        
        # If required fields exist but coverage is below minimum threshold
        elif required_total > 0 and coverage_ratio < self.min_required_coverage:
            return "call_llm_low_coverage"
        
        # If any required field is missing
        elif required_total > 0 and required_present < required_total:
            return "call_llm_missing_required"
        
        # Borderline case: sample based on rate
        else:
            if random.random() < self.borderline_sample_rate:
                return "call_llm_borderline_sampled"
            else:
                return "skip_llm_borderline_not_sampled"
    
    def _merge_extraction_results(self, existing_kvs: List[Any], new_kvs: List[Any]):
        """Merge new extraction results with existing ones, avoiding duplicates."""
        existing_keys = {kv.key for kv in existing_kvs}
        
        for new_kv in new_kvs:
            if new_kv.key not in existing_keys:
                existing_kvs.append(new_kv)
            else:
                # Optionally update with higher confidence value
                for i, existing_kv in enumerate(existing_kvs):
                    if existing_kv.key == new_kv.key and new_kv.confidence > existing_kv.confidence:
                        existing_kvs[i] = new_kv
                        break
    
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