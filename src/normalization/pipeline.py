"""
Main document processing pipeline.
"""

import os
import re
import random
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .schema import NormalizedDocument, Section, Chunk, KeyValue, calculate_coverage_stats
from .signatures import SignatureManager
from .rules_engine import RulesEngine
from .llm_extractor import AzureOpenAILLMExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.email_extractor import EmailExtractor
from .storage import DocumentRepository

logger = logging.getLogger(__name__)

# Adaptive hybrid parsing configuration constants
MIN_CONF_THRESHOLD = 0.75
LOW_COVERAGE_THRESHOLD = 0.5
BORDERLINE_RANGE = (0.5, 0.9)
BORDERLINE_SAMPLING_RATE = 0.30
RULE_LEARNING_ENABLED = True

# Gating decision constants
GATING_DECISIONS = {
    'SKIP_LLM_HIGH_CONF': 'SKIP_LLM_HIGH_CONF',
    'USED_LLM_LOW_COVERAGE': 'USED_LLM_LOW_COVERAGE', 
    'USED_LLM_BOOTSTRAP': 'USED_LLM_BOOTSTRAP',
    'USED_LLM_BORDERLINE_SAMPLED': 'USED_LLM_BORDERLINE_SAMPLED',
    'SKIP_LLM_BORDERLINE_NOT_SAMPLED': 'SKIP_LLM_BORDERLINE_NOT_SAMPLED'
}


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
        """Process a single document end-to-end with adaptive hybrid parsing."""
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

        # --- Rules-first extraction (hybrid approach) ---
        rule_kvs, rules_applied = self._apply_rules(text, signature.signature_id)
        document.key_values.extend(rule_kvs)
        document.processing_meta.rules_applied = rules_applied
        log_lines.append(f"Rules extracted {len(rule_kvs)} fields from {rules_applied}")

        # --- Compute required fields and coverage ---
        required_fields = self.rules_engine.get_required_fields(signature.signature_id)
        document.processing_meta.required_fields = required_fields
        
        coverage_ratio, avg_rule_confidence = self._compute_coverage_and_confidence(document.key_values, required_fields)
        log_lines.append(f"Coverage: {coverage_ratio:.2f} ({len([kv for kv in document.key_values if kv.key in required_fields])}/{len(required_fields)} required fields), Avg confidence: {avg_rule_confidence:.2f}")

        # --- Gating logic for LLM invocation ---
        gating_decision, should_invoke_llm, rationale = self._make_gating_decision(
            coverage_ratio, avg_rule_confidence, signature.signature_id, required_fields, document.key_values
        )
        document.processing_meta.gating_decision = gating_decision
        document.processing_meta.llm_invoked = should_invoke_llm
        log_lines.append(f"Gating decision: {gating_decision} - {rationale}")

        # --- Conditional LLM extraction ---
        if should_invoke_llm:
            llm_kvs, llm_meta = self.llm_extractor.extract(text)
            document.processing_meta.model_calls_made += 1 if llm_kvs else 0
            
            # Merge LLM key-values (avoid duplicates; rule values win unless empty)
            merged_kvs = self._merge_key_values(document.key_values, llm_kvs)
            document.key_values = merged_kvs
            
            log_lines.append(f"LLM extracted {len(llm_kvs)} fields, merged to {len(document.key_values)} total")
            
            # Learn signature-specific rules if enabled
            if RULE_LEARNING_ENABLED and llm_kvs:
                self._learn_signature_rules(signature.signature_id, llm_kvs, text, log_lines)
        else:
            log_lines.append("LLM invocation skipped by gating logic")

        # --- Compute final confidences ---
        self._compute_final_confidences(document.key_values)
        document.processing_meta.document_confidence = self._compute_document_confidence(document.key_values, required_fields)
        log_lines.append(f"Document confidence: {document.processing_meta.document_confidence:.2f}")

        # --- Legacy compatibility for existing functionality ---
        if not self.use_rules:
            # If rules are disabled, ensure we still call LLM for backward compatibility
            if not should_invoke_llm:
                llm_kvs, llm_meta = self.llm_extractor.extract(text)
                document.key_values.extend(llm_kvs)
                document.processing_meta.model_calls_made += 1 if llm_kvs else 0
                document.processing_meta.llm_invoked = True
                log_lines.append(f"Legacy LLM call: {len(llm_kvs)} fields extracted")

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

    def _compute_coverage_and_confidence(self, key_values: List[KeyValue], required_fields: List[str]) -> Tuple[float, float]:
        """Compute coverage ratio and average rule confidence."""
        if not required_fields:
            return 1.0, 1.0  # No required fields means full coverage
            
        extracted_required = [kv for kv in key_values if kv.key in required_fields]
        coverage_ratio = len(extracted_required) / len(required_fields)
        
        # Compute average confidence of rule-based extractions
        rule_confidences = [kv.confidence for kv in key_values if kv.extraction_method == "rule"]
        avg_rule_confidence = sum(rule_confidences) / len(rule_confidences) if rule_confidences else 0.0
        
        return coverage_ratio, avg_rule_confidence

    def _make_gating_decision(self, coverage_ratio: float, avg_confidence: float, 
                             signature_id: str, required_fields: List[str], 
                             extracted_kvs: List[KeyValue]) -> Tuple[str, bool, str]:
        """Make gating decision for LLM invocation."""
        
        # Check if this is a new signature (bootstrap case)
        signature_rules = self.rules_engine.signature_rules.get(signature_id, [])
        if not signature_rules:
            return (GATING_DECISIONS['USED_LLM_BOOTSTRAP'], True, 
                   "New signature with no specific rules - bootstrapping with LLM")
        
        # High confidence and full coverage - skip LLM
        if coverage_ratio >= 1.0 and avg_confidence >= MIN_CONF_THRESHOLD:
            return (GATING_DECISIONS['SKIP_LLM_HIGH_CONF'], False,
                   f"High coverage ({coverage_ratio:.2f}) and confidence ({avg_confidence:.2f}) - skipping LLM")
        
        # Low coverage or missing critical fields - use LLM
        if coverage_ratio < LOW_COVERAGE_THRESHOLD:
            return (GATING_DECISIONS['USED_LLM_LOW_COVERAGE'], True,
                   f"Low coverage ({coverage_ratio:.2f}) below threshold ({LOW_COVERAGE_THRESHOLD}) - using LLM")
        
        # Borderline range - probabilistic sampling
        if BORDERLINE_RANGE[0] <= coverage_ratio < BORDERLINE_RANGE[1]:
            if random.random() < BORDERLINE_SAMPLING_RATE:
                return (GATING_DECISIONS['USED_LLM_BORDERLINE_SAMPLED'], True,
                       f"Borderline coverage ({coverage_ratio:.2f}) - sampled for LLM ({BORDERLINE_SAMPLING_RATE:.0%} rate)")
            else:
                return (GATING_DECISIONS['SKIP_LLM_BORDERLINE_NOT_SAMPLED'], False,
                       f"Borderline coverage ({coverage_ratio:.2f}) - not sampled for LLM")
        
        # Default to skipping LLM
        return (GATING_DECISIONS['SKIP_LLM_HIGH_CONF'], False,
               f"Coverage ({coverage_ratio:.2f}) and confidence ({avg_confidence:.2f}) sufficient - skipping LLM")

    def _merge_key_values(self, rule_kvs: List[KeyValue], llm_kvs: List[KeyValue]) -> List[KeyValue]:
        """Merge rule and LLM key-values, with rule values taking precedence."""
        merged = {}
        
        # Add rule-based values first (these take precedence)
        for kv in rule_kvs:
            merged[kv.key] = kv
        
        # Add LLM values only if not already present or if rule value is empty
        for kv in llm_kvs:
            if kv.key not in merged or not merged[kv.key].value:
                merged[kv.key] = kv
            elif kv.key in merged and merged[kv.key].extraction_method == "rule":
                # If LLM extracted the same value as a high-confidence rule, reduce LLM confidence
                if str(merged[kv.key].value).lower() == str(kv.value).lower() and merged[kv.key].confidence > 0.8:
                    kv.confidence = 0.40  # Reduce confidence for duplicate value
        
        return list(merged.values())

    def _learn_signature_rules(self, signature_id: str, llm_kvs: List[KeyValue], 
                              text: str, log_lines: List[str]):
        """Learn new signature-specific rules from successful LLM extractions."""
        try:
            learned_count = self.rules_engine.learn_signature_rules(signature_id, llm_kvs, text)
            if learned_count > 0:
                log_lines.append(f"Learned {learned_count} new rules for signature {signature_id}")
        except Exception as e:
            logger.warning(f"Failed to learn rules for signature {signature_id}: {e}")

    def _compute_final_confidences(self, key_values: List[KeyValue]):
        """Compute final confidence scores with heuristic adjustments."""
        for kv in key_values:
            if kv.extraction_method == "rule":
                # Rule confidence adjustments
                base_confidence = kv.confidence
                
                # +0.05 if value length > 5 (bounded)
                if len(str(kv.value)) > 5:
                    base_confidence += 0.05
                
                # -0.05 if pattern contains alternation (check if this was from a pattern with |)
                # Note: This would require storing pattern info, simplified for now
                
                # Clamp to [0.0, 0.99]
                kv.confidence = max(0.0, min(0.99, base_confidence))
                
            elif kv.extraction_method == "model":
                # Model confidence defaults to existing or 0.70
                if kv.confidence == 0.0:
                    kv.confidence = 0.70

    def _compute_document_confidence(self, key_values: List[KeyValue], required_fields: List[str]) -> float:
        """Compute aggregate document confidence as weighted average."""
        if not key_values:
            return 0.0
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for kv in key_values:
            # Required fields get weight 1.5, optional fields get weight 1.0
            weight = 1.5 if kv.key in required_fields else 1.0
            total_weighted_confidence += kv.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
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