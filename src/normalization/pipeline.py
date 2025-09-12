"""
Main document processing pipeline.
"""

import os
import time
import yaml
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .schema import NormalizedDocument, Section, Chunk, calculate_coverage_stats
from .signatures import SignatureManager
from .rules_engine import RulesEngine
from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.email_extractor import EmailExtractor
from .storage import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Main document processing pipeline."""
    
    def __init__(self, 
                 rules_dir: str = "rules",
                 signatures_dir: str = "signatures", 
                 outputs_dir: str = "outputs"):
        """
        Initialize the document pipeline.
        
        Args:
            rules_dir: Directory containing rule files
            signatures_dir: Directory for signature storage
            outputs_dir: Directory for normalized document outputs
        """
        self.rules_dir = rules_dir
        self.signatures_dir = signatures_dir
        self.outputs_dir = outputs_dir
        
        # Ensure directories exist
        os.makedirs(rules_dir, exist_ok=True)
        os.makedirs(signatures_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Initialize components
        self.signature_manager = SignatureManager(signatures_dir)
        self.rules_engine = RulesEngine(rules_dir)
        self.repository = DocumentRepository(outputs_dir)
        
        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextExtractor()
        self.email_extractor = EmailExtractor()
        
        # Initialize optional LLM extractor
        self.llm_extractor = None
        self._load_llm_config()
        
        logger.info("Pipeline initialized successfully")
    
    def _load_llm_config(self):
        """Load LLM configuration if available."""
        llm_config_path = "llm_config.yml"
        
        if os.path.exists(llm_config_path):
            try:
                with open(llm_config_path, 'r') as f:
                    llm_config = yaml.safe_load(f)
                
                # Try to initialize LLM extractor
                from .llm import LLMFieldExtractor
                self.llm_extractor = LLMFieldExtractor(llm_config)
                
                if self.llm_extractor.is_available:
                    logger.info("LLM gap-filling enabled")
                else:
                    logger.warning("LLM config found but LLM not available (check API key and dependencies)")
                    
            except ImportError:
                logger.warning("LLM config found but openai package not installed. Install with: pip install openai>=1.0.0")
            except Exception as e:
                logger.warning(f"Error loading LLM config: {e}")
        else:
            logger.debug("No llm_config.yml found, LLM gap-filling disabled")
    
    def process_document(self, file_path: str) -> Tuple[NormalizedDocument, str]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (normalized_document, processing_log)
        """
        start_time = time.time()
        processing_log = []
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            file_size = len(file_content)
            filename = os.path.basename(file_path)
            file_type = self._detect_file_type(filename)
            
            processing_log.append(f"Processing {filename} ({file_type}, {file_size} bytes)")
            
            # Create normalized document
            document = NormalizedDocument.create(filename, file_size, file_type, file_content)
            
            # Stage 1: Extract content
            text, layout_elements = self._extract_content(file_path, file_type)
            processing_log.append(f"Extracted {len(text)} characters from {len(layout_elements)} layout elements")
            
            # Stage 2: Process signature
            signature, similarity_score = self._process_signature(layout_elements, filename)
            document.processing_meta.signature_id = signature.signature_id
            document.processing_meta.signature_match_score = similarity_score
            processing_log.append(f"Signature: {signature.signature_id} (similarity: {similarity_score:.2f})")
            
            # Stage 3: Apply rules
            extracted_values, rules_applied = self._apply_rules(text, signature.signature_id)
            document.key_values.extend(extracted_values)
            document.processing_meta.rules_applied = rules_applied
            processing_log.append(f"Extracted {len(extracted_values)} fields using rules: {rules_applied}")
            
            # Stage 4: LLM gap-filling (if enabled and needed)
            llm_extracted_count = 0
            if self.llm_extractor and self.llm_extractor.is_available:
                llm_extracted_count = self._apply_llm_gap_filling(
                    document, text, signature.signature_id, processing_log
                )
            
            # Stage 5: Create sections and chunks
            self._finalize_document(document, text, layout_elements, file_type)
            
            # Calculate final stats
            processing_time = time.time() - start_time
            document.ingest_metadata.processing_time_seconds = processing_time
            coverage_stats = calculate_coverage_stats(document)
            document.processing_meta.coverage_stats = coverage_stats
            
            # Save document
            save_path = self.repository.save_document(document)
            
            processing_log.append(f"Processing completed in {processing_time:.2f}s")
            processing_log.append(f"Saved to: {save_path}")
            
            logger.info(f"Processed document {document.doc_id} successfully")
            
            return document, "\n".join(processing_log)
            
        except Exception as e:
            error_msg = f"Error processing document: {e}"
            processing_log.append(error_msg)
            logger.error(error_msg)
            raise
    
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
    
    def _apply_llm_gap_filling(self, document: NormalizedDocument, text: str, 
                              signature_id: str, processing_log: List[str]) -> int:
        """
        Apply LLM gap-filling for missing required fields.
        
        Args:
            document: Document being processed
            text: Document text content
            signature_id: Document signature ID
            processing_log: Processing log to append to
            
        Returns:
            Number of fields extracted by LLM
        """
        try:
            # Get required fields for this signature
            required_fields = self.rules_engine.get_required_fields(signature_id)
            
            if not required_fields:
                processing_log.append("LLM: No required fields defined, skipping")
                return 0
            
            # Find missing required fields
            extracted_field_names = {kv.key for kv in document.key_values}
            missing_fields = [field for field in required_fields if field not in extracted_field_names]
            
            if not missing_fields:
                processing_log.append(f"LLM: All {len(required_fields)} required fields already extracted")
                return 0
            
            # Check limits
            max_calls = getattr(self.llm_extractor.config, 'max_model_calls_per_doc', 1)
            max_cost = getattr(self.llm_extractor.config, 'max_total_cost_usd_per_doc', 0.05)
            
            if document.processing_meta.model_calls_made >= max_calls:
                processing_log.append(f"LLM: Max model calls ({max_calls}) reached, skipping")
                return 0
            
            if document.processing_meta.total_cost_usd >= max_cost:
                processing_log.append(f"LLM: Max cost (${max_cost}) reached, skipping")
                return 0
            
            processing_log.append(f"LLM: Attempting to extract {len(missing_fields)} missing fields: {missing_fields}")
            
            # Extract missing fields using LLM
            result = self.llm_extractor.extract_missing_fields(
                document_content=text,
                missing_fields=missing_fields,
                signature_id=signature_id,
                content_hash=document.ingest_metadata.content_hash
            )
            
            # Process results
            extracted_fields = result.get('extracted_fields', [])
            estimated_cost = result.get('cost_estimated', 0.0)
            tokens_used = result.get('tokens_used', 0)
            cached = result.get('cached', False)
            
            # Add extracted fields to document (avoid duplicates)
            new_fields_added = 0
            for field in extracted_fields:
                if field.key not in extracted_field_names:
                    document.key_values.append(field)
                    new_fields_added += 1
            
            # Update processing metadata
            document.processing_meta.model_calls_made += 1
            document.processing_meta.total_cost_usd += estimated_cost
            
            # Add call details
            call_detail = {
                "missing_requested": missing_fields,
                "model_added": [f.key for f in extracted_fields],
                "cost_estimated": estimated_cost,
                "tokens_used": tokens_used,
                "cached": cached
            }
            document.processing_meta.model_call_details.append(call_detail)
            
            # Update processing log
            cache_status = " (cached)" if cached else ""
            processing_log.append(
                f"LLM: Extracted {new_fields_added} fields, "
                f"cost: ${estimated_cost:.4f}, tokens: {tokens_used}{cache_status}"
            )
            
            return new_fields_added
            
        except Exception as e:
            error_msg = f"LLM: Error during gap-filling: {e}"
            processing_log.append(error_msg)
            logger.error(error_msg)
            return 0
    
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