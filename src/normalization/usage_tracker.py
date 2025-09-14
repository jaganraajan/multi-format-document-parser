"""Usage tracking and cost estimation for LLM and DI calls.

This module provides a UsageTracker class that accumulates usage statistics
for the document processing pipeline, including LLM calls, DI calls, rule hits,
token usage, and estimated costs.
"""

import os
import threading
import time
from typing import Dict, Any, Optional


class UsageTracker:
    """Track usage statistics and estimate costs for document processing.
    
    Accumulates counters for:
    - LLM calls and token usage
    - Document Intelligence calls and page counts  
    - Rule engine field hits
    - Document processing metrics
    - Cost estimates based on configurable pricing
    """
    
    def __init__(self):
        """Initialize usage tracker with zero counters."""
        self._lock = threading.Lock()
        
        # Core counters
        self.llm_calls = 0
        self.di_calls = 0
        self.rule_field_hits = 0
        self.documents_processed = 0
        self.total_processing_seconds = 0.0
        
        # Token tracking
        self.input_tokens = 0
        self.output_tokens = 0
        
        # DI page tracking
        self.di_pages_processed = 0
        
        # Pricing constants (configurable via environment)
        self.llm_input_cost_per_1k = float(os.getenv('COST_LLM_INPUT_PER_1K', '0.0015'))
        self.llm_output_cost_per_1k = float(os.getenv('COST_LLM_OUTPUT_PER_1K', '0.0020'))
        self.di_page_cost = float(os.getenv('COST_DI_PER_PAGE', '0.01'))
    
    def record_llm_call(self, meta: Dict[str, Any]):
        """Record an LLM call with optional token usage metadata.
        
        Args:
            meta: Metadata dictionary that may include 'usage' with token counts
        """
        with self._lock:
            self.llm_calls += 1
            
            # Extract token usage if available
            usage = meta.get('usage', {})
            if isinstance(usage, dict):
                self.input_tokens += usage.get('prompt_tokens', 0)
                self.output_tokens += usage.get('completion_tokens', 0)
    
    def record_di_call(self, pages: int):
        """Record a Document Intelligence call with page count.
        
        Args:
            pages: Number of pages processed by DI
        """
        with self._lock:
            self.di_calls += 1
            self.di_pages_processed += max(0, pages)
    
    def record_rules_hit(self, count: int):
        """Record rule engine field extractions.
        
        Args:
            count: Number of fields extracted by rules
        """
        with self._lock:
            self.rule_field_hits += max(0, count)
    
    def record_document_time(self, seconds: float):
        """Record document processing time and increment document counter.
        
        Args:
            seconds: Time taken to process the document
        """
        with self._lock:
            self.documents_processed += 1
            self.total_processing_seconds += max(0.0, seconds)
    
    def snapshot(self) -> Dict[str, Any]:
        """Get current usage statistics and cost estimates.
        
        Returns:
            Dictionary with usage stats, averages, and cost breakdown
        """
        with self._lock:
            # Calculate averages
            avg_processing_seconds = (
                self.total_processing_seconds / self.documents_processed 
                if self.documents_processed > 0 else 0.0
            )
            avg_rule_fields_per_doc = (
                self.rule_field_hits / self.documents_processed 
                if self.documents_processed > 0 else 0.0
            )
            
            # Estimate AI usage ratio (approximate)
            ai_docs_estimate = min(self.llm_calls + self.di_calls, self.documents_processed)
            ai_doc_ratio = (
                ai_docs_estimate / self.documents_processed 
                if self.documents_processed > 0 else 0.0
            )
            
            # Calculate costs
            llm_cost = (
                (self.input_tokens / 1000.0) * self.llm_input_cost_per_1k + 
                (self.output_tokens / 1000.0) * self.llm_output_cost_per_1k
            )
            di_cost = self.di_pages_processed * self.di_page_cost
            total_cost = llm_cost + di_cost
            
            return {
                'documents_processed': self.documents_processed,
                'llm_calls': self.llm_calls,
                'di_calls': self.di_calls,
                'rule_field_hits': self.rule_field_hits,
                'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'di_pages_processed': self.di_pages_processed,
                'total_processing_seconds': round(self.total_processing_seconds, 2),
                'avg_processing_seconds': round(avg_processing_seconds, 2),
                'avg_rule_fields_per_doc': round(avg_rule_fields_per_doc, 1),
                'ai_doc_ratio': round(ai_doc_ratio, 2),
                'cost': {
                    'llm_cost': round(llm_cost, 4),
                    'di_cost': round(di_cost, 4),
                    'total_cost': round(total_cost, 4)
                }
            }
    
    def reset(self):
        """Reset all counters to zero (for testing or new sessions)."""
        with self._lock:
            self.llm_calls = 0
            self.di_calls = 0
            self.rule_field_hits = 0
            self.documents_processed = 0
            self.total_processing_seconds = 0.0
            self.input_tokens = 0
            self.output_tokens = 0
            self.di_pages_processed = 0


__all__ = ["UsageTracker"]