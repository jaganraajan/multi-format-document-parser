"""
LLM-based field extractor for gap-filling missing required fields.

Orchestrates prompt creation, LLM invocation, and JSON parsing
to extract missing fields from document content.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from ..schema import KeyValue
from .client import OpenAIClient
from .cache import LLMResponseCache

logger = logging.getLogger(__name__)


class LLMFieldExtractor:
    """LLM-based field extraction for gap-filling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM field extractor.
        
        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        
        # Initialize OpenAI client
        model = config.get('model', 'gpt-4o-mini')
        temperature = config.get('temperature', 0.0)
        self.client = OpenAIClient(model=model, temperature=temperature)
        
        # Initialize cache
        cache_config = config.get('cache', {})
        cache_enabled = cache_config.get('enabled', True)
        cache_dir = cache_config.get('dir', 'llm_cache')
        self.cache = LLMResponseCache(cache_dir=cache_dir, enabled=cache_enabled)
        
        # Cost estimation settings
        cost_estimates = config.get('cost_estimates_per_1k_tokens', {})
        self.prompt_cost_per_1k = cost_estimates.get('prompt', 0.003)
        self.completion_cost_per_1k = cost_estimates.get('completion', 0.006)
        
        # Prompt templates
        self.prompt_preamble = config.get('prompt_preamble', 
            "You are an information extraction assistant. Extract ONLY the requested fields.")
        self.json_schema_note = config.get('json_schema_note',
            "Return valid JSON object with keys exactly matching the requested field names.")
        
        logger.info(f"LLM extractor initialized (available: {self.client.is_available})")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM extraction is available."""
        return self.client.is_available
    
    def extract_missing_fields(self, 
                              document_content: str,
                              missing_fields: List[str],
                              signature_id: Optional[str] = None,
                              content_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract missing fields using LLM.
        
        Args:
            document_content: Full document text content
            missing_fields: List of field names to extract
            signature_id: Document signature ID (for caching)
            content_hash: Document content hash (for caching)
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        if not self.is_available:
            logger.debug("LLM not available, skipping extraction")
            return {
                "extracted_fields": [],
                "cost_estimated": 0.0,
                "tokens_used": 0,
                "cached": False,
                "error": "LLM not available"
            }
        
        if not missing_fields:
            logger.debug("No missing fields to extract")
            return {
                "extracted_fields": [],
                "cost_estimated": 0.0,
                "tokens_used": 0,
                "cached": False
            }
        
        # Check cache first
        cached_response = None
        if signature_id and content_hash:
            cached_response = self.cache.get(signature_id, content_hash, missing_fields)
            if cached_response:
                logger.debug(f"Using cached response for {len(missing_fields)} fields")
                return cached_response.get("response_data", {})
        
        # Generate prompt
        prompt = self._create_extraction_prompt(document_content, missing_fields)
        
        # Estimate token usage
        prompt_tokens = self.client.estimate_tokens(prompt)
        estimated_completion_tokens = len(missing_fields) * 10  # Rough estimate
        
        # Get LLM response
        response_text = self.client.generate_completion(prompt, max_tokens=500)
        
        if not response_text:
            error_result = {
                "extracted_fields": [],
                "cost_estimated": 0.0,
                "tokens_used": 0,
                "cached": False,
                "error": "Failed to get LLM response"
            }
            return error_result
        
        # Parse JSON response
        extracted_fields = self._parse_llm_response(response_text, missing_fields)
        
        # Calculate actual token usage and cost
        actual_completion_tokens = self.client.estimate_tokens(response_text)
        total_tokens = prompt_tokens + actual_completion_tokens
        estimated_cost = self.client.estimate_cost(
            prompt_tokens, actual_completion_tokens,
            self.prompt_cost_per_1k, self.completion_cost_per_1k
        )
        
        # Prepare result
        result = {
            "extracted_fields": extracted_fields,
            "cost_estimated": estimated_cost,
            "tokens_used": total_tokens,
            "cached": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": actual_completion_tokens
        }
        
        # Cache the response
        if signature_id and content_hash:
            self.cache.set(signature_id, content_hash, missing_fields, result)
        
        logger.info(f"LLM extracted {len(extracted_fields)} fields, "
                   f"cost: ${estimated_cost:.4f}, tokens: {total_tokens}")
        
        return result
    
    def _create_extraction_prompt(self, document_content: str, 
                                 missing_fields: List[str]) -> str:
        """
        Create extraction prompt for LLM.
        
        Args:
            document_content: Document text content
            missing_fields: List of field names to extract
            
        Returns:
            Formatted prompt string
        """
        fields_list = "\n".join(f"- {field}" for field in missing_fields)
        
        prompt = f"""{self.prompt_preamble}

Document Content:
{document_content[:2000]}...

Extract ONLY these fields:
{fields_list}

{self.json_schema_note}

If a field cannot be found, set its value to null.

Response:"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str, 
                           expected_fields: List[str]) -> List[KeyValue]:
        """
        Parse LLM JSON response into KeyValue objects.
        
        Args:
            response_text: Raw LLM response
            expected_fields: List of expected field names
            
        Returns:
            List of KeyValue objects for successfully extracted fields
        """
        extracted_fields = []
        
        try:
            # Try to parse JSON response
            response_data = json.loads(response_text.strip())
            
            if not isinstance(response_data, dict):
                logger.warning("LLM response is not a JSON object")
                return extracted_fields
            
            # Extract each field
            for field_name in expected_fields:
                if field_name in response_data:
                    value = response_data[field_name]
                    
                    # Skip null/empty values
                    if value is not None and str(value).strip():
                        # Create KeyValue with model extraction method
                        kv = KeyValue(
                            key=field_name,
                            value=value,
                            confidence=0.55,  # Base confidence for model-derived fields
                            extraction_method="model",
                            metadata={
                                "llm_extracted": True,
                                "model": self.config.get('model', 'gpt-4o-mini')
                            }
                        )
                        extracted_fields.append(kv)
                        logger.debug(f"Extracted field: {field_name} = {value}")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return extracted_fields
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM extractor statistics."""
        return {
            "available": self.is_available,
            "model": self.config.get('model', 'unknown'),
            "cache_stats": self.cache.get_stats()
        }