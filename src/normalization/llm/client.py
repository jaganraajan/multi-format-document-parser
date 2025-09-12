"""
OpenAI client abstraction for LLM-based field extraction.

Provides a minimal provider abstraction with graceful degradation
when dependencies or API keys are missing.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client wrapper with graceful degradation."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize OpenAI client.
        
        Args:
            model: OpenAI model name
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        self._client = None
        self._available = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client if available."""
        try:
            # Check for API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. LLM features will be disabled.")
                return
            
            # Try to import and initialize OpenAI
            import openai
            self._client = openai.OpenAI(api_key=api_key)
            self._available = True
            logger.info(f"OpenAI client initialized with model: {self.model}")
            
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai>=1.0.0")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if OpenAI client is available."""
        return self._available
    
    def generate_completion(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Generate completion using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if unavailable
        """
        if not self._available:
            logger.debug("OpenAI client not available, skipping completion")
            return None
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        This is a rough heuristic: ~4 characters per token on average.
        For production use, consider using tiktoken library.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 4 characters per token
        return len(text) // 4
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, 
                     cost_per_1k_prompt: float = 0.003, 
                     cost_per_1k_completion: float = 0.006) -> float:
        """
        Estimate cost based on token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost_per_1k_prompt: Cost per 1000 prompt tokens
            cost_per_1k_completion: Cost per 1000 completion tokens
            
        Returns:
            Estimated cost in USD
        """
        prompt_cost = (prompt_tokens / 1000) * cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * cost_per_1k_completion
        return prompt_cost + completion_cost