"""LLM-based field extraction using Azure OpenAI.

This module provides an extractor that sends the document text to an Azure OpenAI
model with a structured prompt asking for specific fields and returns normalized
KeyValue objects.

Environment variables required (Azure OpenAI):
  AZURE_OPENAI_ENDPOINT      e.g. https://your-resource.openai.azure.com/
  AZURE_OPENAI_API_KEY       API key string
  AZURE_OPENAI_DEPLOYMENT    Deployment name of the GPT model (e.g. gpt-4o-mini)
  (Optional)
  AZURE_OPENAI_API_VERSION   API version, default: 2024-02-15-preview

If variables are missing, the extractor will operate in mock mode returning an empty list
and a log message so the pipeline does not fail locally.
"""
from __future__ import annotations

import json
import os
import logging
from typing import List, Dict, Any, Tuple

from .schema import KeyValue

logger = logging.getLogger(__name__)

DEFAULT_FIELDS = [
    {"name": "invoice_number", "description": "Unique invoice identifier"},
    {"name": "total_amount", "description": "Total amount due as a number"},
    {"name": "date", "description": "Invoice date in ISO format if possible"},
    {"name": "vendor_name", "description": "Vendor or company name"},
    {"name": "email", "description": "Primary email address present"},
    {"name": "phone_number", "description": "Primary phone number"},
]

SYSTEM_PROMPT = (
    "You are an information extraction assistant. Extract the requested fields from the document text. "
    "Return STRICT JSON only with keys exactly matching the field names. For any missing field use null."
)

USER_PROMPT_TEMPLATE = """Extract the following fields from this document.\n\nFields:\n{field_lines}\n\nReturn JSON object with these keys only. Document Text:\n---\n{document_text}\n---\n"""


class AzureOpenAILLMExtractor:
    """Extractor that uses Azure OpenAI to extract structured fields."""

    def __init__(self, fields: List[Dict[str, str]] | None = None):
        self.fields = fields or DEFAULT_FIELDS
        self.enabled = all([
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        ])
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        if not self.enabled:
            logger.warning("Azure OpenAI environment variables missing; LLM extraction disabled (mock mode).")
        else:
            try:
                from openai import AzureOpenAI  # type: ignore
                self._client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=self.api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                )
            except Exception as e:
                logger.error(f"Failed to initialize AzureOpenAI client: {e}")
                self.enabled = False

    def _build_prompt(self, text: str) -> str:
        field_lines = "\n".join([f"- {f['name']}: {f['description']}" for f in self.fields])
        truncated = text if len(text) < 8000 else text[:8000] + "\n...[TRUNCATED]"
        return USER_PROMPT_TEMPLATE.format(field_lines=field_lines, document_text=truncated)

    def extract(self, text: str) -> Tuple[List[KeyValue], Dict[str, Any]]:
        """Run extraction. Returns list of KeyValue and metadata log."""
        metadata: Dict[str, Any] = {"model_used": None, "raw_response": None, "error": None}
        if not self.enabled:
            metadata["error"] = "llm_disabled"
            return [], metadata

        prompt = self._build_prompt(text)
        try:
            # Chat Completions API call
            response = self._client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            metadata["model_used"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            content = response.choices[0].message.content if response.choices else "{}"
            metadata["raw_response"] = content
            data = self._safe_parse_json(content)
            kvs: List[KeyValue] = []
            for field in self.fields:
                name = field["name"]
                if name in data and data[name] not in (None, ""):
                    value = data[name]
                    # Attempt numeric conversion for amount fields
                    if name in ("total_amount",) and isinstance(value, str):
                        cleaned = value.replace("$", "").replace(",", "").strip()
                        try:
                            if "." in cleaned:
                                value = float(cleaned)
                            else:
                                value = int(cleaned)
                        except ValueError:
                            pass
                    kvs.append(KeyValue(key=name, value=value, confidence=0.85, extraction_method="model"))
            return kvs, metadata
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            metadata["error"] = str(e)
            return [], metadata

    def _safe_parse_json(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except Exception:
            # Attempt to fix common JSON issues
            try:
                cleaned = content.strip().strip('`')
                return json.loads(cleaned)
            except Exception:
                return {}

__all__ = ["AzureOpenAILLMExtractor"]
