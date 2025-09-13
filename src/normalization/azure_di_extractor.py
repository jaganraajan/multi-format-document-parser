"""Azure Document Intelligence (Form Recognizer) based fallback extractor.

Uses Azure Document Intelligence to extract key fields when the primary LLM
extraction returns insufficient results for PDFs.

Environment variables required:
  AZURE_DI_ENDPOINT   (e.g. https://<resource>.cognitiveservices.azure.com/)
  AZURE_DI_KEY        (API key)
  AZURE_DI_MODEL_ID   (custom model ID or 'prebuilt-invoice')

If variables missing, extractor is disabled and returns empty list.
"""
from __future__ import annotations

import os
import logging
from typing import List, Tuple, Dict, Any

from .schema import KeyValue

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "prebuilt-invoice"

class AzureDocumentIntelligenceExtractor:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_DI_ENDPOINT")
        self.key = os.getenv("AZURE_DI_KEY") or os.getenv("AZURE_FORMRECOGNIZER_KEY")
        self.model_id = os.getenv("AZURE_DI_MODEL_ID", DEFAULT_MODEL_ID)
        self.enabled = bool(self.endpoint and self.key and self.model_id)
        if self.enabled:
            try:
                from azure.ai.formrecognizer import DocumentAnalysisClient
                from azure.core.credentials import AzureKeyCredential
                self._client = DocumentAnalysisClient(self.endpoint, AzureKeyCredential(self.key))
            except Exception as e:
                logger.error(f"Failed to init DocumentAnalysisClient: {e}")
                self.enabled = False
        else:
            logger.warning("Azure DI vars missing; DI fallback disabled.")

    def extract(self, file_path: str) -> Tuple[List[KeyValue], Dict[str, Any]]:
        """Run Azure DI analysis and return normalized key-value pairs.

        Normalization steps:
          * Map varied DI field names to unified internal names
          * Flatten address objects to a single readable string
          * Transform line items (dictionaries) into list of simple dicts
          * Dedupe scalar fields keeping the highest-confidence value
          * Skip obviously empty or noisy placeholder values
        """
        meta: Dict[str, Any] = {"model_used": self.model_id, "error": None}
        if not self.enabled:
            meta["error"] = "di_disabled"
            return [], meta
        try:
            from azure.ai.formrecognizer import AnalyzeResult, DocumentField
            with open(file_path, 'rb') as f:
                poller = self._client.begin_analyze_document(self.model_id, document=f)
                result: AnalyzeResult = poller.result()

            # Aggregators
            best_scalar: Dict[str, Tuple[float, Any]] = {}  # name -> (confidence, value)
            list_fields: Dict[str, List[Any]] = {}

            if hasattr(result, 'documents'):
                for doc in result.documents:
                    for raw_name, field in (getattr(doc, 'fields', {}) or {}).items():
                        if not field:
                            continue
                        mapped = self._map_field_name(raw_name)
                        normalized_value, is_list, conf = self._normalize_field(field, mapped)
                        if normalized_value in (None, ""):
                            continue
                        if is_list:
                            list_fields.setdefault(mapped, []).extend(normalized_value if isinstance(normalized_value, list) else [normalized_value])
                        else:  # scalar
                            prev = best_scalar.get(mapped)
                            if (not prev) or conf > prev[0]:
                                best_scalar[mapped] = (conf, normalized_value)

            kvs: List[KeyValue] = []
            for name, (conf, val) in best_scalar.items():
                kvs.append(KeyValue(key=name, value=val, confidence=round(float(conf), 3), extraction_method="azure_di"))
            for name, items in list_fields.items():
                kvs.append(KeyValue(key=name, value=items, confidence=0.70, extraction_method="azure_di"))

            meta["fields_extracted"] = len(kvs)
            return kvs, meta
        except Exception as e:
            logger.error(f"Azure DI extraction failed: {e}")
            meta["error"] = str(e)
            return [], meta

    def _map_field_name(self, name: str) -> str:
        low = name.lower()
        mapping = {
            "invoiceid": "invoice_number",
            "invoice_id": "invoice_number",
            "invoicenumber": "invoice_number",
            "invoice_number": "invoice_number",
            "vendorname": "vendor_name",
            "suppliername": "vendor_name",
            "customername": "recipient_name",
            "vendoraddressrecipient": "recipient_name",
            "billingaddressrecipient": "recipient_name",
            "totalamount": "total_amount",
            "amountdue": "total_amount",
            "duedate": "due_date",
            "invoicedate": "date",
            "invoice_date": "date",
            "billingaddress": "supplier_address",
            "customeraddress": "recipient_address",
            "vendoraddress": "supplier_address",
            "vendortaxid": "vendor_tax_id",
            "customertaxid": "customer_tax_id",
            "totaltax": "total_tax",
            "invoicetotal": "total_amount",
            "taxdetails": "tax_details",
            "items": "line_items",
        }
        return mapping.get(low, low)

    def _normalize_field(self, field, mapped_name: str) -> Tuple[Any, bool, float]:
        """Return (value, is_list, confidence)."""
        confidence = getattr(field, 'confidence', 0.7) or 0.7
        value = getattr(field, 'value', None)
        vtype = getattr(field, 'value_type', None)
        content = getattr(field, 'content', None)

        # Address flattening
        if vtype in ("address",) or (value and value.__class__.__name__.lower().startswith("address")):
            parts = []
            for attr in ["street_address", "house", "house_number", "road", "suburb", "city", "state", "postal_code", "country_region", "state_district", "city_district", "level"]:
                comp = getattr(value, attr, None) if value else None
                if comp and str(comp).strip():
                    parts.append(str(comp).strip().rstrip(','))
            if not parts and content:
                parts = [content.strip()]
            flat = ", ".join(dict.fromkeys([p for p in parts if p]))  # de-duplicate order
            return (flat, False, confidence)

        # Currency / numeric value
        if vtype == 'currency' and value is not None:
            amount = getattr(value, 'amount', None)
            if amount is not None:
                return (amount, False, confidence)

        # Line items / collections (dictionary with nested fields)
        if mapped_name == 'line_items' or vtype in ("array",):
            # Field.value may be list[DocumentField]
            items_out = []
            iterable = []
            if isinstance(value, list):
                iterable = value
            elif isinstance(field.value, list):
                iterable = field.value
            elif isinstance(field.value, dict):
                iterable = [field]  # treat single dict-like field
            else:
                # For some SDK versions items come as a list inside field.value (DocumentField objects)
                pass
            for itm in iterable:
                try:
                    if hasattr(itm, 'value') and isinstance(itm.value, dict):
                        row = {}
                        for k2, v2 in itm.value.items():
                            val_norm, _, _ = self._normalize_field(v2, k2.lower())
                            row[k2.lower()] = val_norm
                        if row:
                            items_out.append(row)
                    else:
                        txt = getattr(itm, 'content', None)
                        if txt:
                            items_out.append({"raw": txt.strip()})
                except Exception:
                    continue
            if items_out:
                return (items_out, True, confidence)
            # fallback to raw content if nothing parsed
            if content:
                return ([{"raw": content.strip()}], True, confidence)

        # Dictionaries (e.g., nested tax details) -> flatten relevant primitive fields
        if isinstance(value, dict):
            flat_map = {}
            for k, v in value.items():
                try:
                    if hasattr(v, 'value') and getattr(v, 'value', None) is not None:
                        flat_map[k.lower()] = getattr(v, 'value')
                    else:
                        vc = getattr(v, 'content', None)
                        if vc:
                            flat_map[k.lower()] = vc
                except Exception:
                    continue
            if flat_map:
                return (flat_map, False, confidence)

        # Fallback: use .value if primitive else .content
        primitive = None
        if value is not None and isinstance(value, (str, int, float)):
            primitive = value
        elif content:
            primitive = content
        if primitive is not None:
            # Basic cleanup
            if isinstance(primitive, str):
                primitive = primitive.strip()
            return (primitive, False, confidence)
        return (None, False, confidence)

__all__ = ["AzureDocumentIntelligenceExtractor"]
