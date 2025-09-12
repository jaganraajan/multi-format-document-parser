"""
Test script for LLM gap-filling functionality.
Tests the pipeline with a document missing required fields.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization.pipeline import DocumentPipeline


def test_llm_gap_filling():
    """Test LLM gap-filling with mocked OpenAI response."""
    print("🧪 Testing LLM Gap-Filling Functionality")
    print("=" * 50)
    
    # Test document with missing invoice_number (required field)
    test_doc_path = "test_documents/sample_invoice_missing_field.txt"
    
    if not os.path.exists(test_doc_path):
        print("❌ Test document not found")
        return
    
    # Initialize pipeline
    print("1. Initializing pipeline...")
    try:
        pipeline = DocumentPipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return
    
    # Test without LLM (should miss invoice_number)
    print("\n2. Testing without LLM (baseline)...")
    try:
        document, processing_log = pipeline.process_document(test_doc_path)
        extracted_fields = [kv.key for kv in document.key_values]
        print(f"✅ Extracted fields: {extracted_fields}")
        
        # Check if invoice_number is missing
        if 'invoice_number' not in extracted_fields:
            print("✅ As expected, invoice_number is missing (required field not extracted by rules)")
        else:
            print("⚠️  invoice_number was extracted by rules (unexpected)")
            
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return
    
    # Mock LLM response for testing
    print("\n3. Testing with mocked LLM response...")
    if pipeline.llm_extractor and pipeline.llm_extractor.is_available:
        try:
            # Create mock response that includes the missing invoice_number
            mock_response = {
                "extracted_fields": [
                    type('MockKeyValue', (), {
                        'key': 'invoice_number',
                        'value': 'INV-MOCK-9999',
                        'confidence': 0.55,
                        'extraction_method': 'model',
                        'metadata': {'llm_extracted': True, 'model': 'gpt-4o-mini'}
                    })()
                ],
                "cost_estimated": 0.012,
                "tokens_used": 150,
                "cached": False
            }
            
            # Patch the LLM extractor's extract_missing_fields method
            with patch.object(pipeline.llm_extractor, 'extract_missing_fields', return_value=mock_response):
                document, processing_log = pipeline.process_document(test_doc_path)
                
                extracted_fields = [kv.key for kv in document.key_values]
                llm_fields = [kv.key for kv in document.key_values if kv.extraction_method == 'model']
                
                print(f"✅ Total extracted fields: {extracted_fields}")
                print(f"✅ LLM-extracted fields: {llm_fields}")
                print(f"✅ Model calls made: {document.processing_meta.model_calls_made}")
                print(f"✅ Total cost: ${document.processing_meta.total_cost_usd:.4f}")
                
                # Verify LLM gap-filling worked
                if 'invoice_number' in llm_fields:
                    print("✅ LLM gap-filling successful: invoice_number extracted by model")
                else:
                    print("❌ LLM gap-filling failed: invoice_number not extracted")
                
                # Check processing log
                print("\n4. Processing Log:")
                print("-" * 30)
                for line in processing_log.split('\n'):
                    if 'LLM:' in line:
                        print(f"  {line}")
                        
        except Exception as e:
            print(f"❌ LLM test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  LLM extractor not available (missing API key or dependencies)")
        print("   This is expected behavior when OPENAI_API_KEY is not set")
    
    print("\n✅ LLM gap-filling test completed!")


if __name__ == "__main__":
    test_llm_gap_filling()