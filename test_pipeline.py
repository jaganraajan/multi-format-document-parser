#!/usr/bin/env python3
"""
Test script for the Multi-Format Document Parser.

This script demonstrates the basic functionality of the document processing pipeline.
"""

import sys
import os
import json
try:
    from dotenv import load_dotenv
    # Load nearest .env (current working dir or project root)
    load_dotenv()
except Exception:
    pass  # Safe to ignore if not installed yet

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def test_pipeline():
    """Test the document processing pipeline."""
    print("🚀 Testing Multi-Format Document Parser")
    print("=" * 50)
    
    try:
        # Initialize pipeline in hybrid mode to test gating logic
        print("1. Initializing pipeline...")
        pipeline = DocumentPipeline(use_rules=True)
        print("✅ Pipeline initialized successfully (hybrid mode enabled)")
        
        # Test document path
        test_doc_path = "test_documents/sample_invoice.txt"
        
        if not os.path.exists(test_doc_path):
            print(f"❌ Test document not found: {test_doc_path}")
            return
        
        # Process document
        print(f"2. Processing document: {test_doc_path}")
        document, processing_log = pipeline.process_document(test_doc_path)
        print("✅ Document processed successfully")
        
        # Display results
        print("\n3. Processing Results:")
        print("-" * 30)
        print(f"Document ID: {document.doc_id}")
        print(f"File Type: {document.ingest_metadata.file_type}")
        print(f"Processing Time: {document.ingest_metadata.processing_time_seconds:.2f} seconds")
        print(f"Signature ID: {document.processing_meta.signature_id}")
        
        # Display gating decision and LLM usage
        print(f"\n4. Gating Decision & LLM Usage:")
        print("-" * 30)
        print(f"Gating Decision: {document.processing_meta.gating_decision}")
        print(f"LLM Invoked: {'Yes' if document.processing_meta.llm_invoked else 'No'}")
        print(f"Model Calls Made: {document.processing_meta.model_calls_made}")
        
        # Display confidence metrics
        print(f"\n5. Confidence Metrics:")
        print("-" * 30)
        print(f"Document Confidence: {document.processing_meta.document_confidence:.2%}")
        print(f"Required Fields: {len(document.processing_meta.required_fields)}")
        print(f"Required Fields List: {document.processing_meta.required_fields}")
        
        # Calculate required field coverage
        extracted_required = [kv for kv in document.key_values if kv.key in document.processing_meta.required_fields]
        coverage_ratio = len(extracted_required) / len(document.processing_meta.required_fields) if document.processing_meta.required_fields else 1.0
        print(f"Required Field Coverage: {coverage_ratio:.2%} ({len(extracted_required)}/{len(document.processing_meta.required_fields)})")
        
        print(f"Fields Extracted Total: {len(document.key_values)}")

        # Display extracted fields
        if document.key_values:
            print("\n6. Extracted Key-Value Pairs:")
            print("-" * 30)
            for kv in document.key_values:
                required_marker = " ⭐" if kv.key in document.processing_meta.required_fields else ""
                print(f"  {kv.key}{required_marker}: {kv.value} (confidence: {kv.confidence:.2f}, method: {kv.extraction_method})")
        else:
            print("\n6. No fields extracted. If you expect values, ensure Azure OpenAI env vars are set: "
                  "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
        
        # Display sections
        print(f"\n7. Document Sections: {len(document.sections)}")
        for i, section in enumerate(document.sections):
            print(f"  Section {i+1}: {section.title}")
        
        # Display processing log
        print("\n8. Processing Log:")
        print("-" * 30)
        print(processing_log)
        
        # Get pipeline stats
        print("\n9. Pipeline Statistics:")
        print("-" * 30)
        stats = pipeline.get_pipeline_stats()
        print(json.dumps(stats, indent=2, default=str))
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pipeline()