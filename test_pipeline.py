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
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = DocumentPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Test document path
        test_doc_path = "test_documents/Srilankan_inv.pdf"
        
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
        print(f"Fields Extracted (LLM): {len(document.key_values)}")
        
        # Display extracted fields
        if document.key_values:
            print("\n4. Extracted Key-Value Pairs:")
            print("-" * 30)
            for kv in document.key_values:
                print(f"  {kv.key}: {kv.value} (confidence: {kv.confidence:.2f})")
        else:
            print("\n4. No fields extracted. If you expect values, ensure Azure OpenAI env vars are set: "
                  "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
        
        # Display sections
        print(f"\n5. Document Sections: {len(document.sections)}")
        for i, section in enumerate(document.sections):
            print(f"  Section {i+1}: {section.title}")
        
        # Display processing log
        print("\n6. Processing Log:")
        print("-" * 30)
        print(processing_log)
        
        # Get pipeline stats
        print("\n7. Pipeline Statistics:")
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