#!/usr/bin/env python3
"""
Test script for the Multi-Format Document Parser.

This script demonstrates the toggle functionality for LLM and DI extraction modes.
"""

import sys
import os
import json
import argparse
try:
    from dotenv import load_dotenv
    # Load nearest .env (current working dir or project root)
    load_dotenv()
except Exception:
    pass  # Safe to ignore if not installed yet

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def test_pipeline_with_toggles(enable_llm=True, enable_di=True):
    """Test the document processing pipeline with specific toggle settings."""
    print(f"🚀 Testing with LLM={enable_llm}, DI={enable_di}")
    print("=" * 60)
    
    try:
        # Initialize pipeline with specified toggles
        print("1. Initializing pipeline...")
        pipeline = DocumentPipeline(enable_llm=enable_llm, enable_di=enable_di)
        
        # Display active mode
        if not enable_llm and not enable_di:
            mode = "Pure Local Mode (no external AI calls)"
        elif enable_llm and not enable_di:
            mode = "LLM Only Mode"
        elif enable_llm and enable_di:
            mode = "LLM + DI Fallback Mode"
        else:  # not enable_llm and enable_di
            mode = "DI Only Mode (PDFs after local extraction)"
        
        print(f"✅ Pipeline initialized in: {mode}")
        
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
        print(f"Fields Extracted: {len(document.key_values)}")
        print(f"Model Calls Made: {document.processing_meta.model_calls_made}")
        
        # NEW: Display gating information
        print("\n4. Gating & Confidence Metrics:")
        print("-" * 30)
        print(f"Gating Decision: {document.processing_meta.gating_decision}")
        print(f"Document Confidence: {document.processing_meta.document_confidence:.2f}")
        print(f"Coverage Ratio: {document.processing_meta.coverage_ratio:.2f}")
        print(f"Required Fields: {document.processing_meta.required_fields_present}/{document.processing_meta.required_fields_total}")
        print(f"AI Used: LLM={document.processing_meta.ai_used}, DI={document.processing_meta.di_used}")
        
        # Highlight AI-related log lines
        print("\n5. AI Processing Log:")
        print("-" * 30)
        log_lines = processing_log.split('\n')
        ai_related_lines = [line for line in log_lines if any(keyword in line for keyword in ['LLM', 'DI ', 'disabled', 'selected', 'env', 'Gating', 'Coverage', 'confidence'])]
        for line in ai_related_lines:
            print(f"  {line}")
        
        # Display extracted fields if any
        if document.key_values:
            print("\n6. Extracted Key-Value Pairs:")
            print("-" * 30)
            for kv in document.key_values:
                print(f"  {kv.key}: {kv.value} (confidence: {kv.confidence:.2f}, method: {kv.extraction_method})")
        else:
            print("\n6. No fields extracted.")
            if enable_llm or enable_di:
                print("   Note: Azure environment variables may not be configured.")
        
        print(f"\n✅ Test completed successfully in {mode}!")
        return document.processing_meta.model_calls_made
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_all_modes():
    """Test all toggle combinations to demonstrate the different modes."""
    print("🌟 COMPREHENSIVE TOGGLE TESTING")
    print("=" * 60)
    
    test_configurations = [
        (False, False, "Pure Local Mode"),
        (True, False, "LLM Only Mode"),
        (True, True, "LLM + DI Mode"),
        (False, True, "DI Only Mode"),
    ]
    
    results = []
    
    for enable_llm, enable_di, description in test_configurations:
        print(f"\n📋 Testing {description}")
        print("-" * 40)
        model_calls = test_pipeline_with_toggles(enable_llm, enable_di)
        results.append((description, model_calls))
        print("\n")
    
    # Summary
    print("📊 SUMMARY OF ALL MODES:")
    print("=" * 60)
    for description, model_calls in results:
        status = f"Model calls: {model_calls}" if model_calls is not None else "Failed"
        print(f"  {description:<25} | {status}")
    
    return results


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description="Test Multi-Format Document Parser with toggle options")
    parser.add_argument("--llm", action="store_true", help="Enable LLM extraction")
    parser.add_argument("--di", action="store_true", help="Enable Document Intelligence fallback")
    parser.add_argument("--all-modes", action="store_true", help="Test all toggle combinations")
    
    args = parser.parse_args()
    
    if args.all_modes:
        test_all_modes()
    else:
        # Check environment variables for defaults if no CLI flags provided
        if len(sys.argv) == 1:  # No arguments provided
            enable_llm = os.getenv("TEST_ENABLE_LLM", "false").lower() == "true"
            enable_di = os.getenv("TEST_ENABLE_DI", "false").lower() == "true"
            print(f"No CLI flags provided. Using environment defaults: LLM={enable_llm}, DI={enable_di}")
            if not enable_llm and not enable_di:
                print("Running all modes demonstration...")
                test_all_modes()
            else:
                test_pipeline_with_toggles(enable_llm, enable_di)
        else:
            test_pipeline_with_toggles(args.llm, args.di)

if __name__ == "__main__":
    main()