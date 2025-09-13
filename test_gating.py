#!/usr/bin/env python3
"""
Test script for gating logic demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def test_gating_scenarios():
    """Test different gating scenarios."""
    print("🧪 GATING LOGIC TESTING")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        ("test_documents/sample_invoice.txt", "Complete Invoice (should skip LLM)"),
        ("test_documents/partial_invoice.txt", "Partial Invoice (should call LLM)")
    ]
    
    for file_path, description in scenarios:
        print(f"\n📋 Testing: {description}")
        print("-" * 40)
        
        # Test with LLM enabled to see gating decisions
        pipeline = DocumentPipeline(enable_llm=True, enable_di=False)
        
        try:
            document, log = pipeline.process_document(file_path)
            
            print(f"Document: {os.path.basename(file_path)}")
            print(f"Gating Decision: {document.processing_meta.gating_decision}")
            print(f"Document Confidence: {document.processing_meta.document_confidence:.2f}")
            print(f"Coverage Ratio: {document.processing_meta.coverage_ratio:.2f}")
            print(f"Required Fields: {document.processing_meta.required_fields_present}/{document.processing_meta.required_fields_total}")
            print(f"AI Used: LLM={document.processing_meta.ai_used}, DI={document.processing_meta.di_used}")
            print(f"Fields Extracted: {len(document.key_values)}")
            
            if document.key_values:
                print("Extracted Fields:")
                for kv in document.key_values:
                    print(f"  - {kv.key}: {kv.value} (conf: {kv.confidence:.2f})")
            else:
                print("No fields extracted")
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_gating_scenarios()