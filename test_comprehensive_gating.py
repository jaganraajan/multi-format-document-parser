#!/usr/bin/env python3
"""
Comprehensive test script for all gating scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def create_test_documents():
    """Create test documents for different gating scenarios."""
    
    # Document 1: Complete invoice (should skip LLM - high confidence)
    complete_doc = """Complete Invoice Document

Invoice Number: INV-98765
Date: 12/20/2024
Total Amount: $2,500.00

From: Beta Corporation
Email: accounts@beta.com
Phone: (555) 987-6543

This is a complete invoice with all required fields."""
    
    # Document 2: Missing total amount (should call LLM - missing required field)
    missing_required_doc = """Invoice Document Missing Total

Invoice Number: INV-11111
Date: 12/21/2024

From: Gamma Corp
Email: billing@gamma.com
Phone: (555) 111-2222

This invoice is missing the total amount field."""
    
    # Document 3: No required fields (should call LLM - low coverage)
    no_required_doc = """Document Without Required Fields

From: Delta Inc
Email: contact@delta.com
Phone: (555) 333-4444
Date: 12/22/2024

This document has no invoice number or total amount."""
    
    # Document 4: Borderline case (will be sampled)
    borderline_doc = """Borderline Invoice Document

Invoice Number: INV-BORDER
Total Amount: $500.00
Date: 12/23/2024

From: Epsilon LLC
Email: info@epsilon.com

This has required fields but might trigger borderline sampling."""
    
    # Create test files
    os.makedirs('/tmp/gating_tests', exist_ok=True)
    
    test_docs = [
        ('/tmp/gating_tests/complete_invoice.txt', complete_doc),
        ('/tmp/gating_tests/missing_total.txt', missing_required_doc),
        ('/tmp/gating_tests/no_required_fields.txt', no_required_doc),
        ('/tmp/gating_tests/borderline_invoice.txt', borderline_doc)
    ]
    
    for filepath, content in test_docs:
        with open(filepath, 'w') as f:
            f.write(content)
    
    return [filepath for filepath, _ in test_docs]


def test_all_gating_scenarios():
    """Test all different gating scenarios."""
    print("🧪 COMPREHENSIVE GATING LOGIC TESTING")
    print("=" * 70)
    
    # Create test documents
    test_files = create_test_documents()
    
    # Initialize pipeline with different threshold settings for testing
    pipeline = DocumentPipeline(
        enable_llm=True, 
        enable_di=False,
        high_conf_threshold=0.85,
        min_required_coverage=0.75,
        borderline_sample_rate=0.5  # Higher rate for testing
    )
    
    scenarios = [
        (test_files[0], "Complete Invoice (should skip LLM - high confidence)"),
        (test_files[1], "Missing Total Amount (should call LLM - missing required)"),
        (test_files[2], "No Required Fields (should call LLM - low coverage)"),
        (test_files[3], "Borderline Case (may call or skip LLM - sampling)")
    ]
    
    results = []
    
    for file_path, description in scenarios:
        print(f"\n📋 Testing: {description}")
        print("-" * 50)
        
        try:
            document, log = pipeline.process_document(file_path)
            
            result = {
                'file': os.path.basename(file_path),
                'description': description,
                'gating_decision': document.processing_meta.gating_decision,
                'doc_confidence': document.processing_meta.document_confidence,
                'coverage_ratio': document.processing_meta.coverage_ratio,
                'required_present': document.processing_meta.required_fields_present,
                'required_total': document.processing_meta.required_fields_total,
                'fields_extracted': len(document.key_values),
                'ai_used': document.processing_meta.ai_used,
                'di_used': document.processing_meta.di_used
            }
            results.append(result)
            
            print(f"File: {result['file']}")
            print(f"Gating Decision: {result['gating_decision']}")
            print(f"Document Confidence: {result['doc_confidence']:.2f}")
            print(f"Coverage Ratio: {result['coverage_ratio']:.2f}")
            print(f"Required Fields: {result['required_present']}/{result['required_total']}")
            print(f"Fields Extracted: {result['fields_extracted']}")
            print(f"AI Used: LLM={result['ai_used']}, DI={result['di_used']}")
            
            # Show extracted fields
            if document.key_values:
                print("Extracted Fields:")
                for kv in document.key_values:
                    print(f"  - {kv.key}: {kv.value} (conf: {kv.confidence:.2f})")
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'file': os.path.basename(file_path),
                'description': description,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 GATING DECISIONS SUMMARY")
    print("=" * 70)
    for result in results:
        if 'error' not in result:
            status = "✅" if result['gating_decision'].startswith('skip_llm') else "🤖"
            print(f"{status} {result['file']}: {result['gating_decision']}")
        else:
            print(f"❌ {result['file']}: {result['error']}")
    
    print(f"\n🎯 GATING LOGIC VERIFICATION:")
    print("- Documents with all required fields + high confidence → skip_llm_high_conf")
    print("- Documents missing required fields → call_llm_missing_required") 
    print("- Documents with low coverage → call_llm_low_coverage")
    print("- Borderline documents → sampling (call_llm_borderline_sampled or skip_llm_borderline_not_sampled)")


if __name__ == "__main__":
    test_all_gating_scenarios()