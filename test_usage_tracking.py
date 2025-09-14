#!/usr/bin/env python3
"""
Test script for usage tracking functionality.
"""

import sys
import os
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline
from src.normalization.usage_tracker import UsageTracker


def test_usage_tracker():
    """Test basic usage tracker functionality."""
    print("🧪 Testing UsageTracker class...")
    
    tracker = UsageTracker()
    
    # Test initial state
    initial = tracker.snapshot()
    print(f"Initial state: {initial}")
    assert initial['documents_processed'] == 0
    assert initial['llm_calls'] == 0
    assert initial['di_calls'] == 0
    assert initial['cost']['total_cost'] == 0.0
    
    # Test recording calls
    tracker.record_llm_call({'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
    tracker.record_di_call(3)
    tracker.record_rules_hit(5)
    tracker.record_document_time(2.5)
    
    # Test snapshot
    snapshot = tracker.snapshot()
    print(f"After operations: {snapshot}")
    assert snapshot['documents_processed'] == 1
    assert snapshot['llm_calls'] == 1
    assert snapshot['di_calls'] == 1
    assert snapshot['rule_field_hits'] == 5
    assert snapshot['input_tokens'] == 100
    assert snapshot['output_tokens'] == 50
    assert snapshot['di_pages_processed'] == 3
    assert snapshot['cost']['total_cost'] > 0
    
    # Test reset
    tracker.reset()
    reset_snapshot = tracker.snapshot()
    assert reset_snapshot['documents_processed'] == 0
    assert reset_snapshot['cost']['total_cost'] == 0.0
    
    print("✅ UsageTracker tests passed!")


def test_pipeline_integration():
    """Test pipeline integration with usage tracking."""
    print("\n🧪 Testing Pipeline integration...")
    
    # Test with different modes
    for enable_llm, enable_di, description in [
        (False, False, "Pure Local Mode"),
        (True, False, "LLM Only Mode"),
        (True, True, "LLM + DI Mode"),
    ]:
        print(f"\nTesting {description}...")
        
        pipeline = DocumentPipeline(enable_llm=enable_llm, enable_di=enable_di)
        
        # Get initial usage
        initial_usage = pipeline.get_usage_summary()
        print(f"  Initial usage: docs={initial_usage['documents_processed']}, cost=${initial_usage['cost']['total_cost']:.4f}")
        
        # Process test document
        test_doc_path = "test_documents/sample_invoice.txt"
        if os.path.exists(test_doc_path):
            document, log = pipeline.process_document(test_doc_path)
            
            # Get usage after processing
            final_usage = pipeline.get_usage_summary()
            print(f"  Final usage: docs={final_usage['documents_processed']}, "
                  f"llm_calls={final_usage['llm_calls']}, "
                  f"di_calls={final_usage['di_calls']}, "
                  f"rule_hits={final_usage['rule_field_hits']}, "
                  f"cost=${final_usage['cost']['total_cost']:.4f}")
            
            # Verify document was tracked
            assert final_usage['documents_processed'] == initial_usage['documents_processed'] + 1
            
            print(f"  ✅ {description} tracking verified!")
        else:
            print(f"  ⚠️ Test document not found, skipping processing test")
    
    print("✅ Pipeline integration tests passed!")


def test_cost_calculations():
    """Test cost calculation with environment variables."""
    print("\n🧪 Testing cost calculations...")
    
    # Test with custom pricing
    os.environ['COST_LLM_INPUT_PER_1K'] = '0.002'
    os.environ['COST_LLM_OUTPUT_PER_1K'] = '0.004'
    os.environ['COST_DI_PER_PAGE'] = '0.02'
    
    tracker = UsageTracker()
    
    # Simulate usage
    tracker.record_llm_call({'usage': {'prompt_tokens': 1000, 'completion_tokens': 500}})
    tracker.record_di_call(5)
    tracker.record_document_time(1.0)
    
    snapshot = tracker.snapshot()
    expected_llm_cost = (1000/1000) * 0.002 + (500/1000) * 0.004  # 0.002 + 0.002 = 0.004
    expected_di_cost = 5 * 0.02  # 0.1
    expected_total = expected_llm_cost + expected_di_cost  # 0.104
    
    print(f"  Expected LLM cost: ${expected_llm_cost:.4f}")
    print(f"  Actual LLM cost: ${snapshot['cost']['llm_cost']:.4f}")
    print(f"  Expected DI cost: ${expected_di_cost:.4f}")
    print(f"  Actual DI cost: ${snapshot['cost']['di_cost']:.4f}")
    print(f"  Expected total: ${expected_total:.4f}")
    print(f"  Actual total: ${snapshot['cost']['total_cost']:.4f}")
    
    assert abs(snapshot['cost']['llm_cost'] - expected_llm_cost) < 0.0001
    assert abs(snapshot['cost']['di_cost'] - expected_di_cost) < 0.0001
    assert abs(snapshot['cost']['total_cost'] - expected_total) < 0.0001
    
    # Clean up env vars
    del os.environ['COST_LLM_INPUT_PER_1K']
    del os.environ['COST_LLM_OUTPUT_PER_1K']
    del os.environ['COST_DI_PER_PAGE']
    
    print("✅ Cost calculation tests passed!")


def main():
    """Run all tests."""
    print("🚀 Running Usage Tracking Tests")
    print("=" * 50)
    
    try:
        test_usage_tracker()
        test_pipeline_integration()
        test_cost_calculations()
        
        print("\n🎉 All tests passed successfully!")
        print("\nUsage tracking feature is ready!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())