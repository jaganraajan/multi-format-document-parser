#!/usr/bin/env python3
"""
Demo script showing usage tracking functionality in action.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def demo_usage_tracking():
    """Demonstrate the usage tracking functionality."""
    print("🚀 Usage Tracking Demo")
    print("=" * 50)
    
    # Set custom pricing for demo
    os.environ['COST_LLM_INPUT_PER_1K'] = '0.005'
    os.environ['COST_LLM_OUTPUT_PER_1K'] = '0.010'
    os.environ['COST_DI_PER_PAGE'] = '0.02'
    
    print("📊 Custom pricing set:")
    print("  LLM Input:  $0.005 per 1K tokens")
    print("  LLM Output: $0.010 per 1K tokens") 
    print("  DI Pages:   $0.020 per page")
    print()
    
    # Create pipeline
    pipeline = DocumentPipeline(enable_llm=True, enable_di=True)
    
    # Show initial usage
    initial_usage = pipeline.get_usage_summary()
    print("📈 Initial Usage Summary:")
    print(f"  Documents: {initial_usage['documents_processed']}")
    print(f"  LLM Calls: {initial_usage['llm_calls']}")
    print(f"  DI Calls: {initial_usage['di_calls']}")
    print(f"  Rule Fields: {initial_usage['rule_field_hits']}")
    print(f"  Total Cost: ${initial_usage['cost']['total_cost']:.4f}")
    print()
    
    # Simulate some usage for demo
    print("🔧 Simulating usage...")
    
    # Simulate LLM call with token usage
    pipeline.usage_tracker.record_llm_call({
        'usage': {'prompt_tokens': 1500, 'completion_tokens': 300}
    })
    
    # Simulate DI call with 5 pages
    pipeline.usage_tracker.record_di_call(5)
    
    # Simulate rule hits
    pipeline.usage_tracker.record_rules_hit(8)
    
    # Simulate processing time
    pipeline.usage_tracker.record_document_time(3.2)
    
    print("  ✅ Simulated 1 LLM call (1500 input + 300 output tokens)")
    print("  ✅ Simulated 1 DI call (5 pages)")
    print("  ✅ Simulated 8 rule field hits")
    print("  ✅ Simulated 3.2s processing time")
    print()
    
    # Show updated usage
    final_usage = pipeline.get_usage_summary()
    print("📈 Final Usage Summary:")
    print(f"  Documents: {final_usage['documents_processed']}")
    print(f"  LLM Calls: {final_usage['llm_calls']}")
    print(f"  DI Calls: {final_usage['di_calls']}")
    print(f"  Rule Fields: {final_usage['rule_field_hits']}")
    print(f"  Input Tokens: {final_usage['input_tokens']:,}")
    print(f"  Output Tokens: {final_usage['output_tokens']:,}")
    print(f"  DI Pages: {final_usage['di_pages_processed']}")
    print(f"  Avg Time/Doc: {final_usage['avg_processing_seconds']:.2f}s")
    print(f"  AI Usage Rate: {final_usage['ai_doc_ratio']:.0%}")
    print()
    
    # Show cost breakdown
    cost = final_usage['cost']
    print("💰 Cost Breakdown:")
    print(f"  LLM Cost:   ${cost['llm_cost']:.4f}")
    print(f"    Input:    ${(1500/1000) * 0.005:.4f} (1.5K tokens × $0.005)")
    print(f"    Output:   ${(300/1000) * 0.010:.4f} (0.3K tokens × $0.010)")
    print(f"  DI Cost:    ${cost['di_cost']:.4f} (5 pages × $0.020)")
    print(f"  Total Cost: ${cost['total_cost']:.4f}")
    print()
    
    # Test processing a real document
    test_doc_path = "test_documents/sample_invoice.txt"
    if os.path.exists(test_doc_path):
        print("📄 Processing real document...")
        pipeline.usage_tracker.reset()  # Reset for clean demo
        
        document, log = pipeline.process_document(test_doc_path)
        
        real_usage = pipeline.get_usage_summary()
        print("  📈 Real Document Processing Results:")
        print(f"    Documents: {real_usage['documents_processed']}")
        print(f"    Processing Time: {real_usage['total_processing_seconds']:.3f}s")
        print(f"    Fields Extracted: {len(document.key_values)}")
        print(f"    Extraction Method: {document.key_values[0].extraction_method if document.key_values else 'None'}")
        
        # Show what would happen with real AI calls (simulated)
        if real_usage['llm_calls'] == 0:
            print()
            print("  💡 In a real scenario with Azure OpenAI/DI configured:")
            print("    - LLM would be called for field extraction")
            print("    - Token usage would be captured automatically")
            print("    - Costs would be calculated in real-time")
            print("    - UI would update with live metrics")
    
    print()
    print("✅ Usage tracking demo complete!")
    print("💡 The feature is fully implemented and ready for production use.")


if __name__ == "__main__":
    demo_usage_tracking()