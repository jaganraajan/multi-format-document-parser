#!/usr/bin/env python3
"""
Example showing how to use the new usage tracking API.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalization.pipeline import DocumentPipeline


def main():
    """Example usage of the new usage tracking features."""
    print("📄 Multi-Format Document Parser - Usage Tracking Example")
    print("=" * 60)
    
    # Initialize pipeline with usage tracking
    pipeline = DocumentPipeline(enable_llm=True, enable_di=True)
    
    # Process some documents (using existing test doc)
    test_doc = "test_documents/sample_invoice.txt"
    if os.path.exists(test_doc):
        print(f"Processing: {test_doc}")
        document, log = pipeline.process_document(test_doc)
        print(f"✅ Processed document: {document.doc_id}")
        print(f"   Fields extracted: {len(document.key_values)}")
        print()
    
    # Get usage summary using the new API
    usage = pipeline.get_usage_summary()
    
    print("📊 Usage Summary (via pipeline.get_usage_summary()):")
    print(f"  Documents Processed: {usage['documents_processed']}")
    print(f"  LLM Calls: {usage['llm_calls']}")
    print(f"  DI Calls: {usage['di_calls']}")
    print(f"  Rule Field Hits: {usage['rule_field_hits']}")
    print(f"  Input Tokens: {usage['input_tokens']:,}")
    print(f"  Output Tokens: {usage['output_tokens']:,}")
    print(f"  DI Pages Processed: {usage['di_pages_processed']}")
    print(f"  Total Processing Time: {usage['total_processing_seconds']:.3f}s")
    print()
    
    print("📈 Computed Metrics:")
    print(f"  Avg Processing Time: {usage['avg_processing_seconds']:.3f}s per doc")
    print(f"  Avg Rule Fields: {usage['avg_rule_fields_per_doc']:.1f} per doc")
    print(f"  AI Usage Ratio: {usage['ai_doc_ratio']:.1%}")
    print()
    
    print("💰 Cost Estimates:")
    cost = usage['cost']
    print(f"  LLM Cost: ${cost['llm_cost']:.4f}")
    print(f"  DI Cost: ${cost['di_cost']:.4f}")
    print(f"  Total Cost: ${cost['total_cost']:.4f}")
    print()
    
    print("⚙️ Pricing Configuration (override with env vars):")
    print(f"  COST_LLM_INPUT_PER_1K: ${pipeline.usage_tracker.llm_input_cost_per_1k:.4f}")
    print(f"  COST_LLM_OUTPUT_PER_1K: ${pipeline.usage_tracker.llm_output_cost_per_1k:.4f}")
    print(f"  COST_DI_PER_PAGE: ${pipeline.usage_tracker.di_page_cost:.4f}")
    print()
    
    print("🔧 Direct Usage Tracker API:")
    print("  pipeline.usage_tracker.record_llm_call(meta)")
    print("  pipeline.usage_tracker.record_di_call(pages)")
    print("  pipeline.usage_tracker.record_rules_hit(count)")
    print("  pipeline.usage_tracker.record_document_time(seconds)")
    print("  pipeline.usage_tracker.snapshot()")
    print("  pipeline.usage_tracker.reset()")
    print()
    
    print("🎯 Key Benefits:")
    print("  ✅ Track cost drivers (model calls, tokens, DI pages)")
    print("  ✅ Verify hybrid gating strategy effectiveness")
    print("  ✅ Monitor rule coverage improvements")
    print("  ✅ Share transparent operational metrics")
    print("  ✅ Real-time cost estimation")
    print("  ✅ Streamlit UI integration")
    print()
    
    print("✨ The usage tracking feature is fully implemented and ready!")


if __name__ == "__main__":
    main()