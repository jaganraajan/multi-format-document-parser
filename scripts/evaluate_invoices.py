#!/usr/bin/env python3
"""
Invoice Evaluation Script

Compares parsed invoice outputs against ground truth JSON to compute
precision and recall metrics for field extraction evaluation.

Author: Multi-Format Document Parser
License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class InvoiceEvaluator:
    """Evaluates invoice parsing accuracy against ground truth."""
    
    def __init__(self, schema_file: Path):
        """Initialize evaluator with field schema."""
        self.schema = self._load_schema(schema_file)
        self.flat_fields = set(self.schema.get("flat_fields", {}).keys())
    
    def _load_schema(self, schema_file: Path) -> Dict:
        """Load schema definition from JSON file."""
        if not schema_file.exists():
            print(f"⚠️  Schema file not found: {schema_file}")
            print("📝 Using default field set...")
            return {"flat_fields": {}}
        
        try:
            return json.loads(schema_file.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"❌ Failed to load schema: {e}")
            return {"flat_fields": {}}
    
    def _get_invoice_basenames(self, ground_truth_dir: Path, parsed_dir: Path) -> List[str]:
        """Get list of invoice basenames present in both directories."""
        gt_files = {f.stem for f in ground_truth_dir.glob("*.json")}
        parsed_files = {f.stem for f in parsed_dir.glob("*.json")}
        
        common_files = gt_files & parsed_files
        
        if not common_files:
            print("⚠️  No matching files found between ground truth and parsed directories")
            print(f"📁 Ground truth files: {len(gt_files)}")
            print(f"📁 Parsed files: {len(parsed_files)}")
        
        return sorted(list(common_files))
    
    def _extract_flat_fields(self, data: Dict) -> Dict[str, str]:
        """Extract flat fields from invoice data, converting all to strings."""
        extracted = {}
        
        for field in self.flat_fields:
            if field in data:
                value = data[field]
                # Convert to string for comparison, handle None/empty values
                if value is None or value == "":
                    extracted[field] = ""
                else:
                    extracted[field] = str(value).strip()
        
        return extracted
    
    def _compare_fields(self, ground_truth: Dict[str, str], 
                       parsed: Dict[str, str]) -> Tuple[Set[str], Set[str], Set[str]]:
        """Compare field sets and return exact matches, missing, and extra fields."""
        gt_set = set(ground_truth.keys())
        parsed_set = set(parsed.keys())
        
        # Fields that exist in both and have exact matches
        exact_matches = set()
        for field in gt_set & parsed_set:
            if ground_truth[field] == parsed[field]:
                exact_matches.add(field)
        
        # Fields in ground truth but missing from parsed output
        missing_fields = gt_set - parsed_set
        
        # Fields in parsed output but not in ground truth
        extra_fields = parsed_set - gt_set
        
        return exact_matches, missing_fields, extra_fields
    
    def evaluate_single_invoice(self, gt_file: Path, parsed_file: Path) -> Dict:
        """Evaluate a single invoice against ground truth."""
        try:
            # Load ground truth
            gt_data = json.loads(gt_file.read_text(encoding='utf-8'))
            gt_fields = self._extract_flat_fields(gt_data)
            
            # Load parsed output
            parsed_data = json.loads(parsed_file.read_text(encoding='utf-8'))
            parsed_fields = self._extract_flat_fields(parsed_data)
            
            # Compare fields
            exact_matches, missing_fields, extra_fields = self._compare_fields(
                gt_fields, parsed_fields
            )
            
            # Calculate metrics
            total_gt_fields = len(gt_fields)
            total_parsed_fields = len(parsed_fields)
            
            precision = len(exact_matches) / total_parsed_fields if total_parsed_fields > 0 else 0.0
            recall = len(exact_matches) / total_gt_fields if total_gt_fields > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "exact_matches": exact_matches,
                "missing_fields": missing_fields,
                "extra_fields": extra_fields,
                "total_gt_fields": total_gt_fields,
                "total_parsed_fields": total_parsed_fields,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "gt_fields": gt_fields,
                "parsed_fields": parsed_fields
            }
            
        except Exception as e:
            print(f"❌ Failed to evaluate {gt_file.stem}: {e}")
            return None
    
    def evaluate_batch(self, ground_truth_dir: Path, parsed_dir: Path, 
                      limit: Optional[int] = None) -> Dict:
        """Evaluate multiple invoices and compute aggregate metrics."""
        
        basenames = self._get_invoice_basenames(ground_truth_dir, parsed_dir)
        
        if limit:
            basenames = basenames[:limit]
        
        if not basenames:
            return {
                "total_invoices": 0,
                "successful_evaluations": 0,
                "aggregate_precision": 0.0,
                "aggregate_recall": 0.0,
                "aggregate_f1_score": 0.0,
                "field_level_stats": {}
            }
        
        print(f"📊 Evaluating {len(basenames)} invoices...")
        
        results = []
        successful_count = 0
        
        # Aggregate field-level statistics
        field_stats = {}
        
        for basename in basenames:
            gt_file = ground_truth_dir / f"{basename}.json"
            parsed_file = parsed_dir / f"{basename}.json"
            
            result = self.evaluate_single_invoice(gt_file, parsed_file)
            
            if result:
                results.append(result)
                successful_count += 1
                
                # Update field-level statistics
                for field in self.flat_fields:
                    if field not in field_stats:
                        field_stats[field] = {"present_in_gt": 0, "correctly_extracted": 0}
                    
                    if field in result["gt_fields"]:
                        field_stats[field]["present_in_gt"] += 1
                        
                        if field in result["exact_matches"]:
                            field_stats[field]["correctly_extracted"] += 1
        
        if not results:
            print("❌ No successful evaluations")
            return {
                "total_invoices": len(basenames),
                "successful_evaluations": 0,
                "aggregate_precision": 0.0,
                "aggregate_recall": 0.0,
                "aggregate_f1_score": 0.0,
                "field_level_stats": {}
            }
        
        # Calculate aggregate metrics
        total_precision = sum(r["precision"] for r in results)
        total_recall = sum(r["recall"] for r in results)
        total_f1 = sum(r["f1_score"] for r in results)
        
        aggregate_precision = total_precision / len(results)
        aggregate_recall = total_recall / len(results)
        aggregate_f1 = total_f1 / len(results)
        
        # Calculate per-field accuracy
        for field in field_stats:
            if field_stats[field]["present_in_gt"] > 0:
                field_stats[field]["accuracy"] = (
                    field_stats[field]["correctly_extracted"] / 
                    field_stats[field]["present_in_gt"]
                )
            else:
                field_stats[field]["accuracy"] = 0.0
        
        return {
            "total_invoices": len(basenames),
            "successful_evaluations": successful_count,
            "aggregate_precision": aggregate_precision,
            "aggregate_recall": aggregate_recall,
            "aggregate_f1_score": aggregate_f1,
            "field_level_stats": field_stats,
            "individual_results": results
        }


def print_evaluation_results(results: Dict) -> None:
    """Print formatted evaluation results."""
    print("\n" + "="*60)
    print("📊 INVOICE PARSING EVALUATION RESULTS")
    print("="*60)
    
    print(f"📄 Total invoices evaluated: {results['successful_evaluations']}/{results['total_invoices']}")
    
    if results['successful_evaluations'] == 0:
        print("❌ No successful evaluations to report")
        return
    
    print(f"\n📈 Overall Metrics (Flat Fields Only):")
    print(f"   Precision: {results['aggregate_precision']:.3f}")
    print(f"   Recall:    {results['aggregate_recall']:.3f}")
    print(f"   F1 Score:  {results['aggregate_f1_score']:.3f}")
    
    print(f"\n📋 Field-Level Accuracy:")
    field_stats = results["field_level_stats"]
    
    if not field_stats:
        print("   No field statistics available")
        return
    
    # Sort fields by accuracy (descending)
    sorted_fields = sorted(
        field_stats.items(), 
        key=lambda x: x[1].get("accuracy", 0), 
        reverse=True
    )
    
    print(f"   {'Field':<25} {'Accuracy':<10} {'Extracted/Total':<15}")
    print(f"   {'-'*25} {'-'*10} {'-'*15}")
    
    for field, stats in sorted_fields:
        accuracy = stats.get("accuracy", 0)
        extracted = stats.get("correctly_extracted", 0)
        total = stats.get("present_in_gt", 0)
        
        if total > 0:
            print(f"   {field:<25} {accuracy:.3f}      {extracted}/{total}")
    
    # Show some examples of common issues (first few individual results)
    if "individual_results" in results and len(results["individual_results"]) > 0:
        print(f"\n🔍 Sample Issues (First Invoice):")
        first_result = results["individual_results"][0]
        
        if first_result["missing_fields"]:
            missing = list(first_result["missing_fields"])[:5]  # Show first 5
            print(f"   Missing fields: {', '.join(missing)}")
        
        if first_result["extra_fields"]:
            extra = list(first_result["extra_fields"])[:5]  # Show first 5
            print(f"   Extra fields: {', '.join(extra)}")
    
    print(f"\n💡 Note: This evaluation only covers flat fields.")
    print(f"   Line item evaluation is planned for future versions.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate invoice parsing accuracy against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --ground-truth-dir datasets/indian_gst/generated --parsed-dir outputs/
  %(prog)s --limit 50 --fields-file datasets/indian_gst/schema_fields.json
        """
    )
    
    parser.add_argument('--ground-truth-dir', type=Path,
                       default=Path('datasets/indian_gst/generated'),
                       help='Directory containing ground truth JSON files')
    parser.add_argument('--parsed-dir', type=Path,
                       default=Path('outputs'),
                       help='Directory containing parsed output JSON files')
    parser.add_argument('--fields-file', type=Path,
                       default=Path('datasets/indian_gst/schema_fields.json'),
                       help='Schema file defining expected fields')
    parser.add_argument('--limit', type=int,
                       help='Limit evaluation to first N invoices')
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.ground_truth_dir.exists():
        print(f"❌ Ground truth directory not found: {args.ground_truth_dir}")
        print("💡 Generate invoices first with: python scripts/generate_indian_invoices.py")
        sys.exit(1)
    
    if not args.parsed_dir.exists():
        print(f"❌ Parsed output directory not found: {args.parsed_dir}")
        print("💡 Run document parsing pipeline first to generate outputs/")
        sys.exit(1)
    
    try:
        # Initialize evaluator
        evaluator = InvoiceEvaluator(args.fields_file)
        
        # Run evaluation
        results = evaluator.evaluate_batch(
            ground_truth_dir=args.ground_truth_dir,
            parsed_dir=args.parsed_dir,
            limit=args.limit
        )
        
        # Print results
        print_evaluation_results(results)
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()