"""
Simple rule-based extraction engine using regex patterns.
"""

import re
import yaml
import logging
from typing import List, Dict, Any, Tuple
import os

from .schema import KeyValue

logger = logging.getLogger(__name__)


class RulesEngine:
    """Simple rule-based field extraction engine."""
    
    def __init__(self, rules_dir: str = "rules"):
        self.rules_dir = rules_dir
        self.global_rules = []
        self.signature_rules = {}
        self.load_rules()
    
    def load_rules(self):
        """Load rules from YAML files."""
        self.global_rules = []
        self.signature_rules = {}
        
        # Load global rules
        global_rules_path = os.path.join(self.rules_dir, "global_rules.yml")
        if os.path.exists(global_rules_path):
            try:
                with open(global_rules_path, 'r') as f:
                    rules_config = yaml.safe_load(f)
                    self.global_rules = rules_config.get('rules', [])
                logger.info(f"Loaded {len(self.global_rules)} global rules")
            except Exception as e:
                logger.error(f"Error loading global rules: {e}")
        
        # Load signature-specific rules
        signature_rules_dir = os.path.join(self.rules_dir, "signature_overrides")
        if os.path.exists(signature_rules_dir):
            for filename in os.listdir(signature_rules_dir):
                if filename.endswith('.yml'):
                    signature_id = filename[:-4]  # Remove .yml extension
                    try:
                        with open(os.path.join(signature_rules_dir, filename), 'r') as f:
                            rules_config = yaml.safe_load(f)
                            self.signature_rules[signature_id] = rules_config.get('rules', [])
                    except Exception as e:
                        logger.error(f"Error loading signature rules for {signature_id}: {e}")
    
    def apply_rules(self, text: str, signature_id: str = None) -> Tuple[List[KeyValue], List[str]]:
        """
        Apply extraction rules to text.
        
        Returns:
            Tuple of (extracted_key_values, rules_applied)
        """
        extracted_values = []
        rules_applied = []
        
        # Apply global rules
        if self.global_rules:
            for rule in self.global_rules:
                result = self._apply_single_rule(rule, text)
                if result:
                    extracted_values.append(result)
            rules_applied.append("global")
        
        # Apply signature-specific rules if available
        if signature_id and signature_id in self.signature_rules:
            for rule in self.signature_rules[signature_id]:
                result = self._apply_single_rule(rule, text)
                if result:
                    extracted_values.append(result)
            rules_applied.append(f"signature_{signature_id}")
        
        return extracted_values, rules_applied
    
    def _apply_single_rule(self, rule: Dict[str, Any], text: str) -> KeyValue:
        """Apply a single extraction rule."""
        try:
            field_name = rule.get('field_name')
            pattern = rule.get('pattern')
            confidence = rule.get('confidence', 0.5)
            
            if not field_name or not pattern:
                return None
            
            # Apply regex pattern
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Get the first capturing group or the whole match
                value = match.group(1) if match.groups() else match.group(0)
                value = value.strip()
                
                # Try to convert to appropriate type
                value = self._convert_value(value, field_name)
                
                return KeyValue(
                    key=field_name,
                    value=value,
                    confidence=confidence,
                    extraction_method="rule"
                )
        
        except Exception as e:
            logger.warning(f"Error applying rule for {rule.get('field_name')}: {e}")
        
        return None
    
    def _convert_value(self, value: str, field_name: str) -> Any:
        """Convert extracted value to appropriate type."""
        # Try to convert numeric values
        if 'amount' in field_name.lower() or 'cost' in field_name.lower() or 'price' in field_name.lower():
            # Remove currency symbols and commas
            numeric_value = re.sub(r'[$,]', '', value)
            try:
                if '.' in numeric_value:
                    return float(numeric_value)
                else:
                    return int(numeric_value)
            except ValueError:
                pass
        
        return value
    
    def get_required_fields(self, signature_id: str = None) -> List[str]:
        """Get list of required fields for extraction."""
        required_fields = []
        
        # Get required fields from global rules
        for rule in self.global_rules:
            if rule.get('required', False):
                required_fields.append(rule['field_name'])
        
        # Get required fields from signature rules
        if signature_id and signature_id in self.signature_rules:
            for rule in self.signature_rules[signature_id]:
                if rule.get('required', False):
                    required_fields.append(rule['field_name'])
        
        return required_fields
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rules engine statistics."""
        return {
            "total_rules": len(self.global_rules),
            "signature_rule_sets": len(self.signature_rules),
            "global_required_fields": len([r for r in self.global_rules if r.get('required')]),
        }