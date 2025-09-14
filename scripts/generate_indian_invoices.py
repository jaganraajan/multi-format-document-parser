#!/usr/bin/env python3
"""
Synthetic Indian GST Invoice Generator with Sender Profile Variations

Generates realistic but completely synthetic Indian GST tax invoices with
per-sender layout variations and controlled anomalies. This enables hybrid
pipeline testing where:

1. 95% of documents use rule-friendly layouts with consistent per-sender 
   formatting (font, label vocabulary, field ordering) → should converge 
   to stable signatures & high rule coverage.

2. 5% exhibit anomalous layouts (renamed labels, relocated totals, missing 
   labels, noisy prefixes) → should reduce rule confidence and trigger 
   selective LLM fallback.

The variation system uses SenderProfile objects with quirk levels:
- Level 0-1: Normal, rule-friendly layouts  
- Level 2-3: Anomalous layouts requiring AI assistance

Key Features:
- 8 predefined sender profiles with distinct styling
- Configurable anomaly injection (--variation-ratio)
- Deterministic generation with seeds
- Enhanced JSON metadata for signature learning
- Backward compatible with existing pipeline

Usage Examples:
  # Default 5% anomaly rate
  python generate_indian_invoices.py --count 50
  
  # No anomalies (pure rule testing)  
  python generate_indian_invoices.py --count 100 --variation-ratio 0
  
  # High anomaly rate (LLM fallback testing)
  python generate_indian_invoices.py --count 20 --variation-ratio 0.3

Author: Multi-Format Document Parser
License: MIT
"""

import argparse
import json
import random
import string
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from faker import Faker
    from jinja2 import Template
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install faker jinja2")
    sys.exit(1)

def load_weasyprint() -> Tuple[Optional[Any], Optional[str]]:
    """Attempt to import weasyprint lazily.

    Returns (module, error_message). If module is None, PDF support is disabled.
    Catches both ImportError (package not installed) and OSError (system libs missing).
    """
    try:
        import weasyprint  # type: ignore
        return weasyprint, None
    except ImportError as e:
        return None, (
            "WeasyPrint is not installed. Install with: pip install weasyprint\n"
            "macOS system libs (Homebrew): brew install cairo pango gdk-pixbuf libffi\n"
            "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 libffi-dev"
        )
    except OSError as e:  # Missing shared libraries (e.g., libgobject-2.0)
        return None, (
            f"WeasyPrint import failed due to missing system libraries: {e}\n"
            "Install required native deps.\n"
            "macOS (Homebrew): brew install cairo pango gdk-pixbuf libffi\n"
            "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 libffi-dev\n"
            "Then: pip install --upgrade weasyprint"
        )


@dataclass
class SenderProfile:
    """Profile defining sender-specific layout and styling quirks for invoice generation.
    
    This enables the hybrid pipeline to learn layout signatures for consistent senders
    while introducing controlled anomalies to test LLM fallback logic.
    """
    sender_id: str
    font_family: str
    color_primary: str
    invoice_label_variants: List[str]
    field_label_overrides: Dict[str, str]
    layout_style: str
    quirk_level: int  # 0-3, higher means more aggressive formatting drift


def _build_sender_profiles() -> List[SenderProfile]:
    """Build catalog of sender profiles with varying layout characteristics."""
    return [
        SenderProfile(
            sender_id="ACME_CORP",
            font_family="Arial, sans-serif",
            color_primary="#2c3e50",
            invoice_label_variants=["Tax Invoice", "Invoice", "Invoice #"],
            field_label_overrides={},
            layout_style="standard",
            quirk_level=0
        ),
        SenderProfile(
            sender_id="TECH_SOLUTIONS",
            font_family="Helvetica, sans-serif", 
            color_primary="#3498db",
            invoice_label_variants=["Commercial Invoice", "Invoice", "Tax Invoice"],
            field_label_overrides={"invoice_number": "Invoice Ref"},
            layout_style="compact",
            quirk_level=1
        ),
        SenderProfile(
            sender_id="GLOBAL_SERVICES",
            font_family="Times, serif",
            color_primary="#e74c3c",
            invoice_label_variants=["Invoice No.", "Invoice", "Commercial Invoice"],
            field_label_overrides={"invoice_date": "Date"},
            layout_style="totals-top",
            quirk_level=0
        ),
        SenderProfile(
            sender_id="INNOVATIVE_TECH",
            font_family="Verdana, sans-serif",
            color_primary="#9b59b6",
            invoice_label_variants=["Tax Invoice", "Invoice #", "Invoice"],
            field_label_overrides={"total_amount": "Final Amount"},
            layout_style="lines-condensed",
            quirk_level=1
        ),
        SenderProfile(
            sender_id="PREMIUM_CONSULTANTS",
            font_family="Georgia, serif",
            color_primary="#f39c12", 
            invoice_label_variants=["Professional Invoice", "Invoice", "Service Invoice"],
            field_label_overrides={"place_of_supply": "Delivery Location"},
            layout_style="standard",
            quirk_level=2
        ),
        SenderProfile(
            sender_id="DYNAMIC_ENTERPRISES",
            font_family="Calibri, sans-serif",
            color_primary="#27ae60",
            invoice_label_variants=["Invoice", "Commercial Invoice", "Tax Invoice"],
            field_label_overrides={"reverse_charge": "Reverse Charge Applicable"},
            layout_style="compact",
            quirk_level=1
        ),
        SenderProfile(
            sender_id="QUIRKY_CORP",
            font_family="Comic Sans MS, cursive",
            color_primary="#e67e22",
            invoice_label_variants=["*** Invoice ***", "Invoice!", "Our Invoice #"],
            field_label_overrides={"invoice_number": "Ref #", "invoice_date": "Bill Date"},
            layout_style="totals-top",
            quirk_level=3
        ),
        SenderProfile(
            sender_id="LEGACY_SYSTEMS",
            font_family="Courier, monospace",
            color_primary="#34495e",
            invoice_label_variants=["INVOICE DOCUMENT", "Invoice #", "TAX INVOICE"],
            field_label_overrides={"supplier_name": "Vendor", "recipient_name": "Customer"},
            layout_style="lines-condensed", 
            quirk_level=2
        )
    ]


def _pick_sender_profile(variation_ratio: float, sender_count: int, sender_profiles: List[SenderProfile]) -> Tuple[SenderProfile, bool]:
    """Pick a sender profile based on variation ratio.
    
    Returns (profile, requires_ai_hint) where requires_ai_hint indicates
    if this should trigger LLM fallback due to anomalies.
    """
    # Limit to sender_count profiles
    available_profiles = sender_profiles[:sender_count]
    
    # Decide if this should be an anomalous case
    is_anomalous = random.random() < variation_ratio
    
    if is_anomalous:
        # Pick profiles with higher quirk levels or mutate normal profiles
        high_quirk_profiles = [p for p in available_profiles if p.quirk_level >= 2]
        if high_quirk_profiles:
            profile = random.choice(high_quirk_profiles)
        else:
            # Mutate a normal profile on the fly for anomalous behavior
            base_profile = random.choice(available_profiles)
            profile = SenderProfile(
                sender_id=base_profile.sender_id,
                font_family=base_profile.font_family,
                color_primary=base_profile.color_primary,
                invoice_label_variants=base_profile.invoice_label_variants,
                field_label_overrides=base_profile.field_label_overrides.copy(),
                layout_style=base_profile.layout_style,
                quirk_level=3  # Force high quirk for anomalous behavior
            )
        return profile, True
    else:
        # Pick normal profiles (quirk_level 0-1)
        normal_profiles = [p for p in available_profiles if p.quirk_level <= 1]
        if normal_profiles:
            profile = random.choice(normal_profiles)
        else:
            profile = random.choice(available_profiles)
        return profile, False


def _apply_anomalies(invoice_data: Dict, profile: SenderProfile, requires_ai_hint: bool) -> Tuple[Dict, List[str]]:
    """Apply layout anomalies based on profile quirk level and anomaly flag.
    
    Returns (modified_invoice_data, list_of_anomaly_descriptions).
    """
    anomalies = []
    
    if not requires_ai_hint and profile.quirk_level <= 1:
        return invoice_data, anomalies
    
    # Apply anomalies based on quirk level
    if profile.quirk_level >= 2 or requires_ai_hint:
        # Label renaming anomalies
        if random.random() < 0.5:
            # Rename invoice number label to something non-standard
            anomalies.append("invoice_number_label_renamed")
            
        if random.random() < 0.3:
            # Use sentence-embedded format for invoice number
            anomalies.append("invoice_number_sentence_format")
            
        if random.random() < 0.4:
            # Add noise wrappers around labels
            anomalies.append("label_noise_wrappers")
            
        if random.random() < 0.2:
            # Omit a field label while keeping value
            anomalies.append("field_label_omission")
            
        if random.random() < 0.3:
            # Reorder header blocks
            anomalies.append("header_block_reorder")
            
        if profile.layout_style == "totals-top" or random.random() < 0.2:
            # Move totals above line items
            anomalies.append("totals_above_items")
    
    return invoice_data, anomalies


def _construct_template_params(invoice_data: Dict, profile: SenderProfile, anomalies: List[str]) -> Dict:
    """Construct template parameters including style and anomaly-specific rendering flags."""
    
    # Select invoice label variant
    invoice_label_used = random.choice(profile.invoice_label_variants)
    
    # Apply field label overrides
    field_labels = {
        "invoice_number": "Invoice Number:",
        "invoice_date": "Invoice Date:",
        "total_amount": "Total Amount:",
        "place_of_supply": "Place of Supply:",
        "reverse_charge": "Reverse Charge:",
        "supplier_name": "From:",
        "recipient_name": "To:"
    }
    
    # Apply profile overrides
    for field, override_label in profile.field_label_overrides.items():
        if field in field_labels:
            field_labels[field] = f"{override_label}:"
    
    # Apply anomaly modifications
    if "invoice_number_label_renamed" in anomalies:
        field_labels["invoice_number"] = random.choice(["Ref #:", "Our Ref:", "Doc No:"])
    
    if "invoice_number_sentence_format" in anomalies:
        field_labels["invoice_number"] = "Our Reference:"
    
    if "label_noise_wrappers" in anomalies:
        invoice_label_used = f"*** {invoice_label_used} ***"
        
    if "field_label_omission" in anomalies:
        # Randomly omit a label
        omit_field = random.choice(["invoice_date", "reverse_charge"])
        field_labels[omit_field] = ""
    
    # Construct template parameters
    template_params = invoice_data.copy()
    template_params.update({
        "invoice_label_used": invoice_label_used,
        "field_labels": field_labels,
        "font_family": profile.font_family,
        "color_primary": profile.color_primary,
        "layout_style": profile.layout_style,
        "anomalies": anomalies,
        "sender_id": profile.sender_id
    })
    
    return template_params


class IndianGSTInvoiceGenerator:
    """Generator for synthetic Indian GST invoices."""
    
    # Indian state codes and names for GST
    INDIAN_STATES = {
        "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
        "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", 
        "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
        "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
        "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
        "16": "Tripura", "17": "Meghalaya", "18": "Assam",
        "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
        "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
        "25": "Daman and Diu", "26": "Dadra and Nagar Haveli",
        "27": "Maharashtra", "28": "Andhra Pradesh", "29": "Karnataka",
        "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
        "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman and Nicobar Islands",
        "36": "Telangana", "37": "Andhra Pradesh (New)"
    }
    
    # Common GST rates
    GST_RATES = [5, 12, 18]
    
    # Sample products/services for invoice line items
    SAMPLE_PRODUCTS = [
        "Laptop Computer", "Office Chair", "Printer Cartridge", "Mobile Phone",
        "Software License", "Consulting Services", "Marketing Services", 
        "Web Development", "Stationery Items", "Computer Mouse",
        "Keyboard", "Monitor", "Network Cable", "Router", "Switch",
        "Server Maintenance", "Cloud Storage", "Domain Registration",
        "SSL Certificate", "Database License"
    ]
    
    # Sample HSN/SAC codes
    SAMPLE_HSN_SAC = [
        "8471", "9403", "8443", "8517", "9954", "9983", "9984",
        "9985", "4802", "8471", "8471", "8528", "8544", "8517",
        "8517", "9954", "9954", "9954", "9954", "9954"
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        
        self.fake = Faker('en_IN')  # Indian locale
    
    def _generate_pan(self) -> str:
        """Generate a fake PAN (Permanent Account Number) in correct format."""
        letters = ''.join(random.choices(string.ascii_uppercase, k=5))
        digits = ''.join(random.choices(string.digits, k=4))
        check_letter = random.choice(string.ascii_uppercase)
        return f"{letters}{digits}{check_letter}"
    
    def _calculate_checksum(self, gstin_prefix: str) -> str:
        """Calculate GSTIN checksum digit using simple algorithm."""
        # Simplified checksum calculation for demo purposes
        total = sum(ord(char) for char in gstin_prefix)
        return str(total % 10)
    
    def _generate_gstin(self, state_code: str) -> str:
        """Generate a synthetic GSTIN with proper format."""
        pan = self._generate_pan()
        entity_number = random.choice("123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        gstin_prefix = f"{state_code}{pan}{entity_number}Z"
        checksum = self._calculate_checksum(gstin_prefix)
        return f"{gstin_prefix}{checksum}"
    
    def _round_currency(self, value: Decimal) -> float:
        """Round currency value to 2 decimal places."""
        return float(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    
    def _generate_line_items(self, max_items: int, is_intra_state: bool) -> List[Dict]:
        """Generate invoice line items with tax calculations."""
        num_items = random.randint(1, max_items)
        line_items = []
        
        for _ in range(num_items):
            # Basic line item data
            description = random.choice(self.SAMPLE_PRODUCTS)
            hsn_sac = random.choice(self.SAMPLE_HSN_SAC)
            quantity = random.randint(1, 10)
            unit_price = Decimal(str(random.uniform(100, 5000))).quantize(Decimal('0.01'))
            taxable_value = quantity * unit_price
            
            # GST calculations
            gst_rate = random.choice(self.GST_RATES)
            
            line_item = {
                "description": description,
                "hsn_sac": hsn_sac,
                "quantity": quantity,
                "unit_price": self._round_currency(unit_price),
                "taxable_value": self._round_currency(taxable_value)
            }
            
            if is_intra_state:
                # Split GST into CGST and SGST
                cgst_rate = sgst_rate = gst_rate / 2
                cgst_amount = taxable_value * Decimal(str(cgst_rate / 100))
                sgst_amount = taxable_value * Decimal(str(sgst_rate / 100))
                
                line_item.update({
                    "cgst_rate": cgst_rate,
                    "cgst_amount": self._round_currency(cgst_amount),
                    "sgst_rate": sgst_rate,
                    "sgst_amount": self._round_currency(sgst_amount),
                    "igst_rate": None,
                    "igst_amount": None
                })
                
                line_total = taxable_value + cgst_amount + sgst_amount
            else:
                # Use IGST for inter-state
                igst_rate = gst_rate
                igst_amount = taxable_value * Decimal(str(igst_rate / 100))
                
                line_item.update({
                    "cgst_rate": None,
                    "cgst_amount": None,
                    "sgst_rate": None,
                    "sgst_amount": None,
                    "igst_rate": igst_rate,
                    "igst_amount": self._round_currency(igst_amount)
                })
                
                line_total = taxable_value + igst_amount
            
            line_item["line_total"] = self._round_currency(line_total)
            line_items.append(line_item)
        
        return line_items
    
    def _calculate_totals(self, line_items: List[Dict]) -> Tuple[float, float, float]:
        """Calculate invoice totals from line items."""
        taxable_total = sum(item["taxable_value"] for item in line_items)
        
        total_tax = 0
        for item in line_items:
            if item["cgst_amount"] is not None:
                total_tax += item["cgst_amount"] + item["sgst_amount"]
            else:
                total_tax += item["igst_amount"]
        
        total_amount = taxable_total + total_tax
        
        return (
            self._round_currency(Decimal(str(taxable_total))),
            self._round_currency(Decimal(str(total_tax))),
            self._round_currency(Decimal(str(total_amount)))
        )
    
    def generate_invoice_data(self, max_items: int = 4, sender_profile: Optional[SenderProfile] = None, 
                             anomalies: Optional[List[str]] = None, requires_ai_hint: bool = False) -> Dict:
        """Generate complete synthetic invoice data with optional sender profile metadata."""
        # Choose random states for supplier and recipient
        supplier_state_code = random.choice(list(self.INDIAN_STATES.keys()))
        
        # 50/50 chance for intra-state vs inter-state transaction
        is_intra_state = random.choice([True, False])
        if is_intra_state:
            recipient_state_code = supplier_state_code
        else:
            recipient_state_code = random.choice([
                code for code in self.INDIAN_STATES.keys() 
                if code != supplier_state_code
            ])
        
        # Generate invoice basic data
        invoice_data = {
            "invoice_number": f"INV-{datetime.now().year}-{random.randint(1000, 9999)}",
            "invoice_date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            "supplier_name": self.fake.company(),
            "supplier_gstin": self._generate_gstin(supplier_state_code),
            "supplier_address": self.fake.address().replace('\n', ', '),
            "recipient_name": self.fake.company(),
            "recipient_gstin": self._generate_gstin(recipient_state_code),
            "recipient_address": self.fake.address().replace('\n', ', '),
            "place_of_supply": self.INDIAN_STATES[recipient_state_code],
            "reverse_charge": random.choice(["No", "No", "No", "Yes"]),  # Mostly "No"
        }
        
        # Add optional fields (sometimes blank)
        if random.choice([True, False]):
            invoice_data["phone_number"] = self.fake.phone_number()
        else:
            invoice_data["phone_number"] = ""
            
        if random.choice([True, False]):
            invoice_data["email"] = self.fake.company_email()
        else:
            invoice_data["email"] = ""
        
        # Generate line items
        line_items = self._generate_line_items(max_items, is_intra_state)
        invoice_data["line_items"] = line_items
        
        # Calculate totals
        taxable_total, total_tax, total_amount = self._calculate_totals(line_items)
        invoice_data.update({
            "taxable_total": taxable_total,
            "total_tax": total_tax,
            "total_amount": total_amount
        })
        
        # Sometimes add amount in words
        if random.choice([True, False, False]):  # 33% chance
            invoice_data["total_amount_in_words"] = f"Rupees {self.fake.word().title()} Thousand Only"
        else:
            invoice_data["total_amount_in_words"] = ""
        
        # Add sender profile metadata if provided
        if sender_profile:
            invoice_label_used = random.choice(sender_profile.invoice_label_variants)
            invoice_data.update({
                "sender_id": sender_profile.sender_id,
                "layout_variant": sender_profile.layout_style,
                "invoice_label_used": invoice_label_used,
                "anomalies": anomalies or [],
                "requires_ai_hint": requires_ai_hint
            })
        
        return invoice_data


def create_html_template() -> str:
    """Create Jinja2 HTML template for invoice rendering with dynamic styling and layout options.
    
    Template supports:
    - Dynamic font families and colors via sender profiles
    - Conditional layout styles (standard, compact, totals-top, lines-condensed)
    - Variable invoice labels instead of hardcoded "TAX INVOICE"
    - Anomaly-specific rendering (noise wrappers, field omissions, reordering)
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ invoice_label_used }} - {{ invoice_number }}</title>
    <style>
        body { 
            font-family: {{ font_family or 'Arial, sans-serif' }}; 
            margin: 20px; 
            font-size: {% if layout_style == 'compact' %}10px{% elif layout_style == 'lines-condensed' %}11px{% else %}12px{% endif %}; 
        }
        .header { 
            text-align: center; 
            margin-bottom: {% if layout_style == 'compact' %}10px{% else %}20px{% endif %}; 
            color: {{ color_primary or '#000000' }};
        }
        .invoice-title { 
            font-size: {% if layout_style == 'compact' %}16px{% else %}18px{% endif %}; 
            font-weight: bold; 
        }
        .company-details { 
            margin: {% if layout_style == 'compact' %}10px 0{% else %}15px 0{% endif %}; 
        }
        .invoice-details { 
            margin: {% if layout_style == 'compact' %}10px 0{% else %}15px 0{% endif %}; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin: {% if layout_style == 'compact' %}5px 0{% else %}10px 0{% endif %}; 
        }
        th, td { 
            border: 1px solid #ccc; 
            padding: {% if layout_style == 'compact' %}4px{% elif layout_style == 'lines-condensed' %}6px{% else %}8px{% endif %}; 
            text-align: left; 
        }
        th { 
            background-color: #f5f5f5; 
            font-weight: bold; 
        }
        .text-right { text-align: right; }
        .text-center { text-align: center; }
        .totals { 
            margin-top: {% if layout_style == 'compact' %}10px{% else %}20px{% endif %}; 
        }
        .grid { display: flex; gap: 20px; }
        .grid-item { flex: 1; }
        .label { font-weight: bold; }
        .wrapper-spacing { 
            margin: {% if layout_style == 'compact' %}5px 0{% else %}10px 0{% endif %}; 
        }
        {% if 'header_block_reorder' in (anomalies or []) %}
        .header-reordered .grid { flex-direction: row-reverse; }
        {% endif %}
    </style>
</head>
<body>
    <div class="header">
        <div class="invoice-title">{{ invoice_label_used or 'TAX INVOICE' }}</div>
    </div>
    
    {% if 'totals_above_items' in (anomalies or []) %}
    <!-- Anomaly: Totals above line items -->
    <div class="totals">
        <table style="width: 40%; margin-left: auto;">
            <tr>
                <td class="label">{{ field_labels.get('taxable_total', 'Taxable Total:') }}</td>
                <td class="text-right">₹ {{ "%.2f"|format(taxable_total) }}</td>
            </tr>
            <tr>
                <td class="label">Total Tax:</td>
                <td class="text-right">₹ {{ "%.2f"|format(total_tax) }}</td>
            </tr>
            <tr style="font-weight: bold; background-color: #f5f5f5;">
                <td class="label">{{ field_labels.get('total_amount', 'Total Amount:') }}</td>
                <td class="text-right">₹ {{ "%.2f"|format(total_amount) }}</td>
            </tr>
        </table>
    </div>
    {% endif %}
    
    <div class="{% if 'header_block_reorder' in (anomalies or []) %}header-reordered{% endif %}">
        <div class="grid">
            <div class="grid-item">
                <div class="company-details">
                    <div class="label">{{ field_labels.get('supplier_name', 'From:') }}</div>
                    <div><strong>{{ supplier_name }}</strong></div>
                    <div>GSTIN: {{ supplier_gstin }}</div>
                    <div>{{ supplier_address }}</div>
                    {% if phone_number %}<div>Phone: {{ phone_number }}</div>{% endif %}
                    {% if email %}<div>Email: {{ email }}</div>{% endif %}
                </div>
            </div>
            
            <div class="grid-item">
                <div class="company-details">
                    <div class="label">{{ field_labels.get('recipient_name', 'To:') }}</div>
                    <div><strong>{{ recipient_name }}</strong></div>
                    <div>GSTIN: {{ recipient_gstin }}</div>
                    <div>{{ recipient_address }}</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="invoice-details">
        <table>
            <tr>
                <td class="label">{% if field_labels.get('invoice_number') %}{{ field_labels.invoice_number }}{% else %}Invoice Number:{% endif %}</td>
                <td>{% if 'invoice_number_sentence_format' in (anomalies or []) %}Our Reference: {{ invoice_number }}{% else %}{{ invoice_number }}{% endif %}</td>
                <td class="label">{% if field_labels.get('invoice_date') %}{{ field_labels.invoice_date }}{% else %}Invoice Date:{% endif %}</td>
                <td>{{ invoice_date }}</td>
            </tr>
            <tr>
                <td class="label">{% if field_labels.get('place_of_supply') %}{{ field_labels.place_of_supply }}{% else %}Place of Supply:{% endif %}</td>
                <td>{{ place_of_supply }}</td>
                <td class="label">{% if field_labels.get('reverse_charge') %}{{ field_labels.reverse_charge }}{% else %}Reverse Charge:{% endif %}</td>
                <td>{{ reverse_charge }}</td>
            </tr>
        </table>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>S.No</th>
                <th>Description</th>
                <th>HSN/SAC</th>
                <th>Qty</th>
                <th>Unit Price</th>
                <th>Taxable Value</th>
                {% if line_items[0].cgst_rate is not none %}
                <th>CGST %</th>
                <th>CGST Amt</th>
                <th>SGST %</th>
                <th>SGST Amt</th>
                {% else %}
                <th>IGST %</th>
                <th>IGST Amt</th>
                {% endif %}
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
            {% for item in line_items %}
            <tr>
                <td class="text-center">{{ loop.index }}</td>
                <td>{{ item.description }}</td>
                <td class="text-center">{{ item.hsn_sac }}</td>
                <td class="text-right">{{ item.quantity }}</td>
                <td class="text-right">{{ "%.2f"|format(item.unit_price) }}</td>
                <td class="text-right">{{ "%.2f"|format(item.taxable_value) }}</td>
                {% if item.cgst_rate is not none %}
                <td class="text-right">{{ "%.1f"|format(item.cgst_rate) }}%</td>
                <td class="text-right">{{ "%.2f"|format(item.cgst_amount) }}</td>
                <td class="text-right">{{ "%.1f"|format(item.sgst_rate) }}%</td>
                <td class="text-right">{{ "%.2f"|format(item.sgst_amount) }}</td>
                {% else %}
                <td class="text-right">{{ "%.1f"|format(item.igst_rate) }}%</td>
                <td class="text-right">{{ "%.2f"|format(item.igst_amount) }}</td>
                {% endif %}
                <td class="text-right">{{ "%.2f"|format(item.line_total) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    {% if 'totals_above_items' not in (anomalies or []) %}
    <!-- Normal totals position -->
    <div class="totals">
        <table style="width: 40%; margin-left: auto;">
            <tr>
                <td class="label">Taxable Total:</td>
                <td class="text-right">₹ {{ "%.2f"|format(taxable_total) }}</td>
            </tr>
            <tr>
                <td class="label">Total Tax:</td>
                <td class="text-right">₹ {{ "%.2f"|format(total_tax) }}</td>
            </tr>
            <tr style="font-weight: bold; background-color: #f5f5f5;">
                <td class="label">{{ field_labels.get('total_amount', 'Total Amount:') }}</td>
                <td class="text-right">₹ {{ "%.2f"|format(total_amount) }}</td>
            </tr>
        </table>
    </div>
    {% endif %}
    
    {% if total_amount_in_words %}
    <div style="margin-top: 20px;">
        <strong>Amount in Words:</strong> {{ total_amount_in_words }}
    </div>
    {% endif %}
    
    <div style="margin-top: 30px; text-align: center; font-size: 10px; color: #666;">
        This is a computer generated synthetic invoice for testing purposes only.
        {% if sender_id %}Sender: {{ sender_id }}{% endif %}
        {% if anomalies %} | Anomalies: {{ anomalies|join(', ') }}{% endif %}
    </div>
</body>
</html>"""


def generate_invoices(count: int, output_dir: Path, max_items: int,
                     generate_pdf: bool, seed: Optional[int] = None, 
                     variation_ratio: float = 0.05, sender_count: Optional[int] = None) -> None:
    """Generate specified number of synthetic invoices with sender profile variations.

    Lazily loads WeasyPrint only if PDF output requested. Provides a single
    actionable message if PDF generation cannot proceed, then continues with
    HTML/JSON generation.
    
    Args:
        count: Number of invoices to generate
        output_dir: Directory to save generated files
        max_items: Maximum line items per invoice
        generate_pdf: Whether to generate PDF files
        seed: Random seed for reproducible generation
        variation_ratio: Probability of anomalous layouts (0.0-1.0)
        sender_count: Number of sender profiles to use (defaults to all available)
    """

    weasyprint_module: Optional[Any] = None
    if generate_pdf:
        weasyprint_module, wp_error = load_weasyprint()
        if wp_error:
            print("⚠️  PDF generation disabled:")
            print(wp_error)
            print("📄 Falling back to HTML + JSON only. Proceeding...\n")

    print(f"🏭 Generating {count} synthetic Indian GST invoices...")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Max line items: {max_items}")
    print(f"🎭 Variation ratio: {variation_ratio:.1%} (anomalous layouts)")
    pdf_status = "Enabled" if (generate_pdf and weasyprint_module) else "Disabled"
    print(f"📜 PDF generation: {pdf_status}")
    
    # Build sender profiles
    sender_profiles = _build_sender_profiles()
    effective_sender_count = sender_count or len(sender_profiles)
    print(f"👥 Using {effective_sender_count} sender profiles from {len(sender_profiles)} available")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = IndianGSTInvoiceGenerator(seed=seed)
    
    # Create HTML template
    html_template = Template(create_html_template())
    
    success_count = 0
    anomaly_count = 0
    
    for i in range(1, count + 1):
        try:
            # Pick sender profile and determine if anomalous
            sender_profile, requires_ai_hint = _pick_sender_profile(
                variation_ratio, effective_sender_count, sender_profiles
            )
            
            # Generate base invoice data
            invoice_data = generator.generate_invoice_data(
                max_items, sender_profile=sender_profile, requires_ai_hint=requires_ai_hint
            )
            
            # Apply anomalies based on profile and flags
            invoice_data, anomalies = _apply_anomalies(invoice_data, sender_profile, requires_ai_hint)
            
            # Update invoice data with final anomaly information
            invoice_data["anomalies"] = anomalies
            invoice_data["requires_ai_hint"] = requires_ai_hint or len(anomalies) > 0
            
            # Construct template parameters with styling and layout flags
            template_params = _construct_template_params(invoice_data, sender_profile, anomalies)
            
            # File basenames
            base_name = f"invoice_{i:03d}"
            html_file = output_dir / f"{base_name}.html"
            json_file = output_dir / f"{base_name}.json"
            pdf_file = output_dir / f"{base_name}.pdf"
            
            # Generate HTML using enhanced template
            html_content = html_template.render(**template_params)
            html_file.write_text(html_content, encoding='utf-8')
            
            # Generate JSON ground truth with enhanced metadata
            json_file.write_text(json.dumps(invoice_data, indent=2, ensure_ascii=False), 
                                encoding='utf-8')
            
            # Generate PDF if requested and available
            if generate_pdf and weasyprint_module:
                try:
                    html_doc = weasyprint_module.HTML(string=html_content)
                    html_doc.write_pdf(str(pdf_file))
                except Exception as e:
                    print(f"⚠️  PDF generation failed for {base_name}: {e}")
            
            success_count += 1
            if requires_ai_hint or anomalies:
                anomaly_count += 1
            
            if i % 10 == 0 or i == count:
                print(f"✅ Generated: {i}/{count} ({anomaly_count} with anomalies)")
                
        except Exception as e:
            print(f"❌ Failed to generate invoice {i}: {e}")
    
    print(f"\n🎉 Successfully generated {success_count}/{count} invoices")
    print(f"🎭 {anomaly_count} invoices with anomalies ({anomaly_count/success_count:.1%} actual rate)")
    print(f"📁 Files saved to: {output_dir}")
    
    # Summary of files generated
    html_files = len(list(output_dir.glob("*.html")))
    json_files = len(list(output_dir.glob("*.json")))
    pdf_files = len(list(output_dir.glob("*.pdf")))
    
    print(f"📊 Summary:")
    print(f"   HTML files: {html_files}")
    print(f"   JSON files: {json_files}")
    print(f"   PDF files: {pdf_files}")
    
    # Show sender distribution
    if success_count > 0:
        print(f"👥 Sender distribution:")
        for profile in sender_profiles[:effective_sender_count]:
            print(f"   {profile.sender_id} (quirk level {profile.quirk_level})")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian GST invoices for testing with sender profile variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 10
  %(prog)s --count 25 --pdf --max-items 6
  %(prog)s --count 5 --output-dir /custom/path --seed 12345
  %(prog)s --count 50 --variation-ratio 0.1 --sender-count 4
  %(prog)s --count 100 --variation-ratio 0 --seed 42  # No anomalies
        """
    )
    
    parser.add_argument('--count', type=int, default=10,
                       help='Number of invoices to generate (default: 10)')
    parser.add_argument('--output-dir', type=Path, 
                       default=Path('datasets/indian_gst/generated'),
                       help='Output directory (default: datasets/indian_gst/generated)')
    parser.add_argument('--pdf', action='store_true',
                       help='Generate PDF files (requires WeasyPrint)')
    parser.add_argument('--max-items', type=int, default=4,
                       help='Maximum line items per invoice (default: 4)')
    parser.add_argument('--noise-level', type=int, choices=[0, 1], default=0,
                       help='Noise level for future OCR simulation (0=clean, 1=noisy)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible generation')
    parser.add_argument('--variation-ratio', type=float, default=0.05,
                       help='Probability of anomalous layouts triggering LLM fallback (default: 0.05 = 5%%)')
    parser.add_argument('--sender-count', type=int,
                       help='Number of sender profiles to use (default: all available)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.count <= 0:
        print("❌ Count must be positive")
        sys.exit(1)
        
    if args.max_items <= 0:
        print("❌ Max items must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.variation_ratio <= 1.0):
        print("❌ Variation ratio must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.sender_count is not None and args.sender_count <= 0:
        print("❌ Sender count must be positive")
        sys.exit(1)
    
    # Handle noise level (placeholder for future)
    if args.noise_level > 0:
        print("⚠️  Noise level > 0 is not yet implemented (placeholder for future OCR simulation)")
    
    try:
        generate_invoices(
            count=args.count,
            output_dir=args.output_dir,
            max_items=args.max_items,
            generate_pdf=args.pdf,
            seed=args.seed,
            variation_ratio=args.variation_ratio,
            sender_count=args.sender_count
        )
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()