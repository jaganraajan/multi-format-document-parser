#!/usr/bin/env python3
"""
Synthetic Indian GST Invoice Generator

Generates realistic but completely synthetic Indian GST tax invoices
for testing and benchmarking document parsing systems.

Author: Multi-Format Document Parser
License: MIT
"""

import argparse
import json
import random
import string
import sys
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
    
    def generate_invoice_data(self, max_items: int = 4) -> Dict:
        """Generate complete synthetic invoice data."""
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
        
        return invoice_data


def create_html_template() -> str:
    """Create Jinja2 HTML template for invoice rendering."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tax Invoice - {{ invoice_number }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; font-size: 12px; }
        .header { text-align: center; margin-bottom: 20px; }
        .invoice-title { font-size: 18px; font-weight: bold; }
        .company-details { margin: 15px 0; }
        .invoice-details { margin: 15px 0; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; font-weight: bold; }
        .text-right { text-align: right; }
        .text-center { text-align: center; }
        .totals { margin-top: 20px; }
        .grid { display: flex; gap: 20px; }
        .grid-item { flex: 1; }
        .label { font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <div class="invoice-title">TAX INVOICE</div>
    </div>
    
    <div class="grid">
        <div class="grid-item">
            <div class="company-details">
                <div class="label">From:</div>
                <div><strong>{{ supplier_name }}</strong></div>
                <div>GSTIN: {{ supplier_gstin }}</div>
                <div>{{ supplier_address }}</div>
                {% if phone_number %}<div>Phone: {{ phone_number }}</div>{% endif %}
                {% if email %}<div>Email: {{ email }}</div>{% endif %}
            </div>
        </div>
        
        <div class="grid-item">
            <div class="company-details">
                <div class="label">To:</div>
                <div><strong>{{ recipient_name }}</strong></div>
                <div>GSTIN: {{ recipient_gstin }}</div>
                <div>{{ recipient_address }}</div>
            </div>
        </div>
    </div>
    
    <div class="invoice-details">
        <table>
            <tr>
                <td class="label">Invoice Number:</td>
                <td>{{ invoice_number }}</td>
                <td class="label">Invoice Date:</td>
                <td>{{ invoice_date }}</td>
            </tr>
            <tr>
                <td class="label">Place of Supply:</td>
                <td>{{ place_of_supply }}</td>
                <td class="label">Reverse Charge:</td>
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
                <td class="label">Total Amount:</td>
                <td class="text-right">₹ {{ "%.2f"|format(total_amount) }}</td>
            </tr>
        </table>
    </div>
    
    {% if total_amount_in_words %}
    <div style="margin-top: 20px;">
        <strong>Amount in Words:</strong> {{ total_amount_in_words }}
    </div>
    {% endif %}
    
    <div style="margin-top: 30px; text-align: center; font-size: 10px; color: #666;">
        This is a computer generated synthetic invoice for testing purposes only.
    </div>
</body>
</html>"""


def generate_invoices(count: int, output_dir: Path, max_items: int,
                     generate_pdf: bool, seed: Optional[int] = None) -> None:
    """Generate specified number of synthetic invoices.

    Lazily loads WeasyPrint only if PDF output requested. Provides a single
    actionable message if PDF generation cannot proceed, then continues with
    HTML/JSON generation.
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
    pdf_status = "Enabled" if (generate_pdf and weasyprint_module) else "Disabled"
    print(f"� PDF generation: {pdf_status}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = IndianGSTInvoiceGenerator(seed=seed)
    
    # Create HTML template
    html_template = Template(create_html_template())
    
    success_count = 0
    
    for i in range(1, count + 1):
        try:
            # Generate invoice data
            invoice_data = generator.generate_invoice_data(max_items)
            
            # File basenames
            base_name = f"invoice_{i:03d}"
            html_file = output_dir / f"{base_name}.html"
            json_file = output_dir / f"{base_name}.json"
            pdf_file = output_dir / f"{base_name}.pdf"
            
            # Generate HTML
            html_content = html_template.render(**invoice_data)
            html_file.write_text(html_content, encoding='utf-8')
            
            # Generate JSON ground truth
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
            
            if i % 10 == 0 or i == count:
                print(f"✅ Generated: {i}/{count}")
                
        except Exception as e:
            print(f"❌ Failed to generate invoice {i}: {e}")
    
    print(f"\n🎉 Successfully generated {success_count}/{count} invoices")
    print(f"📁 Files saved to: {output_dir}")
    
    # Summary of files generated
    html_files = len(list(output_dir.glob("*.html")))
    json_files = len(list(output_dir.glob("*.json")))
    pdf_files = len(list(output_dir.glob("*.pdf")))
    
    print(f"📊 Summary:")
    print(f"   HTML files: {html_files}")
    print(f"   JSON files: {json_files}")
    print(f"   PDF files: {pdf_files}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian GST invoices for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --count 10
  %(prog)s --count 25 --pdf --max-items 6
  %(prog)s --count 5 --output-dir /custom/path --seed 12345
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.count <= 0:
        print("❌ Count must be positive")
        sys.exit(1)
        
    if args.max_items <= 0:
        print("❌ Max items must be positive")
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
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()