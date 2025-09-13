# Indian GST Invoice Synthetic Dataset

## Purpose & Disclaimer

This directory contains tools and infrastructure for generating **completely synthetic** Indian GST (Goods and Services Tax) invoices for testing and benchmarking document parsing systems. 

⚠️ **Important**: All generated data is synthetic and created using the Faker library. No real company information, personal data, or actual business transactions are included.

## Overview

The synthetic dataset generation system provides:
- Realistic GST invoice layouts with proper tax calculations
- Ground truth JSON for precise evaluation metrics
- Support for both intra-state (CGST+SGST) and inter-state (IGST) transactions
- Configurable invoice complexity (line items, noise levels)
- Optional PDF generation for OCR testing scenarios

## Field Schema

The invoice schema is defined in `schema_fields.json` and includes:

### Core Fields (Flat Structure)
- **invoice_number**: Unique identifier (e.g., "INV-2024-001")
- **invoice_date**: ISO date format (YYYY-MM-DD)
- **supplier_name**: Vendor company name
- **supplier_gstin**: 15-character GSTIN following official format
- **supplier_address**: Complete supplier address
- **recipient_name**: Buyer company name  
- **recipient_gstin**: 15-character buyer GSTIN
- **recipient_address**: Complete buyer address
- **place_of_supply**: State where goods/services are supplied
- **reverse_charge**: "Yes" or "No" for reverse charge applicability
- **taxable_total**: Sum of all line item taxable amounts
- **total_tax**: Total GST amount (CGST+SGST or IGST)
- **total_amount**: Final amount including all taxes
- **total_amount_in_words**: Amount in words (may be blank)
- **phone_number**: Contact phone (optional)
- **email**: Contact email (optional)

### Line Items (Nested Array)
Each line item contains:
- **description**: Product/service description
- **hsn_sac**: HSN code for goods or SAC for services
- **quantity**: Number of units
- **unit_price**: Price per unit before tax
- **taxable_value**: Line total before tax (quantity × unit_price)
- **cgst_rate/cgst_amount**: Central GST (intra-state only)
- **sgst_rate/sgst_amount**: State GST (intra-state only)  
- **igst_rate/igst_amount**: Integrated GST (inter-state only)
- **line_total**: Final line amount including applicable taxes

## GST Tax Logic

### Intra-State Transactions (Same State)
- CGST (Central GST) + SGST (State GST) = Total GST Rate
- Example: 18% GST = 9% CGST + 9% SGST

### Inter-State Transactions (Different States)  
- IGST (Integrated GST) = Total GST Rate
- Example: 18% GST = 18% IGST (no CGST/SGST)

### Supported GST Rates
- 5% (essential goods/services)
- 12% (standard goods/services)  
- 18% (luxury goods/services)

## Generation Usage

### Basic Generation
Generate 10 invoices with default settings:
```bash
python scripts/generate_indian_invoices.py
```

### Advanced Options
```bash
# Generate 25 invoices with PDF output
python scripts/generate_indian_invoices.py --count 25 --pdf

# Custom output directory with more line items
python scripts/generate_indian_invoices.py --count 5 --output-dir /custom/path --max-items 8

# Reproducible generation with seed
python scripts/generate_indian_invoices.py --count 10 --seed 12345

# Future: Add noise for OCR testing (placeholder)
python scripts/generate_indian_invoices.py --count 5 --noise-level 1
```

### Command Line Arguments
- `--count`: Number of invoices to generate (default: 10)
- `--output-dir`: Output directory (default: datasets/indian_gst/generated)
- `--pdf`: Generate PDF files (requires WeasyPrint installation)
- `--max-items`: Maximum line items per invoice (default: 4)
- `--noise-level`: Future OCR noise simulation (0=clean, 1=noisy, default: 0)
- `--seed`: Random seed for reproducible generation

### Output Files
For each invoice, the following files are generated:
- `invoice_001.html`: HTML version of the invoice
- `invoice_001.json`: Ground truth data in JSON format
- `invoice_001.pdf`: PDF version (if --pdf flag used and WeasyPrint available)

## PDF Generation

PDF generation uses WeasyPrint and is optional:

### Install WeasyPrint (Optional)
```bash
pip install weasyprint
```

If WeasyPrint is not installed, the script will:
- Print a helpful installation message
- Continue generating HTML and JSON files
- Skip PDF generation gracefully

### WeasyPrint Requirements
WeasyPrint may require additional system dependencies on some platforms. See the [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation) for details.

## Evaluation

Use the evaluation script to compare extracted fields against ground truth:

```bash
# Basic evaluation
python scripts/evaluate_invoices.py

# Custom directories
python scripts/evaluate_invoices.py --ground-truth-dir datasets/indian_gst/generated --parsed-dir outputs/

# Limit evaluation to specific number of invoices
python scripts/evaluate_invoices.py --limit 50
```

### Evaluation Metrics
The evaluation script computes:
- **Exact Matches**: Fields that match ground truth exactly
- **Missing Fields**: Fields present in ground truth but not in parsed output
- **Extra Fields**: Fields in parsed output but not in ground truth
- **Precision/Recall**: Based on exact field matching

*Note: Line item evaluation is planned for future versions*

## Adding Layout Variety

### Creating New Templates
1. Create additional Jinja2 templates in the generator script
2. Add template selection logic for layout diversity
3. Ensure all templates include required GST fields

### Image Prompt & Generation Script

A companion script generates synthetic prompts for creating scanned-style invoice images via Azure OpenAI DALL-E or compatible image generation endpoints.

### Basic Usage
Generate prompts only (no API calls):
```bash
python scripts/generate_invoice_image_prompts.py --count 5 --dry-run
```

### Image Generation with Azure OpenAI
Set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-02-01"  # optional
```

Generate prompts and images:
```bash
python scripts/generate_invoice_image_prompts.py --count 8 --generate-images --seed 42
```

### Advanced Options
```bash
# Custom image size
python scripts/generate_invoice_image_prompts.py --count 3 --size 832x1216

# Custom fragments and output directory
python scripts/generate_invoice_image_prompts.py --count 10 --fragments custom_fragments.yml --output-dir /custom/path

# Skip metadata generation
python scripts/generate_invoice_image_prompts.py --count 5 --no-metadata
```

### Output Structure
- `prompts/`: Text files with composed prompts
- `images/`: Generated PNG images (if API enabled)
- `metadata/`: JSON metadata with fragment selections and API response info

### Script Arguments
- `--count`: Number of samples to generate (default: 5)
- `--fragments`: Path to YAML fragments file (default: datasets/indian_gst/prompt_fragments.yml)
- `--output-dir`: Output directory (default: datasets/indian_gst/image_invoices)
- `--model`: Image generation model (default: gpt-image-1)
- `--size`: Image size in WIDTHxHEIGHT format (default: 1024x1536)
- `--seed`: Random seed for reproducible generation
- `--generate-images`: Enable API calls for image generation
- `--dry-run`: Show prompts to stdout only
- `--no-metadata`: Skip metadata JSON files
- `--timeout`: API timeout in seconds (default: 60)
- `--verbose`: Enable detailed logging

### Prompt Composition
The script combines fragments from different categories:
- **Required**: 1 base + 1 layout + 1 tax fragment
- **Optional**: 0-1 noise + 0-1 embellishment + 0-2 variants

Each fragment is designed to flow naturally when concatenated, with automatic cleanup of punctuation and addition of a fixed suffix ensuring synthetic data labeling.

## Future Enhancements (TODO)
- Multiple invoice layout templates (modern, traditional, minimalist)
- PDF conversion of generated images  
- Multi-page composite prompts
- Automatic pairing with synthetic JSON ground-truth when text-based generation is aligned
- Confidence scoring for extracted fields
- Integration with Azure Document Intelligence for comparison

## Dataset Structure

```
datasets/indian_gst/
├── README.md                    # This documentation
├── schema_fields.json           # Field definitions and validation rules
├── samples/                     # Optional: small example files
├── ground_truth/               # Placeholder for manual ground truth files
└── generated/                  # Git-ignored directory for generated files
    ├── .gitignore
    ├── invoice_001.html
    ├── invoice_001.json
    ├── invoice_001.pdf         # If --pdf used
    └── ...
```

## GSTIN Generation

GSTINs follow the official 15-character format:
- **Positions 1-2**: State code (01-37 for different Indian states)
- **Positions 3-7**: 5 letters from PAN  
- **Positions 8-11**: 4 digits from PAN
- **Position 12**: 1 letter from PAN
- **Position 13**: Entity number (1-9, A-Z)
- **Position 14**: 'Z' (fixed)
- **Position 15**: Checksum digit

Example: `07ABCDE1234F1Z5` (Delhi-based entity)

## Sample State Codes
- 07: Delhi
- 09: Uttar Pradesh  
- 19: West Bengal
- 27: Maharashtra
- 29: Karnataka
- 33: Tamil Nadu

## License

This synthetic dataset inherits the MIT license from the parent repository. The generated data is completely synthetic and free to use for research, testing, and development purposes.

## Contributing

Contributions are welcome for:
- Additional invoice layout templates
- Enhanced tax calculation scenarios  
- OCR noise simulation improvements
- New evaluation metrics
- Documentation improvements

Please ensure all contributions maintain the synthetic-only nature of the dataset and do not include any real business or personal information.

## Support

For questions about the synthetic dataset generation:
1. Check this README for common usage patterns
2. Review the `schema_fields.json` for field definitions
3. Examine generated sample files for format examples
4. Open an issue in the main repository for bugs or feature requests