Working solution demo - https://drive.google.com/file/d/19px0RxK2DatognF6_gzwYyisJ7vKy__w/view?usp=drive_link

# Multi-Format Document Parser

A starter Streamlit application for multi-format document processing using hybrid document normalization. This system combines rule-based extraction with layout signature learning to process heterogeneous document formats into normalized JSON output.

## 🎯 Features

- **🔄 Multi-format Support**: PDF, text, HTML, and email files
- **🏷️ Layout Signature Learning**: Automatic document layout recognition and reuse  
- **📋 Rule-based Extraction**: Configurable regex-based field extraction
- **📊 Normalized Output**: Consistent JSON schema across all document types
- **🌐 Streamlit Interface**: Web-based document upload and management
- **📈 Pipeline Analytics**: Processing statistics and performance monitoring
- **🧠 Selective AI Gating**: Optional Azure OpenAI + Document Intelligence used only when needed (cost-aware)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jaganraajan/multi-format-document-parser.git
   cd multi-format-document-parser
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Usage

### Document Upload
1. Go to the "Document Upload" page
2. Upload one or more supported documents (PDF, TXT, HTML, EML)
3. Click "Process Documents" to extract structured data
4. View results including extracted fields, layout signatures, and processing logs

### Document Library
- Browse all processed documents
- Filter by file type and rule coverage
- View normalized JSON output for any document

### Pipeline Statistics
- Monitor processing performance
- View signature reuse statistics
- Track rule extraction effectiveness

## 🏗️ Architecture

### Core Components

```
src/normalization/
├── schema.py              # Normalized JSON schema definitions
├── signatures.py          # Layout signature learning system  
├── rules_engine.py        # Rule-based extraction engine
├── pipeline.py            # Main processing orchestrator
├── storage.py             # Document repository
└── extractors/            # Format-specific extractors
    ├── pdf_extractor.py   # PDF processing with pdfplumber
    ├── text_extractor.py  # Text and HTML processing
    └── email_extractor.py # Email (.eml) processing
```

### Data Flow

1. **Ingest**: Load and detect document format
2. **Extract**: Format-specific content and layout extraction
3. **Signature**: Generate/match layout signature for document type
4. **Rules**: Apply global and signature-specific extraction rules
5. **Gating**: Evaluate coverage/confidence → decide on AI usage
6. **AI (Conditional)**: LLM + optional DI fallback only if needed
7. **Normalize**: Convert to standardized JSON schema
8. **Store**: Save to document repository with indexing

## ⚙️ Configuration

### Extraction Rules

Edit `rules/global_rules.yml` to customize field extraction:

```yaml
rules:
  - field_name: "invoice_number"
    pattern: "(?:invoice|inv|bill)\\s*(?:number|#)\\s*:?\\s*([A-Z0-9-]+)"
    confidence: 0.9
    description: "Extract invoice number"
    required: true
```

### Directory Structure

- `signatures/` - Layout signature storage (auto-generated)
- `outputs/` - Normalized document outputs  
- `logs/` - Processing logs
- `rules/` - Extraction rule configurations
  - `global_rules.yml` - Global extraction patterns
  - `signature_overrides/` - Signature-specific rule overrides
- `docs/adr/` - Architectural Decision Records (see ADR 0001)

## 📊 Document Schema

All documents are normalized to a consistent JSON structure:

```json
{
  "doc_id": "abc123",
  "ingest_metadata": {
    "filename": "document.pdf",
    "file_type": "pdf",
    "file_size": 12345,
    "content_hash": "sha256...",
    "processing_time_seconds": 1.23
  },
  "sections": [...],
  "key_values": [...],
  "tables": [...], 
  "chunks": [...],
  "processing_meta": {
    "signature_id": "def456",
    "signature_match_score": 0.95,
    "rules_applied": ["global"],
    "coverage_stats": {"rule_coverage": 0.42}
  }
}
```

## 🔧 Development

### Synthetic Dataset (Indian GST Invoices)

A synthetic invoice generator lives in `scripts/generate_indian_invoices.py` producing HTML/JSON (and optional PDF) GST-style invoices for benchmarking extraction. A companion script `scripts/generate_invoice_image_prompts.py` generates image prompts and optionally creates scanned-style invoice images via Azure OpenAI. See `datasets/indian_gst/README.md` for details.

```bash
# Generate 10 synthetic invoices
python scripts/generate_indian_invoices.py --count 10

# Generate with PDF output (requires WeasyPrint)
python scripts/generate_indian_invoices.py --count 25 --pdf

# Generate image prompts and images (requires Azure OpenAI)
python scripts/generate_invoice_image_prompts.py --count 8 --generate-images

# Evaluate parsing accuracy
python scripts/evaluate_invoices.py
```

### Adding New Extractors

1. Create extractor in `src/normalization/extractors/`
2. Implement `extract_content()` and `convert_to_sections()` methods
3. Register in `pipeline.py` with file type mapping

### Adding New Rules

1. Edit `rules/global_rules.yml` for global patterns
2. Create signature-specific rules in `rules/signature_overrides/{signature_id}.yml`
3. Rules automatically reload on pipeline restart

### Testing

Create sample documents and run through the pipeline:

```python
from src.normalization.pipeline import DocumentPipeline

pipeline = DocumentPipeline()
document, log = pipeline.process_document("path/to/document.pdf")
print(f"Extracted {len(document.key_values)} fields")
```

## 📄 Supported File Types

| Format | Extension | Features |
|--------|-----------|----------|  
| PDF | `.pdf` | Text extraction, layout analysis |
| Text | `.txt` | Plain text processing |
| HTML | `.html`, `.htm` | HTML tag removal, content extraction |
| Email | `.eml` | Header parsing, body extraction |

## 🎯 Based On

This starter implementation is based on the comprehensive hybrid document normalization pipeline from [jaganraajan/rag-document-parser PR #10](https://github.com/jaganraajan/rag-document-parser/pull/10), simplified for demonstration and educational purposes.

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation
- Review the example configurations

## 🧭 Architectural Decision Records (ADRs)
ADRs capture *why* architectural choices were made, supporting governance and future evolution. See:
- `docs/adr/0001-hybrid-vs-monolithic-llm.md` – Rationale for hybrid gating vs monolithic LLM approach

## 🤖 Selective AI Extraction (Azure OpenAI + DI Fallback)
The pipeline supports user-toggled AI usage. If both AI toggles are off, documents are processed in pure local mode (signatures + rules). AI is **not** a hard dependency for baseline normalization.

## 🤖 LLM Extraction (Azure OpenAI)

The pipeline uses an Azure OpenAI model to extract key fields (invoice_number, total_amount, date, vendor_name, email, phone_number) directly from the raw document text.

### 1. Azure Setup
Provision an Azure OpenAI resource and create a model deployment (e.g. `gpt-4o-mini`). Note the endpoint URL and API key.

### 2. Environment Variables
Set the following variables before running the pipeline or Streamlit app:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"  # or your deployment name
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # optional override
```

You can place these in a `.env` file and load them with a tool like `direnv` or `dotenv` if preferred.

### 3. Running
Normal usage (LLM mode active by default):
```bash
streamlit run app/streamlit_app.py
```

If Azure environment variables are missing, the LLM extractor safely returns no fields (mock mode) and the document still processes (fields list will be empty). This allows local development without API calls.
---
