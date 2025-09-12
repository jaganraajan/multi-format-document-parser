# Multi-Format Document Parser

A starter Streamlit application for multi-format document processing using hybrid document normalization. This system combines rule-based extraction with layout signature learning to process heterogeneous document formats into normalized JSON output.

## 🎯 Features

- **🔄 Multi-format Support**: PDF, text, HTML, and email files
- **🏷️ Layout Signature Learning**: Automatic document layout recognition and reuse  
- **📋 Rule-based Extraction**: Configurable regex-based field extraction
- **📊 Normalized Output**: Consistent JSON schema across all document types
- **🌐 Streamlit Interface**: Web-based document upload and management
- **📈 Pipeline Analytics**: Processing statistics and performance monitoring

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
5. **Normalize**: Convert to standardized JSON schema
6. **Store**: Save to document repository with indexing

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
    "coverage_stats": {...}
  }
}
```

## 🔧 Development

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

## 🤖 Optional LLM Gap-Fill

The parser includes optional LLM-powered gap-filling to extract required fields that deterministic rules couldn't find. This feature is disabled by default and designed to be cost-aware and deterministic-first.

### Purpose

- **Gap-filling only**: LLM is invoked only for required fields missing after rule extraction
- **Deterministic-first**: Rules and signatures remain primary; LLM never overwrites existing rule-derived values
- **Cost-controlled**: Configurable limits on model calls per document and per-document cost caps

### Enabling LLM Support

1. **Install optional dependency**
   ```bash
   pip install openai>=1.0.0
   ```

2. **Set OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Create LLM configuration**
   
   Create `llm_config.yml` in the repository root:
   ```yaml
   model_provider: "openai"
   model: "gpt-4o-mini"
   temperature: 0.0
   max_model_calls_per_doc: 1
   max_total_cost_usd_per_doc: 0.05
   cost_estimates_per_1k_tokens:
     prompt: 0.003
     completion: 0.006
   cache:
     enabled: true
     dir: "llm_cache"
   ```

### Cost Controls and Caching

- **Per-document limits**: Maximum model calls and cost per document
- **Intelligent caching**: Avoids repeat costs for identical scenarios based on signature + content hash + missing field set
- **Token estimation**: Rough heuristic-based cost tracking (4 chars ≈ 1 token)
- **Graceful degradation**: Missing API key or dependencies result in rule-only processing with warnings

### Monitoring LLM Usage

- **UI Integration**: Processing results table shows "Model Calls" and "Cost (USD)" columns
- **Detailed Logs**: Processing logs include LLM activity (fields requested, fields returned, estimated cost)
- **Method Tracking**: Key-value pairs show extraction method ("rule" vs "model") for interpretability

### Example LLM Configuration

```yaml
# Enable LLM gap-filling with cost controls
model: "gpt-4o-mini"              # Fast, cost-effective model
max_model_calls_per_doc: 1        # At most one LLM call per document
max_total_cost_usd_per_doc: 0.05  # Cap at 5 cents per document
cache:
  enabled: true                   # Cache responses to avoid repeat costs
  dir: "llm_cache"               # Local cache directory
```

**Note**: Without `llm_config.yml` or the OpenAI API key, the system behaves exactly as before with no errors or changes to existing functionality.

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