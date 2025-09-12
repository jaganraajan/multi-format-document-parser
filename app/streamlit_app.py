"""
Streamlit UI for Multi-Format Document Parser.

This module provides a web interface for document upload, processing,
and management using the hybrid document normalization pipeline.
"""

import streamlit as st
import os
import sys
import json
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization.pipeline import DocumentPipeline
from src.normalization.storage import DocumentRepository


# Page configuration
st.set_page_config(
    page_title="Multi-Format Document Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []


def initialize_pipeline():
    """Initialize the document pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing pipeline..."):
            try:
                st.session_state.pipeline = DocumentPipeline()
                st.success("Pipeline initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing pipeline: {e}")
                return None
    return st.session_state.pipeline


def main():
    """Main Streamlit application."""
    st.title("📄 Multi-Format Document Parser")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Document Upload", "Document Library", "Pipeline Stats", "About"]
    )
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    if pipeline is None:
        st.error("Failed to initialize pipeline. Please check your setup.")
        return
    
    if page == "Document Upload":
        document_upload_page(pipeline)
    elif page == "Document Library":
        document_library_page(pipeline)
    elif page == "Pipeline Stats":
        pipeline_stats_page(pipeline)
    elif page == "About":
        about_page()


def document_upload_page(pipeline: DocumentPipeline):
    """Document upload and processing page."""
    st.header("📤 Document Upload & Processing")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to process",
        type=['pdf', 'txt', 'html', 'eml'],
        accept_multiple_files=True,
        help="Supported formats: PDF, text files, HTML, email (.eml)"
    )
    
    # Process documents
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        process_documents(pipeline, uploaded_files)
    
    # Display processing results
    if st.session_state.processed_docs:
        st.subheader("📊 Processing Results")
        display_processing_results()


def process_documents(pipeline: DocumentPipeline, uploaded_files):
    """Process uploaded documents."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                document, processing_log = pipeline.process_document(tmp_file_path)
                
                result = {
                    'filename': uploaded_file.name,
                    'doc_id': document.doc_id,
                    'status': 'success',
                    'signature_id': document.processing_meta.signature_id,
                    'signature_match': document.processing_meta.signature_match_score,
                    'fields_extracted': len(document.key_values),
                    'processing_time': document.ingest_metadata.processing_time_seconds,
                    'coverage_stats': document.processing_meta.coverage_stats,
                    'processing_log': processing_log,
                    'document': document
                }
                
                st.success(f"✅ Processed {uploaded_file.name}")
                
            except Exception as e:
                result = {
                    'filename': uploaded_file.name,
                    'doc_id': None,
                    'status': 'error',
                    'error': str(e)
                }
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
            
            results.append(result)
            
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
    
    # Update session state
    st.session_state.processed_docs.extend(results)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")


def display_processing_results():
    """Display processing results in a table."""
    # Create DataFrame
    data = []
    for result in st.session_state.processed_docs:
        if result['status'] == 'success':
            coverage = result.get('coverage_stats', {})
            data.append({
                'File': result['filename'],
                'Doc ID': result['doc_id'],
                'Signature': result['signature_id'][:8] + '...' if result['signature_id'] else 'N/A',
                'Match Score': f"{result['signature_match']:.2f}",
                'Fields': result['fields_extracted'],
                'Rule Coverage': f"{coverage.get('rule_coverage', 0):.1%}",
                'Time (s)': f"{result['processing_time']:.1f}",
                'Status': '✅ Success'
            })
        else:
            data.append({
                'File': result['filename'],
                'Doc ID': 'N/A',
                'Signature': 'N/A',
                'Match Score': 'N/A',
                'Fields': 'N/A',
                'Rule Coverage': 'N/A',
                'Time (s)': 'N/A',
                'Status': f"❌ {result.get('error', 'Error')}"
            })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Document detail section
        st.subheader("📋 Document Details")
        
        # Select document for details
        doc_options = [f"{r['filename']} ({r['doc_id']})" for r in st.session_state.processed_docs if r['status'] == 'success']
        
        if doc_options:
            selected_doc = st.selectbox("Select document for details:", doc_options)
            
            if selected_doc:
                # Find selected document
                doc_id = selected_doc.split('(')[-1].rstrip(')')
                selected_result = next(r for r in st.session_state.processed_docs if r.get('doc_id') == doc_id)
                
                display_document_details(selected_result)


def display_document_details(result: Dict[str, Any]):
    """Display detailed information for a selected document."""
    document = result['document']
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 JSON Preview", "📊 Key-Values", "📋 Sections", "📝 Processing Log"])
    
    with tab1:
        st.subheader("Normalized JSON")
        
        # Download button
        json_str = json.dumps(document.to_dict(), indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{document.doc_id}_normalized.json",
            mime="application/json"
        )
        
        # Display JSON
        st.json(document.to_dict())
    
    with tab2:
        st.subheader("Extracted Key-Value Pairs")
        
        if document.key_values:
            kv_data = []
            for kv in document.key_values:
                kv_data.append({
                    'Field': kv.key,
                    'Value': str(kv.value),
                    'Confidence': f"{kv.confidence:.2f}",
                    'Method': kv.extraction_method.title()
                })
            
            kv_df = pd.DataFrame(kv_data)
            st.dataframe(kv_df, use_container_width=True)
        else:
            st.info("No key-value pairs extracted")
    
    with tab3:
        st.subheader("Document Sections")
        
        if document.sections:
            for i, section in enumerate(document.sections):
                with st.expander(f"Section {i+1}: {section.title}"):
                    st.write(section.content)
        else:
            st.info("No sections found")
    
    with tab4:
        st.subheader("Processing Log")
        
        if result.get('processing_log'):
            st.text(result['processing_log'])
        else:
            st.info("No processing log available")


def document_library_page(pipeline: DocumentPipeline):
    """Document library and search page."""
    st.header("📚 Document Library")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file_type_filter = st.selectbox(
            "File Type",
            ["All", "pdf", "text", "email", "html"]
        )
    
    with col2:
        min_coverage = st.slider(
            "Min Rule Coverage",
            0.0, 1.0, 0.0, 0.1
        )
    
    with col3:
        st.write("")  # Spacing
    
    # Build filters
    filters = {}
    if file_type_filter != "All":
        filters["file_type"] = file_type_filter
    if min_coverage > 0:
        filters["min_coverage"] = min_coverage
    
    # Get documents
    documents = pipeline.repository.search_documents(**filters)
    
    if documents:
        # Display documents table
        data = []
        for doc in documents:
            coverage = doc.get('coverage_stats', {})
            data.append({
                'File': doc['filename'],
                'Doc ID': doc['doc_id'],
                'Type': doc['file_type'],
                'Uploaded': doc['uploaded_at'][:19] if doc.get('uploaded_at') else 'N/A',
                'Signature': doc.get('signature_id', 'N/A')[:8] + '...' if doc.get('signature_id') else 'N/A',
                'Fields': doc['key_values_count'],
                'Coverage': f"{coverage.get('rule_coverage', 0):.1%}",
                'Size (bytes)': doc['file_size']
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Document selection for viewing
        selected_doc_id = st.selectbox("Select document to view:", ["None"] + [doc['doc_id'] for doc in documents])
        if selected_doc_id != "None":
            document_data = pipeline.repository.get_document(selected_doc_id)
            if document_data:
                st.subheader(f"Document: {selected_doc_id}")
                st.json(document_data)
    else:
        st.info("No documents found matching the criteria")


def pipeline_stats_page(pipeline: DocumentPipeline):
    """Pipeline statistics page."""
    st.header("📈 Pipeline Statistics")
    
    try:
        stats = pipeline.get_pipeline_stats()
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Documents",
                stats['repository']['total_documents']
            )
        
        with col2:
            st.metric(
                "Total Signatures",
                stats['signatures']['total_signatures']
            )
        
        with col3:
            st.metric(
                "Total Rules",
                stats['rules']['total_rules']
            )
        
        # Detailed stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Repository Stats")
            st.json(stats['repository'])
        
        with col2:
            st.subheader("🏷️ Signature Stats")
            st.json(stats['signatures'])
        
        # Rules stats
        st.subheader("📋 Rules Stats")
        st.json(stats['rules'])
        
    except Exception as e:
        st.error(f"Error loading stats: {e}")


def about_page():
    """About page with information about the system."""
    st.header("ℹ️ About Multi-Format Document Parser")
    
    st.markdown("""
    ## Overview
    
    This is a starter implementation of a multi-format document parser that demonstrates 
    hybrid document normalization using rule-based extraction and layout signature learning.
    
    ## Key Features
    
    - **🔄 Multi-format Support**: PDF, text, HTML, and email files
    - **🏷️ Layout Signature Learning**: Automatic document layout recognition and reuse
    - **📋 Rule-based Extraction**: Configurable regex-based field extraction
    - **📊 Normalized Output**: Consistent JSON schema across all document types
    - **🌐 Web Interface**: Easy-to-use Streamlit interface
    
    ## Architecture
    
    The system consists of several key components:
    
    1. **Document Extractors**: Format-specific content extraction
    2. **Signature Manager**: Layout pattern learning and matching
    3. **Rules Engine**: Configurable field extraction rules
    4. **Document Repository**: Normalized document storage
    5. **Pipeline Orchestrator**: End-to-end processing coordination
    
    ## Supported File Types
    
    - **PDF**: Text extraction using pdfplumber
    - **Text**: Plain text and HTML files
    - **Email**: RFC822 .eml files
    
    ## Usage
    
    1. Upload documents using the Document Upload page
    2. View processing results and extracted data
    3. Browse the document library with search and filtering
    4. Monitor pipeline statistics and performance
    
    ## Configuration
    
    - **Rules**: Edit `rules/global_rules.yml` to customize extraction patterns
    - **Signatures**: Automatic layout signatures are stored in `signatures/`
    - **Outputs**: Normalized documents are saved in `outputs/`
    
    ## Based On
    
    This starter app is based on the hybrid document normalization pipeline from 
    [rag-document-parser PR #10](https://github.com/jaganraajan/rag-document-parser/pull/10),
    simplified for demonstration and educational purposes.
    """)


if __name__ == "__main__":
    main()