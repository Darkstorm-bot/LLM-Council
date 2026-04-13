# Phase 1: Multi-Modal Document Ingestion - Usage Guide

## Overview
This implementation adds support for ingesting and analyzing documents from multiple sources:
- **PDF files** - Academic papers, technical documentation, reports
- **DOCX files** - Word documents, specifications
- **Text files** - TXT, MD, RST, TEX formats
- **GitHub repositories** - READMEs, package configs, key documentation files
- **Web URLs** - Articles, blog posts, documentation pages

Documents are automatically chunked with intelligent boundaries, include citation tracking, and are injected into the council's context for analysis.

## Installation

Install new dependencies:
```bash
pip install PyMuPDF python-docx beautifulsoup4
# Or update all requirements:
pip install -r requirements.txt
```

## Command-Line Usage

### Basic Document Ingestion
```bash
# Analyze a question with PDF documents
python orchestrator.py "What are the key security considerations for this architecture?" \
    --documents ./docs/security-whitepaper.pdf ./docs/architecture-spec.docx \
    --document-ingestion

# Mix documents with web research
python orchestrator.py "Compare these ML frameworks" \
    --documents ./ml-comparison.pdf https://github.com/pytorch/pytorch \
    --document-ingestion

# GitHub repository analysis
python orchestrator.py "What is the contribution workflow for this project?" \
    --documents https://github.com/owner/repo \
    --document-ingestion
```

### Supported Document Types
| Type | Extensions/Sources | Parser |
|------|-------------------|--------|
| PDF | `.pdf` | PyMuPDF (fitz) |
| Word | `.docx` | python-docx |
| Text | `.txt`, `.md`, `.rst`, `.tex` | Built-in |
| GitHub | `github.com/owner/repo` | GitHub API |
| Web URL | `http://...`, `https://...` | requests + BeautifulSoup |

## Programmatic Usage

### Basic Example
```python
from orchestrator import build_orchestrator_from_config
from tools.document_tools import DocumentIngestionEngine

# Create orchestrator with document ingestion enabled
orchestrator = build_orchestrator_from_config(
    config_path="config.yaml",
    overrides={
        "document_ingestion_enabled": True,
    }
)

# Pre-load documents
orchestrator.doc_engine.ingest("./docs/whitepaper.pdf")
orchestrator.doc_engine.ingest("./docs/specs.docx")
orchestrator.doc_engine.ingest("https://github.com/owner/repo")

# Run analysis
result = orchestrator.run("Analyze the feasibility of this approach")
print(result)
```

### Batch Document Loading
```python
from tools.document_tools import DocumentIngestionEngine

engine = DocumentIngestionEngine()

# Load multiple documents
sources = [
    ("./paper.pdf", "pdf"),
    ("./spec.docx", "docx"),
    ("https://github.com/org/project", "github"),
    ("https://example.com/article", "url"),
]

for source, doc_type in sources:
    try:
        doc = engine.ingest(source, doc_type)
        print(f"✓ Loaded: {doc.title} ({len(doc.chunks)} chunks)")
    except Exception as e:
        print(f"✗ Failed: {source} - {e}")

# Get all chunks for custom processing
all_chunks = engine.get_all_chunks()
for chunk in all_chunks:
    print(chunk.to_citation())
    print(chunk.content[:200])
```

### Advanced: Custom Chunking
```python
from tools.document_tools import DocumentIngestionEngine

# Customize chunk size and overlap
engine = DocumentIngestionEngine(
    max_chunk_size=2000,  # Larger chunks
    chunk_overlap=300,     # More overlap for context
)

doc = engine.ingest("./large-paper.pdf")
print(f"Document: {doc.title}")
print(f"Chunks: {len(doc.chunks)}")
print(f"Pages: {doc.metadata.get('page_count', 'N/A')}")
```

## How It Works

### 1. Document Parsing Pipeline
```
User provides document → Auto-detect type → Route to parser → Extract text → Chunk → Store
```

### 2. Chunking Strategy
- **Sentence-aware boundaries**: Chunks break at periods or newlines when possible
- **Overlapping chunks**: 200-character overlap maintains context between chunks
- **Configurable size**: Default 1500 characters per chunk (adjustable)

### 3. Citation Tracking
Each chunk includes:
- Document title and source
- Page numbers (for PDFs)
- Chunk index within document
- Metadata (file path, domain, etc.)

Example citation: `[Security Whitepaper pp. 3-5 — Chunk 2/8]`

### 4. Context Injection
Documents are formatted and injected into the Memory Palace:
```
=== INGESTED DOCUMENTS (3 docs, 12 chunks) ===

[Security Whitepaper p. 1 — Chunk 1/8]
The primary security considerations include...

[Architecture Spec — Chunk 1/4]
The system uses a microservices pattern...
```

## Integration with Council Flow

Document ingestion happens after web research (Phase 0) and before brainstorming (Phase 1):

```
Phase 0: Web Research (SearXNG queries)
    ↓
[NEW] Document Ingestion (Parse & chunk uploaded docs)
    ↓
Phase 1: Council Brainstorm (Models read web + docs)
    ↓
Phase 2: Cross-examination
    ↓
Phase 3-4: Synthesis & Voting
```

## Memory Palace Extensions

Two new fields added to track documents:
- `ingested_documents`: List of document metadata
- `document_context`: Formatted text for LLM context

## Configuration (config.yaml)

```yaml
features:
  document_ingestion_enabled: true
  
# Optional: customize chunking behavior
document_settings:
  max_chunk_size: 1500
  chunk_overlap: 200
  max_chunks_per_doc: 5
```

## Error Handling

- Missing files: Clear error message with file path
- Unsupported formats: Falls back to text parsing
- Network failures (GitHub/URLs): Logged and skipped
- Context limits: Automatically truncates to fit model context

## Best Practices

1. **Limit document count**: 3-5 large documents work best to avoid context explosion
2. **Use relevant documents**: Only upload documents directly related to the query
3. **Combine with web research**: Documents + web searches provide comprehensive coverage
4. **Monitor chunk counts**: Large PDFs may generate many chunks; adjust `max_chunk_size` if needed

## Example Use Cases

### Academic Paper Analysis
```bash
python orchestrator.py "Summarize the key findings and methodology" \
    --documents ./research-paper.pdf \
    --document-ingestion
```

### Codebase Documentation Review
```bash
python orchestrator.py "What is the recommended deployment strategy?" \
    --documents https://github.com/myorg/myproject \
    --document-ingestion
```

### Technical Specification Comparison
```bash
python orchestrator.py "Compare these two API specifications for consistency" \
    --documents ./api-v1-spec.pdf ./api-v2-spec.docx \
    --document-ingestion
```

### Legal/Compliance Document Analysis
```bash
python orchestrator.py "Identify potential compliance gaps" \
    --documents ./gdpr-requirements.pdf ./our-policy.docx \
    --document-ingestion
```

## Next Steps (Future Phases)

- **Phase 2**: Iterative Deep-Dive Research Agent
- **Phase 3**: Cross-Source Fact Verification

These will build on the document ingestion foundation to enable autonomous research loops and contradiction detection.
