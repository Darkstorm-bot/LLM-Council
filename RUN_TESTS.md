# 🧪 Running Tests for LLM Council Research System

This guide explains how to run the comprehensive test suite that validates all implemented features.

## Prerequisites

First, install the testing dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Code coverage reporting

## Running Tests

### Full Test Suite

Run all tests with coverage report:

```bash
pytest
```

Or with verbose output:

```bash
pytest -v
```

### Run Specific Test Classes

**Test Document Ingestion (Phase 1):**
```bash
pytest tests/test_all_features.py::TestDocumentIngestion -v
```

**Test Iterative Deep-Dive (Phase 2):**
```bash
pytest tests/test_all_features.py::TestIterativeDeepDive -v
```

**Test Fact Verification (Phase 3):**
```bash
pytest tests/test_all_features.py::TestFactVerification -v
```

**Test Memory Palace Integration:**
```bash
pytest tests/test_all_features.py::TestMemoryPalaceIntegration -v
```

**Test Orchestrator Integration:**
```bash
pytest tests/test_all_features.py::TestOrchestratorIntegration -v
```

**Test Error Handling:**
```bash
pytest tests/test_all_features.py::TestErrorHandling -v
```

### Run with Coverage Report

Generate HTML coverage report:

```bash
pytest --cov=tools --cov=core --cov-report=html
```

Open the report in your browser:
```bash
# On Linux
xdg-open cov_html/index.html

# On macOS
open cov_html/index.html

# On Windows
start cov_html/index.html
```

### Quick Smoke Test

For a quick validation without coverage overhead:

```bash
pytest tests/test_all_features.py -v --tb=short -q
```

## Test Categories

### Unit Tests
- Parser initialization tests
- Data structure creation tests
- Logic function tests

### Integration Tests
- Memory Palace integration
- Orchestrator configuration
- End-to-end workflow (mocked)

### Error Handling Tests
- Invalid file handling
- Empty results handling
- Malformed content handling

## Expected Output

When running tests, you should see output like:

```
tests/test_all_features.py::TestDocumentIngestion::test_pdf_parser_initialization PASSED
tests/test_all_features.py::TestDocumentIngestion::test_docx_parser_initialization PASSED
tests/test_all_features.py::TestDocumentIngestion::test_text_parser_initialization PASSED
...

==================== 45 passed in 2.34s ====================
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're running from the workspace root:

```bash
cd /workspace
pytest
```

### Async Test Warnings

Async tests require `pytest-asyncio`. If you see warnings:

```bash
pip install pytest-asyncio
```

### Connection Refused Errors

Some integration tests may show connection errors if Ollama/SearXNG aren't running. This is expected and handled gracefully. The tests verify structure, not live execution.

## Continuous Testing

For development, run tests automatically on file changes:

```bash
pip install pytest-watch
ptw -- tests/test_all_features.py
```

## GUI Testing

To test the GUI manually:

```bash
python gui.py
```

Then in the browser interface:
1. Enable feature checkboxes (📄 Document Ingestion, 🔍 Deep-Dive, ✅ Fact Verification)
2. Upload test documents or paste URLs
3. Enter a research prompt
4. Click "▶ Convene Council"
5. Check new tabs: 📚 Citations & Sources, 🔍 Research Trail, ✅ Verification Report

## Performance Benchmarks

Typical test execution times:
- Full suite: ~2-5 seconds
- Individual class: ~0.5-1 second
- With coverage: ~5-10 seconds

## Next Steps

After tests pass:
1. Review coverage report for gaps
2. Add tests for new features
3. Integrate with CI/CD pipeline
4. Run before each commit
