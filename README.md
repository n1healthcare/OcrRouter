# OCRRouter

> A powerful Python library for converting PDFs and images to Markdown using multiple expert VLM backends

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

## What is OCRRouter?

OCRRouter is a production-ready document processing library that converts PDFs and images to high-quality Markdown. It stands out with:

- **6 Expert VLM Backends** — Choose from MinerU, DeepSeek-OCR, DotsOCR, PaddleOCR, Hunyuan-OCR, or GeneralVLM (GPT/Claude/Gemini)
- **Composite Mode** — Mix layout detection from one model with OCR from another for optimal results (unique feature!)
- **Rich Document Support** — Tables, formulas, images, code blocks, lists, and complex layouts
- **Flexible APIs** — Sync/async, single/batch processing, multiple output formats
- **Production Ready** — Built-in observability (Langfuse), retries, error handling, debug mode

## Quick Start

### Installation

```bash
pip install ocrrouter
```

### 30-Second Example

```python
from ocrrouter import process_document

# One-liner document conversion
result = process_document(
    "document.pdf",
    "output/",
    backend="deepseek",
    openai_api_key="your-api-key"
)

print(result["markdown"])
```

### Basic Usage

```python
from ocrrouter import DocumentPipeline, Settings

# Configure pipeline
settings = Settings(
    backend="deepseek",
    openai_base_url="https://api.example.com/v1",
    openai_api_key="your-api-key",
    output_mode="all"  # layout + OCR
)

# Process document
pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("document.pdf", "output/")

# Access results
print(f"Markdown: {result['markdown'][:100]}...")
print(f"Output directory: {result['output_dir']}")
```

### Async Processing

```python
# Async processing for better performance
result = await pipeline.aio_process("document.pdf", "output/")

# Batch processing with concurrency control
results = await pipeline.aio_process_batch(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    "output/",
    session_id="batch-001"
)
```

## Key Features

### 1. Multiple Expert Backends

Each backend is optimized for different document types:

| Backend | Layout | OCR | Best For |
|---------|:------:|:---:|----------|
| **MinerU** | ✓ | ✓ | Academic papers, complex layouts, formulas |
| **DeepSeek** | ✓ | ✓ | General documents, efficiency, grounding mode |
| **DotsOCR** | ✓ | ✓ | Flexible extraction (one-step or two-step) |
| **PaddleOCR** | — | ✓ | Fast OCR, multilingual support |
| **Hunyuan** | — | ✓ | Markdown-optimized output |
| **GeneralVLM** | — | ✓ | GPT-4V, Claude, Gemini, custom VLMs |

### 2. Composite Mode (Mix & Match)

Combine the strengths of different models:

```python
settings = Settings(
    backend="composite",
    layout_model="mineru",      # Best layout detection
    ocr_model="paddleocr",      # Fast OCR extraction
)
```

**Why use composite mode?**
- Optimize for cost vs quality
- Leverage each model's strengths
- Example: MinerU's excellent layout + PaddleOCR's speed
- 2-3x faster than single-model approaches in many cases

### 3. Three Output Modes

Control processing behavior:

```python
# Full layout + OCR (default)
Settings(output_mode="all")

# Layout detection only
Settings(output_mode="layout_only")

# Direct OCR without layout analysis
Settings(output_mode="ocr_only")
```

### 4. Rich Output Formats

Multiple output files for different use cases:

- **Markdown** (`.md`) — Human-readable converted text
- **Layout PDF** (`_layout.pdf`) — Visual layout with bounding boxes
- **Model JSON** (`_model.json`) — Raw model output
- **Middle JSON** (`_middle.json`) — Processed structural data
- **Content List** (`_content_list.json`) — Simplified flat structure
- **Images** — Extracted figures, tables, equations

## Use Cases

### Academic Research
Extract formulas, citations, and complex layouts from research papers and theses:
```python
settings = Settings(
    backend="mineru",
    formula_enable=True,
    table_merge_enable=True  # Cross-page table merging
)
```

### Business Documents
Parse invoices, contracts, and forms with table extraction:
```python
settings = Settings(
    backend="deepseek",
    table_enable=True,
    output_mode="all"
)
```

### Document Digitization
Batch process archives with multilingual support:
```python
settings = Settings(
    backend="composite",
    layout_model="deepseek",
    ocr_model="paddleocr",  # Strong multilingual support
    max_concurrency=10
)
```

### AI/ML Pipelines
Extract structured data for RAG or training:
```python
settings = Settings(
    backend="deepseek",
    dump_content_list=True,  # Simplified JSON for ML
    dump_middle_json=True     # Structured data
)
```

## Backend Selection Guide

### How to Choose?

**Need layout detection + OCR?**
- Academic/Scientific → **MinerU** (best formula extraction)
- General documents → **DeepSeek** (efficient grounding mode)
- Flexible extraction → **DotsOCR** (one-step or two-step)

**Need OCR only?**
- Fast processing → **PaddleOCR**
- Markdown-focused → **Hunyuan**
- Use GPT-4/Claude → **GeneralVLM**

**Want to optimize cost/speed?**
- Use **Composite Mode**: strong layout + fast OCR

See [Backend Guide](docs/BACKENDS.md) for detailed comparison.

## Documentation

- **[Backend Guide](docs/BACKENDS.md)** — Model comparison and selection
- **[Examples](docs/EXAMPLES.md)** — Code examples and recipes
- **[API Reference](docs/API.md)** — Complete API documentation
- **[Configuration](docs/CONFIGURATION.md)** — Settings and environment variables
- **[Output Formats](docs/OUTPUT_FORMATS.md)** — Understanding output files

## Configuration

OCRRouter uses explicit configuration (no automatic .env loading):

```python
from ocrrouter import Settings

# Method 1: Settings object
settings = Settings(
    backend="deepseek",
    openai_api_key="your-key",
    max_concurrency=20,
    http_timeout=120,
    max_retries=3
)

# Method 2: Constructor arguments
pipeline = DocumentPipeline(
    backend="deepseek",
    openai_api_key="your-key"
)

# Method 3: Settings with overrides
pipeline = DocumentPipeline(
    settings=settings,
    max_concurrency=50  # Override
)
```

See [Configuration Guide](docs/CONFIGURATION.md) for all available settings.

## Advanced Features

### Observability with Langfuse

```python
from langfuse import Langfuse
from ocrrouter import DocumentPipeline, Settings

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

settings = Settings(backend="deepseek", openai_api_key="your-key")
pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)

# Traces appear in Langfuse dashboard
result = await pipeline.aio_process("document.pdf", "output/")
```

### Error Handling & Debug Mode

```python
settings = Settings(
    backend="deepseek",
    max_retries=5,
    debug=True,           # Save failed requests
    debug_dir="./debug"   # Debug output location
)
```

### Direct Backend Access

```python
from ocrrouter import get_backend, Settings

settings = Settings(openai_api_key="your-key")
backend = get_backend("mineru", settings=settings)

# Advanced control
middle_json, model_output = await backend.analyze(pdf_bytes, image_writer)
```

## Examples

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for comprehensive examples including:
- Basic document processing
- Batch processing with concurrency
- Composite mode configurations
- FastAPI integration
- Custom pipelines
- Use case-specific recipes

Or check out the demo scripts in `demo/`:
- `demo/quickstart.py` — Minimal example
- `demo/composite_mode.py` — Composite mode showcase
- `demo/demo.py` — Comprehensive demo

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- VLM server access (for backends requiring API calls)
- See [pyproject.toml](pyproject.toml) for full dependency list

## Installation

```bash
# From PyPI
pip install ocrrouter

# From source
git clone https://github.com/yourusername/ocrrouter.git
cd ocrrouter
pip install -e .
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/ocrrouter/issues)
- **Documentation**: Full documentation at [docs/](docs/)
- **Examples**: See [docs/EXAMPLES.md](docs/EXAMPLES.md) and [demo/](demo/)

## Acknowledgments

We would like to thank the following projects for providing code and models:

- [Dots.OCR](https://github.com/rednote-hilab/dots.ocr)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)
- [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)

---

**Built with ❤️ for document processing needs**
