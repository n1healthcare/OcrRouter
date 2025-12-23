# OCRRouter Demo Scripts

This directory contains example scripts demonstrating various OCRRouter features.

## Available Demos

### 1. quickstart.py — Simplest Example

**Purpose**: Get started with OCRRouter in the simplest way possible

**What it demonstrates**:
- Basic document processing
- Simple configuration with Settings
- Environment variable usage
- Error handling

**Use case**: First-time users, quick testing

**Run**:
```bash
# Set environment variables
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="your-api-key"

# Run demo
python demo/quickstart.py
```

**What you'll learn**:
- How to create a pipeline
- How to process a single document
- How to access results

---

### 2. composite_mode.py — Composite Mode Showcase

**Purpose**: Demonstrate OCRRouter's unique composite mode feature

**What it demonstrates**:
- Mixing layout detection and OCR models
- Multiple composite configurations
- Async batch processing
- Performance comparison

**Use case**: Understanding composite mode benefits

**Run**:
```bash
# Set environment variables
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="your-api-key"

# Run demo
python demo/composite_mode.py

# Choose mode:
#   1 - Simple example (single configuration)
#   2 - Comparison (multiple configurations in parallel)
```

**What you'll learn**:
- How to use composite mode
- When to use different model combinations
- Performance trade-offs
- Best practices for different use cases

**Example configurations**:
```python
# Quality layout + Fast OCR
Settings(
    backend="composite",
    layout_model="mineru",
    ocr_model="paddleocr"
)

# Fast layout + Premium OCR
Settings(
    backend="composite",
    layout_model="deepseek",
    ocr_model="generalvlm",
    generalvlm_model_name="gpt-4-vision-preview"
)
```

---

### 3. demo.py — Comprehensive Demo

**Purpose**: Comprehensive example showing advanced features

**What it demonstrates**:
- Batch processing
- Langfuse observability integration
- Advanced configuration
- Error handling and logging

**Use case**: Production-ready patterns

**Run**:
```bash
# Set environment variables
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="your-api-key"

# Optional: Langfuse for observability
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"

# Run demo
python demo/demo.py
```

**What you'll learn**:
- Batch processing patterns
- Langfuse integration
- Production-ready configuration
- Advanced error handling

---

## Prerequisites

### 1. Install OCRRouter

```bash
pip install ocrrouter
```

### 2. Set Up Environment Variables

Create a `.env` file in the demo directory or set environment variables:

```bash
# Required
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="your-api-key"

# Optional (for Langfuse observability)
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

**Using .env file**:
```bash
# Install python-dotenv
pip install python-dotenv

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Prepare Test Documents

Place PDF files in the demo directory or update the `pdf_path` variables in the scripts.

---

## Quick Start Guide

### First Time Using OCRRouter?

1. **Start with**: `quickstart.py`
2. **Then try**: `composite_mode.py` (simple mode)
3. **For production**: `demo.py`

### Want to See Composite Mode?

1. **Run**: `composite_mode.py`
2. **Choose**: Option 2 (comparison mode)
3. **Compare**: Different model combinations

### Building a Production System?

1. **Study**: `demo.py` for patterns
2. **Read**: [../docs/EXAMPLES.md](../docs/EXAMPLES.md) for more examples
3. **Refer**: [../docs/API.md](../docs/API.md) for API details

---

## Expected Output

All demos create output in the `output/` directory:

```
demo/
├── quickstart.py
├── composite_mode.py
├── demo.py
└── output/
    ├── document_name/          # From quickstart.py
    │   └── vlm/
    │       ├── document_name.md
    │       ├── document_name_layout.pdf
    │       ├── document_name_middle.json
    │       ├── document_name_model.json
    │       ├── document_name_content_list.json
    │       └── images/
    │
    ├── composite_simple/       # From composite_mode.py (simple)
    │   └── ...
    │
    └── composite_comparison/   # From composite_mode.py (comparison)
        ├── mineru_paddleocr/
        │   └── ...
        ├── deepseek_deepseek/
        │   └── ...
        └── dotsocr_generalvlm/
            └── ...
```

---

## Common Issues

### Issue: "OPENAI_API_KEY not set"

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

Or update the script to use hardcoded values (for testing only):
```python
openai_api_key = "your-api-key"
```

---

### Issue: "File not found: document.pdf"

**Solution**: Update the `pdf_path` variable in the script:
```python
pdf_path = "path/to/your/document.pdf"
```

---

### Issue: Connection errors

**Solution**: Check your `OPENAI_BASE_URL`:
```bash
# Make sure the URL is correct
export OPENAI_BASE_URL="https://api.example.com/v1"

# Test connection
curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

### Issue: Slow processing

**Solution**: Adjust concurrency settings:
```python
settings = Settings(
    backend="deepseek",
    max_concurrency=10,  # Increase for faster processing
    http_timeout=300,    # Increase for large documents
)
```

---

## Customizing the Demos

### Change Backend

```python
# In any script, update the backend setting:
settings = Settings(
    backend="mineru",  # or "deepseek", "dotsocr", etc.
    # ...
)
```

### Process Specific Pages

```python
# Process only pages 0-4 (first 5 pages)
result = pipeline.process(
    "document.pdf",
    "output/",
    start_page_id=0,
    end_page_id=5
)
```

### Control Output Files

```python
settings = Settings(
    # ...
    dump_md=True,            # Generate markdown
    dump_content_list=True,  # Generate content list
    dump_middle_json=False,  # Skip middle JSON
    dump_model_output=False, # Skip model output
    draw_layout_bbox=False,  # Skip layout PDF
)
```

### Add Langfuse Observability

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)
```

---

## Next Steps

- **More Examples**: See [../docs/EXAMPLES.md](../docs/EXAMPLES.md)
- **API Reference**: See [../docs/API.md](../docs/API.md)
- **Backend Guide**: See [../docs/BACKENDS.md](../docs/BACKENDS.md)
- **Configuration**: See [../docs/CONFIGURATION.md](../docs/CONFIGURATION.md)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ocrrouter/issues)
- **Documentation**: [../docs/](../docs/)
- **Main README**: [../README.md](../README.md)
