# Configuration Guide

Complete reference for configuring OCRRouter.

## Table of Contents

- [Configuration Methods](#configuration-methods)
- [Settings Reference](#settings-reference)
  - [API & Network](#api--network)
  - [Backend Selection](#backend-selection)
  - [Composite Backend](#composite-backend)
  - [VLM Models](#vlm-models)
  - [Processing Options](#processing-options)
  - [Page Range](#page-range)
  - [Output Options](#output-options)
  - [Logging & Debug](#logging--debug)
- [Environment Variables](#environment-variables)
- [Configuration Patterns](#configuration-patterns)

---

## Configuration Methods

OCRRouter uses **explicit configuration** (no automatic .env loading). You can configure in three ways:

### Method 1: Constructor Arguments

Pass configuration directly to `DocumentPipeline`:

```python
from ocrrouter import DocumentPipeline

pipeline = DocumentPipeline(
    backend="deepseek",
    openai_api_key="your-key",
    max_concurrency=20,
    http_timeout=120
)
```

### Method 2: Settings Object

Create a reusable `Settings` object:

```python
from ocrrouter import DocumentPipeline, Settings

settings = Settings(
    backend="deepseek",
    openai_api_key="your-key",
    max_concurrency=20
)

pipeline = DocumentPipeline(settings=settings)
```

### Method 3: Settings with Overrides

Combine Settings with constructor overrides:

```python
# Base settings
settings = Settings(
    backend="mineru",
    openai_api_key="your-key"
)

# Create pipeline with overrides
pipeline = DocumentPipeline(
    settings=settings,
    max_concurrency=50,  # Override
    debug=True            # Override
)
```

---

## Settings Reference

Complete reference for all configuration options.

### API & Network

#### `openai_base_url`

- **Type**: `str | None`
- **Default**: `None`
- **Description**: VLM server URL (OpenAI-compatible endpoint)
- **Example**:
  ```python
  Settings(openai_base_url="https://api.openai.com/v1")
  ```
- **Notes**: Required for API-based backends. Different for each VLM provider:
  - OpenAI: `https://api.openai.com/v1`
  - OpenRouter: `https://openrouter.ai/api/v1`
  - Custom: Your server URL

---

#### `openai_api_key`

- **Type**: `str | None`
- **Default**: `None`
- **Description**: API authentication key
- **Example**:
  ```python
  Settings(openai_api_key="sk-...")
  ```
- **Notes**: Required for API-based backends. Keep secure, never commit to version control.

---

#### `http_timeout`

- **Type**: `int`
- **Default**: `120`
- **Description**: HTTP request timeout in seconds
- **Example**:
  ```python
  Settings(http_timeout=300)  # 5 minutes
  ```
- **Notes**: Increase for large documents or slow models.

---

#### `max_concurrency`

- **Type**: `int`
- **Default**: `20`
- **Description**: Maximum concurrent requests
- **Example**:
  ```python
  Settings(max_concurrency=10)  # Lower for resource-constrained systems
  ```
- **Notes**: Controls parallelism in batch processing. Higher = faster but more resource usage.

---

#### `max_retries`

- **Type**: `int`
- **Default**: `3`
- **Description**: Maximum retry attempts for failed requests
- **Example**:
  ```python
  Settings(max_retries=5)  # More retries for unreliable networks
  ```
- **Notes**: Uses exponential backoff between retries.

---

### Backend Selection

#### `backend`

- **Type**: `Literal["mineru", "deepseek", "dotsocr", "composite", "hunyuanocr", "generalvlm"]`
- **Default**: `"mineru"`
- **Description**: Document processing backend to use
- **Valid Values**:
  - `"mineru"` — Two-step extraction, best for academic papers
  - `"deepseek"` — Grounding mode, efficient for general documents
  - `"dotsocr"` — Flexible one-step or two-step
  - `"composite"` — Mix layout + OCR from different models
  - `"hunyuanocr"` — OCR-only, markdown-focused
  - `"generalvlm"` — Use GPT-4, Claude, Gemini, etc.
- **Example**:
  ```python
  Settings(backend="deepseek")
  ```
- **See Also**: [Backend Guide](BACKENDS.md) for comparison

---

### Composite Backend

#### `layout_model`

- **Type**: `Literal["mineru", "deepseek", "dotsocr"]`
- **Default**: `"mineru"`
- **Description**: Model to use for layout detection in composite mode
- **Valid Values**:
  - `"mineru"` — Best layout detection accuracy
  - `"deepseek"` — Fast grounding mode
  - `"dotsocr"` — Flexible extraction
- **Example**:
  ```python
  Settings(
      backend="composite",
      layout_model="mineru"
  )
  ```
- **Notes**: Only used when `backend="composite"`

---

#### `ocr_model`

- **Type**: `Literal["mineru", "deepseek", "dotsocr", "hunyuanocr", "paddleocr", "generalvlm"]`
- **Default**: `"mineru"`
- **Description**: Model to use for OCR extraction in composite mode
- **Valid Values**:
  - `"mineru"` — Excellent OCR quality
  - `"deepseek"` — Fast and accurate
  - `"dotsocr"` — Good balance
  - `"hunyuanocr"` — Markdown-optimized
  - `"paddleocr"` — Fast, multilingual
  - `"generalvlm"` — Use GPT/Claude/Gemini
- **Example**:
  ```python
  Settings(
      backend="composite",
      layout_model="mineru",
      ocr_model="paddleocr"  # Fast OCR
  )
  ```
- **Notes**: Only used when `backend="composite"`

---

### VLM Models

#### `mineru_model_name`

- **Type**: `str`
- **Default**: `"mineru-2.5"`
- **Description**: MinerU VLM model name
- **Example**:
  ```python
  Settings(
      backend="mineru",
      mineru_model_name="mineru-2.5"
  )
  ```

---

#### `deepseek_model_name`

- **Type**: `str`
- **Default**: `"deepseek-ocr"`
- **Description**: DeepSeek-OCR model name
- **Example**:
  ```python
  Settings(
      backend="deepseek",
      deepseek_model_name="deepseek-ocr"
  )
  ```

---

#### `dotsocr_model_name`

- **Type**: `str`
- **Default**: `"dots-ocr"`
- **Description**: DotsOCR model name
- **Example**:
  ```python
  Settings(
      backend="dotsocr",
      dotsocr_model_name="dots-ocr"
  )
  ```

---

#### `dotsocr_extraction_mode`

- **Type**: `Literal["one_step", "two_step"]`
- **Default**: `"one_step"`
- **Description**: DotsOCR extraction mode
- **Valid Values**:
  - `"one_step"` — Layout + OCR in single call (faster)
  - `"two_step"` — Separate layout and OCR calls (more accurate)
- **Example**:
  ```python
  Settings(
      backend="dotsocr",
      dotsocr_extraction_mode="two_step"
  )
  ```

---

#### `hunyuan_model_name`

- **Type**: `str`
- **Default**: `"hunyuan-ocr"`
- **Description**: Hunyuan-OCR model name
- **Example**:
  ```python
  Settings(
      backend="hunyuanocr",
      hunyuan_model_name="hunyuan-ocr"
  )
  ```

---

#### `paddleocr_model_name`

- **Type**: `str`
- **Default**: `"paddle-ocr"`
- **Description**: PaddleOCR model name
- **Example**:
  ```python
  Settings(
      backend="composite",
      ocr_model="paddleocr",
      paddleocr_model_name="paddle-ocr"
  )
  ```

---

#### `generalvlm_model_name`

- **Type**: `str`
- **Default**: `"gemini-2.5-pro"`
- **Description**: General VLM model name (supports GPT, Claude, Gemini, etc.)
- **Example**:
  ```python
  # Using GPT-4V
  Settings(
      backend="generalvlm",
      generalvlm_model_name="gpt-4-vision-preview",
      openai_api_key="sk-...",
      openai_base_url="https://api.openai.com/v1"
  )

  # Using Claude via OpenRouter
  Settings(
      backend="generalvlm",
      generalvlm_model_name="anthropic/claude-3.5-sonnet",
      openai_api_key="sk-or-...",
      openai_base_url="https://openrouter.ai/api/v1"
  )

  # Using Gemini
  Settings(
      backend="generalvlm",
      generalvlm_model_name="gemini-2.5-pro",
      openai_api_key="your-gemini-key",
      openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
  )
  ```

---

### Processing Options

#### `output_mode`

- **Type**: `Literal["all", "layout_only", "ocr_only"]`
- **Default**: `"all"`
- **Description**: Output mode controlling processing and outputs
- **Valid Values**:
  - `"all"` — Full layout detection + OCR extraction
  - `"layout_only"` — Layout detection only, no markdown
  - `"ocr_only"` — Full-page OCR, markdown only (no layout)
- **Example**:
  ```python
  # Full processing
  Settings(output_mode="all")

  # Just layout structure
  Settings(output_mode="layout_only")

  # Just OCR text
  Settings(output_mode="ocr_only")
  ```
- **Notes**:
  - MinerU backend does NOT support `ocr_only` mode
  - `layout_only` skips text extraction (faster, less output)

---

#### `formula_enable`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable mathematical formula extraction
- **Example**:
  ```python
  Settings(formula_enable=True)  # Extract LaTeX formulas
  ```
- **Notes**: Formulas are output as LaTeX in markdown

---

#### `table_enable`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable table extraction and recognition
- **Example**:
  ```python
  Settings(table_enable=True)
  ```
- **Notes**: Tables are output as HTML in markdown

---

#### `table_merge_enable`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable cross-page table merging
- **Example**:
  ```python
  Settings(table_merge_enable=True)  # Merge tables split across pages
  ```
- **Notes**: Useful for documents with large tables spanning multiple pages

---

### Page Range

#### `start_page`

- **Type**: `int`
- **Default**: `0`
- **Description**: Starting page index (0-based)
- **Example**:
  ```python
  Settings(start_page=0)  # Start from first page
  Settings(start_page=5)  # Start from page 6
  ```
- **Notes**: Can be overridden per-document in `process()` methods

---

#### `end_page`

- **Type**: `int | None`
- **Default**: `None`
- **Description**: Ending page index (None = all pages)
- **Example**:
  ```python
  Settings(end_page=None)  # Process all pages
  Settings(end_page=10)    # Process up to page 10 (0-indexed)
  ```
- **Notes**:
  - `end_page` is **exclusive** (page at index not included)
  - Example: `start_page=0, end_page=5` processes pages 0,1,2,3,4 (first 5 pages)

---

### Output Options

#### `draw_layout_bbox`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Draw layout bounding boxes on output PDF
- **Example**:
  ```python
  Settings(draw_layout_bbox=True)
  ```
- **Notes**: Creates `*_layout.pdf` with visual bounding boxes for debugging

---

#### `dump_md`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Output markdown file
- **Example**:
  ```python
  Settings(dump_md=True)
  ```
- **Notes**: Creates `*.md` file with converted text

---

#### `dump_middle_json`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Output middle JSON file
- **Example**:
  ```python
  Settings(dump_middle_json=True)
  ```
- **Notes**: Creates `*_middle.json` with processed structural data

---

#### `dump_model_output`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Output model JSON file
- **Example**:
  ```python
  Settings(dump_model_output=True)
  ```
- **Notes**: Creates `*_model.json` with raw model output

---

#### `dump_orig_pdf`

- **Type**: `bool`
- **Default**: `False`
- **Description**: Save original PDF to output directory
- **Example**:
  ```python
  Settings(dump_orig_pdf=True)
  ```
- **Notes**: Useful for archiving or reference

---

#### `dump_content_list`

- **Type**: `bool`
- **Default**: `True`
- **Description**: Output content list JSON
- **Example**:
  ```python
  Settings(dump_content_list=True)
  ```
- **Notes**: Creates `*_content_list.json` with simplified flat structure

---

#### `make_md_mode`

- **Type**: `Literal["mm_markdown", "nlp_markdown", "content_list", "content_list_v2"]`
- **Default**: `"mm_markdown"`
- **Description**: Markdown generation mode
- **Valid Values**:
  - `"mm_markdown"` — Multimodal markdown (default)
  - `"nlp_markdown"` — NLP-focused markdown
  - `"content_list"` — From content list
  - `"content_list_v2"` — From content list v2
- **Example**:
  ```python
  Settings(make_md_mode="mm_markdown")
  ```

---

### Logging & Debug

#### `log_level`

- **Type**: `str`
- **Default**: `"INFO"`
- **Description**: Logging level
- **Valid Values**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Example**:
  ```python
  Settings(log_level="DEBUG")  # Verbose logging
  Settings(log_level="WARNING")  # Only warnings and errors
  ```

---

#### `debug`

- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable debug mode
- **Example**:
  ```python
  Settings(debug=True)
  ```
- **Notes**: Saves failed requests to `debug_dir` for inspection

---

#### `debug_dir`

- **Type**: `Path | None`
- **Default**: `None`
- **Description**: Debug output directory
- **Example**:
  ```python
  from pathlib import Path
  Settings(
      debug=True,
      debug_dir=Path("./debug")
  )
  ```
- **Notes**: Failed requests are saved here when `debug=True`

---

## Environment Variables

OCRRouter **does not automatically load** environment variables or `.env` files. To use environment variables, load them manually:

```python
from dotenv import load_dotenv
import os
from ocrrouter import Settings

# Load .env file
load_dotenv()

# Use environment variables
settings = Settings(
    backend=os.getenv("BACKEND", "mineru"),
    openai_base_url=os.getenv("OPENAI_BASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_concurrency=int(os.getenv("MAX_CONCURRENCY", "20")),
)
```

### Example `.env` File

See [`.env.example`](../.env.example) in the repository root for a comprehensive template.

```bash
# Backend Selection
BACKEND=deepseek

# API Configuration
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_API_KEY=your-api-key

# Composite Mode
LAYOUT_MODEL=mineru
OCR_MODEL=deepseek

# Processing
FORMULA_ENABLE=True
TABLE_ENABLE=True
TABLE_MERGE_ENABLE=True

# Performance
MAX_CONCURRENCY=20
HTTP_TIMEOUT=120
MAX_RETRIES=3

# Output
OUTPUT_MODE=all
DUMP_MD=True
DUMP_CONTENT_LIST=True

# Debug
DEBUG=False
LOG_LEVEL=INFO
```

---

## Configuration Patterns

### Development Configuration

For local development and testing:

```python
settings = Settings(
    backend="deepseek",
    openai_base_url="http://localhost:8000",
    openai_api_key="dev-key",

    # Lower concurrency for local testing
    max_concurrency=5,

    # More verbose logging
    log_level="DEBUG",

    # Enable debugging
    debug=True,
    debug_dir="./debug",

    # Minimal output
    dump_md=True,
    dump_middle_json=False,
    dump_model_output=False,
    draw_layout_bbox=False,
)
```

---

### Production Configuration

For production use with monitoring:

```python
from langfuse import Langfuse

settings = Settings(
    backend="composite",
    layout_model="mineru",
    ocr_model="deepseek",

    # Production API
    openai_base_url="https://api.production.com/v1",
    openai_api_key=os.getenv("PROD_API_KEY"),

    # Performance optimization
    max_concurrency=20,
    http_timeout=300,
    max_retries=5,

    # Full output
    output_mode="all",
    dump_md=True,
    dump_content_list=True,

    # Production logging
    log_level="INFO",
    debug=False,
)

# Add observability
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
)

pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)
```

---

### Testing Configuration

For automated testing:

```python
settings = Settings(
    backend="deepseek",
    openai_api_key="test-key",

    # Fast processing
    max_concurrency=1,
    max_retries=1,

    # Minimal output
    dump_md=True,
    dump_middle_json=False,
    dump_model_output=False,
    dump_content_list=False,
    draw_layout_bbox=False,

    # Error logging only
    log_level="ERROR",
)
```

---

### Academic Paper Configuration

Optimized for research papers:

```python
settings = Settings(
    backend="mineru",  # Best for academic content

    # Enable all features
    formula_enable=True,
    table_enable=True,
    table_merge_enable=True,

    # Full output
    output_mode="all",
    dump_md=True,
    dump_middle_json=True,
    dump_content_list=True,
    draw_layout_bbox=True,

    # Conservative processing
    max_concurrency=5,  # Don't overwhelm
    http_timeout=300,   # Large documents
)
```

---

### Batch Processing Configuration

Optimized for high-throughput:

```python
settings = Settings(
    backend="composite",
    layout_model="dotsocr",  # Fast layout
    ocr_model="paddleocr",   # Fast OCR

    # High parallelism
    max_concurrency=50,
    http_timeout=120,
    max_retries=3,

    # Minimal output for storage
    dump_md=True,
    dump_content_list=True,
    dump_middle_json=False,
    dump_model_output=False,
    draw_layout_bbox=False,

    # Quiet logging
    log_level="WARNING",
)
```

---

## See Also

- [Examples](EXAMPLES.md) — Configuration examples in context
- [Backend Guide](BACKENDS.md) — Backend selection
- [API Reference](API.md) — API documentation
- [Output Formats](OUTPUT_FORMATS.md) — Understanding output files
