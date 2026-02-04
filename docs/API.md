# API Reference

Complete API documentation for OCRRouter.

## Table of Contents

- [Quick Start Function](#quick-start-function)
- [DocumentPipeline Class](#documentpipeline-class)
- [Settings Class](#settings-class)
- [Backend Access](#backend-access)
- [Return Values](#return-values)
- [Observability](#observability)

---

## Quick Start Function

### `process_document()`

Simple one-liner function for document processing.

```python
def process_document(
    input_path: str | bytes,
    output_dir: str | None = None,
    settings: Settings | None = None,
    filename: str | None = None,
    **overrides: Any,
) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_path` | `str \| bytes` | Path to the input PDF/image file, or raw bytes of the document |
| `output_dir` | `str \| None` | Directory for output files. If `None`, uses a temporary directory |
| `settings` | `Settings \| None` | Optional Settings object with configuration |
| `filename` | `str \| None` | Required when `input_path` is bytes. The filename for the document |
| `**overrides` | `Any` | Configuration overrides (backend, openai_api_key, etc.) |

**Returns:**
- `dict`: Processing results containing `markdown`, `middle_json`, `content_list`, etc.

**Example:**
```python
from ocrrouter import process_document

# From file path
result = process_document(
    "document.pdf",
    "output/",
    backend="deepseek",
    openai_api_key="sk-...",
)

print(result["markdown"])

# From bytes (filename required)
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

result = process_document(
    pdf_bytes,
    "output/",
    filename="document.pdf",
    backend="deepseek",
    openai_api_key="sk-...",
)

# Using temporary directory (output_dir=None)
result = process_document(
    "document.pdf",
    output_dir=None,  # Uses temp directory
    backend="deepseek",
    openai_api_key="sk-...",
)
```

---

## DocumentPipeline Class

Main pipeline class that orchestrates the entire document processing workflow.

### Constructor

```python
DocumentPipeline(
    settings: Settings | None = None,
    langfuse: Any | None = None,
    **overrides: Any,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `settings` | `Settings \| None` | Settings object with configuration. If not provided, created from overrides |
| `langfuse` | `Any \| None` | Optional Langfuse client for observability |
| `**overrides` | `Any` | Configuration overrides applied on top of settings |

**Common Overrides:**
- `backend` — Backend selection (`"mineru"`, `"deepseek"`, etc.)
- `openai_base_url` — VLM server URL
- `openai_api_key` — API authentication key
- `max_concurrency` — Maximum concurrent requests
- `http_timeout` — HTTP request timeout in seconds

**Example:**
```python
from ocrrouter import DocumentPipeline, Settings

# Method 1: Constructor arguments only
pipeline = DocumentPipeline(
    backend="deepseek",
    openai_api_key="your-key",
    max_concurrency=20
)

# Method 2: Settings object
settings = Settings(backend="mineru", openai_api_key="your-key")
pipeline = DocumentPipeline(settings=settings)

# Method 3: Settings with overrides
pipeline = DocumentPipeline(
    settings=settings,
    max_concurrency=50  # Override settings value
)
```

---

### Methods

#### `process()`

Process a document synchronously.

```python
def process(
    input_path: str | Path | bytes,
    output_dir: str | None = None,
    start_page_id: int | None = None,
    end_page_id: int | None = None,
    filename: str | None = None,
    **options: Any,
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path \| bytes` | — | Path to the input file (PDF or image), or raw bytes |
| `output_dir` | `str \| None` | `None` | Directory to write output files. If `None`, uses a temporary directory |
| `start_page_id` | `int \| None` | `None` | Starting page index (0-based) |
| `end_page_id` | `int \| None` | `None` | Ending page index (0-based) |
| `filename` | `str \| None` | `None` | Required when `input_path` is bytes. The filename for the document |
| `**options` | `Any` | — | Additional processing options |

**Returns:**
- `dict`: Processing results

**Example:**
```python
pipeline = DocumentPipeline(backend="deepseek", openai_api_key="key")

# Process entire document
result = pipeline.process("document.pdf", "output/")

# Process specific page range (pages 0-4)
result = pipeline.process(
    "document.pdf",
    "output/",
    start_page_id=0,
    end_page_id=5
)

# Process from bytes
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

result = pipeline.process(pdf_bytes, "output/", filename="document.pdf")

# Use temporary directory (don't save locally)
result = pipeline.process("document.pdf", output_dir=None)
print(result["markdown"])  # Access results directly
```

---

#### `aio_process()`

Process a document asynchronously (async/await).

```python
async def aio_process(
    input_path: str | Path | bytes,
    output_dir: str | None = None,
    start_page_id: int | None = None,
    end_page_id: int | None = None,
    session_id: str | None = None,
    filename: str | None = None,
    **options: Any,
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path \| bytes` | — | Path to the input file (PDF or image), or raw bytes |
| `output_dir` | `str \| None` | `None` | Directory to write output files. If `None`, uses a temporary directory |
| `start_page_id` | `int \| None` | `None` | Starting page index (0-based) |
| `end_page_id` | `int \| None` | `None` | Ending page index (0-based) |
| `session_id` | `str \| None` | `None` | Optional session ID for grouping traces in batch processing |
| `filename` | `str \| None` | `None` | Required when `input_path` is bytes. The filename for the document |
| `**options` | `Any` | — | Additional processing options |

**Returns:**
- `dict`: Processing results

**Example:**
```python
import asyncio

async def convert():
    pipeline = DocumentPipeline(backend="deepseek", openai_api_key="key")
    result = await pipeline.aio_process("document.pdf", "output/")
    return result

result = asyncio.run(convert())

# From bytes
async def convert_bytes():
    pipeline = DocumentPipeline(backend="deepseek", openai_api_key="key")
    with open("document.pdf", "rb") as f:
        pdf_bytes = f.read()
    result = await pipeline.aio_process(pdf_bytes, "output/", filename="document.pdf")
    return result

# Using temporary directory
async def convert_temp():
    pipeline = DocumentPipeline(backend="deepseek", openai_api_key="key")
    result = await pipeline.aio_process("document.pdf", output_dir=None)
    return result["markdown"]  # Access results directly without saving
```

---

#### `process_batch()`

Process multiple documents synchronously.

```python
def process_batch(
    input_paths: list[str | Path],
    output_dir: str,
    start_page_id: int | None = None,
    end_page_id: int | None = None,
    session_id: str | None = None,
    **options: Any,
) -> list[dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_paths` | `list[str \| Path]` | — | List of input file paths |
| `output_dir` | `str` | — | Directory to write output files |
| `start_page_id` | `int \| None` | `None` | Starting page index for all documents |
| `end_page_id` | `int \| None` | `None` | Ending page index for all documents |
| `session_id` | `str \| None` | `None` | Optional session ID for grouping traces |
| `**options` | `Any` | — | Additional processing options |

**Returns:**
- `list[dict]`: List of processing results (one per document)

**Example:**
```python
from pathlib import Path

pipeline = DocumentPipeline(backend="deepseek", openai_api_key="key")

# Find all PDFs
pdf_files = list(Path("input/").glob("*.pdf"))

# Process batch
results = pipeline.process_batch(pdf_files, "output/")

print(f"Processed {len(results)} documents")
```

---

#### `aio_process_batch()`

Process multiple documents asynchronously.

```python
async def aio_process_batch(
    input_paths: list[str | Path],
    output_dir: str,
    start_page_id: int | None = None,
    end_page_id: int | None = None,
    session_id: str | None = None,
    **options: Any,
) -> list[dict]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_paths` | `list[str \| Path]` | — | List of input file paths |
| `output_dir` | `str` | — | Directory to write output files |
| `start_page_id` | `int \| None` | `None` | Starting page index for all documents |
| `end_page_id` | `int \| None` | `None` | Ending page index for all documents |
| `session_id` | `str \| None` | `None` | Optional session ID for grouping traces |
| `**options` | `Any` | — | Additional processing options |

**Returns:**
- `list[dict]`: List of processing results (one per document)

**Example:**
```python
import asyncio
from pathlib import Path

async def batch_convert():
    pipeline = DocumentPipeline(
        backend="deepseek",
        openai_api_key="key",
        max_concurrency=10  # Process 10 in parallel
    )

    pdf_files = list(Path("input/").glob("*.pdf"))

    results = await pipeline.aio_process_batch(
        pdf_files,
        "output/",
        session_id="batch-001"
    )

    return results

results = asyncio.run(batch_convert())
```

---

### Properties

#### `settings`

Get the current settings.

```python
@property
def settings(self) -> Settings
```

**Returns:**
- `Settings`: The current settings object

**Example:**
```python
pipeline = DocumentPipeline(backend="deepseek")
print(pipeline.settings.backend)  # "deepseek"
print(pipeline.settings.max_concurrency)  # 20 (default)
```

---

#### `backend`

Get the backend instance (lazily initialized).

```python
@property
def backend(self) -> BaseModelBackend
```

**Returns:**
- `BaseModelBackend`: The current backend instance

**Example:**
```python
pipeline = DocumentPipeline(backend="mineru", openai_api_key="key")
backend = pipeline.backend
print(type(backend).__name__)  # "MinerUBackend"
```

---

## Settings Class

Configuration class using Pydantic for validation.

```python
from ocrrouter import Settings

settings = Settings(
    backend="deepseek",
    openai_api_key="your-key",
    # ... other options
)
```

### Key Settings Categories

#### API & Network

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `openai_base_url` | `str \| None` | `None` | VLM server URL |
| `openai_api_key` | `str \| None` | `None` | API authentication key |
| `http_timeout` | `int` | `120` | HTTP request timeout (seconds) |
| `max_concurrency` | `int` | `20` | Maximum concurrent requests |
| `max_retries` | `int` | `3` | Maximum retry attempts |

#### Backend Selection

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | `Literal[...]` | `"mineru"` | Backend to use: `mineru`, `deepseek`, `dotsocr`, `composite`, `hunyuanocr`, `generalvlm` |

#### Composite Backend

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `layout_model` | `Literal[...]` | `"mineru"` | Layout model: `mineru`, `deepseek`, `dotsocr` |
| `ocr_model` | `Literal[...]` | `"mineru"` | OCR model: `mineru`, `deepseek`, `dotsocr`, `hunyuanocr`, `paddleocr`, `generalvlm` |

#### Model Names

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `mineru_model_name` | `str` | `"mineru-2.5"` | MinerU model name |
| `deepseek_model_name` | `str` | `"deepseek-ocr"` | DeepSeek-OCR 2 model name |
| `dotsocr_model_name` | `str` | `"dots-ocr"` | DotsOCR model name |
| `hunyuan_model_name` | `str` | `"hunyuan-ocr"` | Hunyuan model name |
| `paddleocr_model_name` | `str` | `"paddle-ocr"` | PaddleOCR model name |
| `generalvlm_model_name` | `str` | `"gemini-2.5-pro"` | General VLM model name |

#### Processing Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_mode` | `Literal[...]` | `"all"` | Output mode: `all`, `layout_only`, `ocr_only` |
| `formula_enable` | `bool` | `True` | Enable formula extraction |
| `table_enable` | `bool` | `True` | Enable table extraction |
| `table_merge_enable` | `bool` | `True` | Enable cross-page table merging |
| `start_page` | `int` | `0` | Starting page index (0-based) |
| `end_page` | `int \| None` | `None` | Ending page index (None = all pages) |

#### Output Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `dump_md` | `bool` | `True` | Output markdown file |
| `dump_middle_json` | `bool` | `True` | Output middle JSON file |
| `dump_model_output` | `bool` | `True` | Output model JSON file |
| `dump_content_list` | `bool` | `True` | Output content list JSON |
| `dump_orig_pdf` | `bool` | `False` | Save original PDF |
| `draw_layout_bbox` | `bool` | `True` | Draw layout bounding boxes |

#### Debug Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `debug` | `bool` | `False` | Enable debug mode |
| `debug_dir` | `Path \| None` | `None` | Debug output directory |
| `log_level` | `str` | `"INFO"` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

See [Configuration Guide](CONFIGURATION.md) for complete reference.

---

## Backend Access

### `get_backend()`

Get a backend instance directly (advanced usage).

```python
from ocrrouter import get_backend, Settings

def get_backend(
    backend_name: str,
    settings: Settings
) -> BaseModelBackend
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend_name` | `str` | Backend name: `mineru`, `deepseek`, `dotsocr`, `composite`, `hunyuanocr`, `generalvlm` |
| `settings` | `Settings` | Settings object with configuration |

**Returns:**
- `BaseModelBackend`: Backend instance

**Example:**
```python
from ocrrouter import get_backend, Settings
from ocrrouter.utils.io.writers import FileBasedDataWriter

settings = Settings(openai_api_key="your-key")
backend = get_backend("mineru", settings=settings)

# Direct backend usage
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

image_writer = FileBasedDataWriter("output/images")
middle_json, model_output = await backend.analyze(pdf_bytes, image_writer)
```

---

## Return Values

### Processing Result Dictionary

All processing methods return a dictionary with the following structure:

```python
{
    "pdf_file_name": str,      # Document name (without extension)
    "output_dir": str,         # Output directory path
    "middle_json": list[list], # Processed structural data (pages → blocks)
    "markdown": str,           # Generated markdown text
    "content_list": list[dict] # Simplified content blocks
}
```

#### Return Value Details

| Key | Type | Description |
|-----|------|-------------|
| `pdf_file_name` | `str` | Document name without extension |
| `output_dir` | `str` | Output directory path where files were written |
| `middle_json` | `list[list]` | Structured document representation (pages → blocks) |
| `markdown` | `str` | Generated markdown text |
| `content_list` | `list[dict]` | Simplified flat list of content blocks |

#### Example:

```python
result = pipeline.process("document.pdf", "output/")

# Access results
print(result["pdf_file_name"])  # "document"
print(result["output_dir"])     # "output/document/vlm"
print(result["markdown"][:100]) # First 100 chars of markdown
print(len(result["middle_json"]))     # Number of pages
print(len(result["content_list"]))    # Number of content blocks
```

---

## Observability

### Langfuse Integration

OCRRouter supports Langfuse for observability and tracing.

**Important**: OCRRouter creates spans within the provided Langfuse client but does not manage or update traces. Trace management (creating, updating, finalizing) is the responsibility of the parent application. OCRRouter is designed to be one component in a larger pipeline.

```python
from langfuse import Langfuse
from ocrrouter import DocumentPipeline, Settings

# Create Langfuse client (owned by parent app)
langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# Pass to pipeline
settings = Settings(backend="deepseek", openai_api_key="key")
pipeline = DocumentPipeline(settings=settings, langfuse=langfuse)

# Process with tracing
result = await pipeline.aio_process(
    "document.pdf",
    "output/",
    session_id="production-batch-001"
)

# Flush and shutdown (parent app's responsibility)
langfuse.shutdown()
```

### Session Management

Group related processing operations using `session_id`:

```python
# Batch processing with session
results = await pipeline.aio_process_batch(
    pdf_files,
    "output/",
    session_id="batch-2024-01-15"  # Groups all in same session
)
```

Session IDs help track:
- Batch processing jobs
- Related document sets
- Multi-document workflows

---

## Type Signatures

For type checking and IDE support:

```python
from ocrrouter import DocumentPipeline, Settings, process_document, get_backend
from typing import Any
from pathlib import Path

# Function signatures
def process_document(
    input_path: str | bytes,
    output_dir: str | None = None,
    settings: Settings | None = None,
    filename: str | None = None,
    **overrides: Any,
) -> dict: ...

# Class signatures
class DocumentPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        langfuse: Any | None = None,
        **overrides: Any,
    ) -> None: ...

    def process(
        self,
        input_path: str | Path | bytes,
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        filename: str | None = None,
        **options: Any,
    ) -> dict: ...

    async def aio_process(
        self,
        input_path: str | Path | bytes,
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        session_id: str | None = None,
        filename: str | None = None,
        **options: Any,
    ) -> dict: ...

    def process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        session_id: str | None = None,
        **options: Any,
    ) -> list[dict]: ...

    async def aio_process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | None = None,
        start_page_id: int | None = None,
        end_page_id: int | None = None,
        session_id: str | None = None,
        **options: Any,
    ) -> list[dict]: ...

    @property
    def settings(self) -> Settings: ...

    @property
    def backend(self) -> Any: ...  # BaseModelBackend
```

---

## See Also

- [Examples](EXAMPLES.md) — Practical code examples
- [Configuration](CONFIGURATION.md) — Complete settings reference
- [Backend Guide](BACKENDS.md) — Backend selection and comparison
- [Output Formats](OUTPUT_FORMATS.md) — Understanding output files
