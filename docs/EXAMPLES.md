# Examples

Practical code examples and recipes for using OCRRouter in various scenarios.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Intermediate Examples](#intermediate-examples)
- [Advanced Examples](#advanced-examples)
- [Use Case Examples](#use-case-examples)
  - [Academic Research](#academic-research)
  - [Business Documents](#business-documents)
  - [Document Digitization](#document-digitization)
  - [AI/ML Pipelines](#aiml-pipelines)
- [Integration Examples](#integration-examples)

---

## Basic Examples

### Example 1: Simple Document Processing

The simplest way to convert a PDF to Markdown:

```python
from ocrrouter import process_document

# One-liner conversion
result = process_document(
    "invoice.pdf",
    "output/",
    backend="deepseek",
    openai_api_key="your-api-key"
)

print(result["markdown"])
print(f"Output saved to: {result['output_dir']}")
```

**Output**:
```
{
    "markdown": "# Invoice\n\n...",
    "content_list": [...],
    "middle_json": {...},
    "output_dir": "output/invoice/vlm"
}
```

---

### Example 2: Using Settings Object

For reusable configuration:

```python
from ocrrouter import DocumentPipeline, Settings

# Create settings
settings = Settings(
    backend="deepseek",
    openai_base_url="https://api.example.com/v1",
    openai_api_key="your-api-key",
    output_mode="all"
)

# Create pipeline
pipeline = DocumentPipeline(settings=settings)

# Process document
result = pipeline.process("document.pdf", "output/")

print(f"Converted {result['output_dir']}")
```

---

### Example 3: Async Processing

For better performance with async/await:

```python
import asyncio
from ocrrouter import DocumentPipeline, Settings

async def convert_document():
    settings = Settings(
        backend="deepseek",
        openai_api_key="your-api-key"
    )

    pipeline = DocumentPipeline(settings=settings)

    # Async processing
    result = await pipeline.aio_process("document.pdf", "output/")

    return result

# Run
result = asyncio.run(convert_document())
```

---

### Example 4: Page Range Selection

Process only specific pages:

```python
settings = Settings(
    backend="mineru",
    openai_api_key="your-key",
    start_page=0,    # First page (0-indexed)
    end_page=5,      # Process pages 0-4 (first 5 pages)
)

pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("large_document.pdf", "output/")
```

---

## Intermediate Examples

### Example 5: Batch Processing with Concurrency

Process multiple documents efficiently:

```python
import asyncio
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings

async def batch_process():
    settings = Settings(
        backend="deepseek",
        openai_api_key="your-key",
        max_concurrency=10  # Process 10 documents in parallel
    )

    pipeline = DocumentPipeline(settings=settings)

    # Find all PDFs
    pdf_files = list(Path("input/").glob("*.pdf"))

    # Process in batch
    results = await pipeline.aio_process_batch(
        pdf_files,
        "output/",
        session_id="batch-001"
    )

    print(f"Processed {len(results)} documents")
    for file, result in zip(pdf_files, results):
        print(f"  {file.name} → {result['output_dir']}")

asyncio.run(batch_process())
```

---

### Example 6: Composite Mode - Basic

Mix layout detection and OCR from different models:

```python
settings = Settings(
    backend="composite",
    layout_model="mineru",      # Use MinerU for layout
    ocr_model="paddleocr",      # Use PaddleOCR for OCR
    openai_api_key="your-key"
)

pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("document.pdf", "output/")
```

---

### Example 7: Composite Mode - Advanced

Compare different composite combinations:

```python
import asyncio
from ocrrouter import DocumentPipeline, Settings

async def compare_backends():
    pdf_path = "document.pdf"

    # Configuration 1: MinerU layout + DeepSeek OCR
    config1 = Settings(
        backend="composite",
        layout_model="mineru",
        ocr_model="deepseek",
        openai_api_key="your-key"
    )

    # Configuration 2: DeepSeek layout + PaddleOCR
    config2 = Settings(
        backend="composite",
        layout_model="deepseek",
        ocr_model="paddleocr",
        openai_api_key="your-key"
    )

    # Configuration 3: DotsOCR layout + GPT-4V OCR
    config3 = Settings(
        backend="composite",
        layout_model="dotsocr",
        ocr_model="generalvlm",
        generalvlm_model_name="gpt-4-vision-preview",
        openai_api_key="your-key"
    )

    # Process with all configurations
    pipeline1 = DocumentPipeline(settings=config1)
    pipeline2 = DocumentPipeline(settings=config2)
    pipeline3 = DocumentPipeline(settings=config3)

    results = await asyncio.gather(
        pipeline1.aio_process(pdf_path, "output/mineru_deepseek"),
        pipeline2.aio_process(pdf_path, "output/deepseek_paddle"),
        pipeline3.aio_process(pdf_path, "output/dots_gpt4v")
    )

    print("All backends processed. Compare results in output/")

asyncio.run(compare_backends())
```

---

### Example 8: Custom Output Configuration

Control which files are generated:

```python
settings = Settings(
    backend="deepseek",
    openai_api_key="your-key",

    # Output files
    dump_md=True,            # Generate markdown
    dump_middle_json=True,   # Structure JSON
    dump_model_output=True,  # Raw model output
    dump_content_list=True,  # Simplified content
    dump_orig_pdf=False,     # Don't copy original PDF
    draw_layout_bbox=True,   # Visual layout PDF

    # Markdown mode
    make_md_mode="mm_markdown"  # or "nlp_markdown", "content_list"
)

pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("document.pdf", "output/")
```

---

## Advanced Examples

### Example 9: Langfuse Observability

Track document processing with Langfuse:

```python
from langfuse import Langfuse
from ocrrouter import DocumentPipeline, Settings
import asyncio

async def process_with_tracking():
    # Initialize Langfuse
    langfuse = Langfuse(
        public_key="pk-...",
        secret_key="sk-...",
        host="https://cloud.langfuse.com"
    )

    # Create pipeline with observability
    settings = Settings(
        backend="deepseek",
        openai_api_key="your-key"
    )
    pipeline = DocumentPipeline(
        settings=settings,
        langfuse=langfuse
    )

    # Process documents - traces appear in Langfuse
    result = await pipeline.aio_process(
        "document.pdf",
        "output/",
        session_id="production-batch-001"
    )

    # Flush and shutdown
    langfuse.shutdown()

    return result

asyncio.run(process_with_tracking())
```

---

### Example 10: Error Handling & Debug Mode

Handle errors gracefully with debugging:

```python
from ocrrouter import DocumentPipeline, Settings
from loguru import logger

settings = Settings(
    backend="deepseek",
    openai_api_key="your-key",

    # Retry configuration
    max_retries=5,
    http_timeout=300,  # 5-minute timeout

    # Debug mode
    debug=True,           # Save failed requests
    debug_dir="./debug",  # Debug output location

    # Logging
    log_level="DEBUG"
)

pipeline = DocumentPipeline(settings=settings)

try:
    result = pipeline.process("document.pdf", "output/")
    logger.info(f"Success: {result['output_dir']}")
except Exception as e:
    logger.error(f"Processing failed: {e}")
    logger.info("Check debug/ directory for failed request details")
```

---

### Example 11: Direct Backend Access

For advanced control over the processing pipeline:

```python
from ocrrouter import get_backend, Settings
from ocrrouter.utils.io.writers import FileBasedDataWriter
import asyncio

async def custom_processing():
    # Create backend directly
    settings = Settings(openai_api_key="your-key")
    backend = get_backend("mineru", settings=settings)

    # Read PDF
    with open("document.pdf", "rb") as f:
        pdf_bytes = f.read()

    # Create image writer
    image_writer = FileBasedDataWriter("output/images")

    # Direct backend usage
    middle_json, model_output = await backend.analyze(
        pdf_bytes,
        image_writer=image_writer
    )

    print(f"Pages processed: {len(middle_json)}")
    return middle_json

asyncio.run(custom_processing())
```

---

### Example 12: Progressive Processing with Progress Bar

Track processing progress:

```python
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from ocrrouter import DocumentPipeline, Settings

async def process_with_progress():
    settings = Settings(
        backend="deepseek",
        openai_api_key="your-key",
        max_concurrency=5
    )

    pipeline = DocumentPipeline(settings=settings)
    pdf_files = list(Path("input/").glob("*.pdf"))

    # Process with progress bar
    tasks = [
        pipeline.aio_process(pdf, "output/")
        for pdf in pdf_files
    ]

    results = []
    for coro in tqdm.as_completed(tasks, total=len(tasks)):
        result = await coro
        results.append(result)

    return results

asyncio.run(process_with_progress())
```

---

## Use Case Examples

### Academic Research

#### Example 13: Research Paper Processing

Extract formulas, citations, and complex layouts:

```python
from ocrrouter import DocumentPipeline, Settings

settings = Settings(
    backend="mineru",  # Best for academic papers
    openai_api_key="your-key",

    # Enable academic features
    formula_enable=True,         # Extract LaTeX formulas
    table_enable=True,           # Extract tables
    table_merge_enable=True,     # Merge cross-page tables

    # Output configuration
    dump_md=True,
    dump_content_list=True,      # For citation extraction
    draw_layout_bbox=True,       # Visual verification
)

pipeline = DocumentPipeline(settings=settings)

# Process research paper
result = pipeline.process("research_paper.pdf", "output/")

# Extract formulas from content list
formulas = [
    block for block in result["content_list"]
    if block.get("type") == "equation"
]

print(f"Extracted {len(formulas)} formulas")
for i, formula in enumerate(formulas, 1):
    print(f"Formula {i}: {formula.get('latex_text', '')}")
```

---

#### Example 14: Batch Paper Processing for Literature Review

Process multiple papers efficiently:

```python
import asyncio
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings
import json

async def process_papers():
    settings = Settings(
        backend="composite",
        layout_model="mineru",   # Accurate layout
        ocr_model="deepseek",    # Fast OCR
        formula_enable=True,
        table_enable=True,
        max_concurrency=3,       # Don't overload
        openai_api_key="your-key"
    )

    pipeline = DocumentPipeline(settings=settings)

    # Find all papers
    papers = list(Path("papers/").glob("*.pdf"))

    # Process batch
    results = await pipeline.aio_process_batch(
        papers,
        "output/papers",
        session_id="literature-review"
    )

    # Extract metadata
    metadata = []
    for paper, result in zip(papers, results):
        # Extract title (usually first text block)
        content_list = result.get("content_list", [])
        title = next(
            (block["text"] for block in content_list
             if block.get("type") == "text" and block.get("text_level") == 1),
            paper.stem
        )

        metadata.append({
            "filename": paper.name,
            "title": title,
            "formulas": len([b for b in content_list if b.get("type") == "equation"]),
            "tables": len([b for b in content_list if b.get("type") == "table"]),
            "output_dir": result["output_dir"]
        })

    # Save metadata
    with open("output/papers_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Processed {len(papers)} papers")

asyncio.run(process_papers())
```

---

### Business Documents

#### Example 15: Invoice Processing

Extract structured data from invoices:

```python
from ocrrouter import DocumentPipeline, Settings
import json
import re

settings = Settings(
    backend="deepseek",  # Good for business docs
    table_enable=True,
    openai_api_key="your-key"
)

pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("invoice.pdf", "output/")

# Extract tables (line items)
content_list = result["content_list"]
tables = [block for block in content_list if block.get("type") == "table"]

# Extract total amount (simple pattern matching)
markdown = result["markdown"]
total_pattern = r'\$\s*[\d,]+\.\d{2}'
amounts = re.findall(total_pattern, markdown)

invoice_data = {
    "tables": len(tables),
    "line_items": tables[0]["table_body"] if tables else None,
    "amounts_found": amounts,
    "output_dir": result["output_dir"]
}

print(json.dumps(invoice_data, indent=2))
```

---

#### Example 16: Contract Analysis

Process legal documents:

```python
from ocrrouter import DocumentPipeline, Settings

settings = Settings(
    backend="composite",
    layout_model="deepseek",
    ocr_model="generalvlm",
    generalvlm_model_name="gpt-4-vision-preview",  # Premium quality
    openai_api_key="your-key",

    # Process full document
    start_page=0,
    end_page=None,  # All pages
)

pipeline = DocumentPipeline(settings=settings)
result = pipeline.process("contract.pdf", "output/")

# Extract sections
content_list = result["content_list"]
sections = {}
current_section = None

for block in content_list:
    if block.get("type") == "text" and block.get("text_level") in [1, 2]:
        current_section = block["text"]
        sections[current_section] = []
    elif current_section:
        sections[current_section].append(block)

print(f"Contract sections: {list(sections.keys())}")
```

---

### Document Digitization

#### Example 17: Archive Scanning

Batch process large document archives:

```python
import asyncio
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings
from datetime import datetime

async def digitize_archive():
    settings = Settings(
        backend="composite",
        layout_model="dotsocr",   # Fast layout
        ocr_model="paddleocr",    # Fast multilingual OCR

        # Performance optimization
        max_concurrency=20,
        http_timeout=300,
        max_retries=5,

        # Minimal output for storage
        dump_md=True,
        dump_content_list=True,
        dump_middle_json=False,
        dump_model_output=False,
        draw_layout_bbox=False,

        openai_api_key="your-key"
    )

    pipeline = DocumentPipeline(settings=settings)

    # Find all documents
    archive_files = list(Path("archive/").rglob("*.pdf"))

    print(f"Starting digitization of {len(archive_files)} documents")
    start_time = datetime.now()

    # Process in batches of 100
    batch_size = 100
    for i in range(0, len(archive_files), batch_size):
        batch = archive_files[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{len(archive_files)//batch_size + 1}")

        results = await pipeline.aio_process_batch(
            batch,
            "output/archive",
            session_id=f"archive-batch-{batch_num}"
        )

        print(f"  Completed {len(results)} documents")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\nDigitization complete!")
    print(f"  Total documents: {len(archive_files)}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Average: {duration/len(archive_files):.1f}s per document")

asyncio.run(digitize_archive())
```

---

#### Example 18: Multilingual Document Processing

Process documents in multiple languages:

```python
from ocrrouter import DocumentPipeline, Settings

settings = Settings(
    backend="composite",
    layout_model="deepseek",
    ocr_model="paddleocr",  # Excellent multilingual support
    openai_api_key="your-key"
)

pipeline = DocumentPipeline(settings=settings)

# Process multilingual documents
languages = {
    "english": "documents/english.pdf",
    "chinese": "documents/chinese.pdf",
    "japanese": "documents/japanese.pdf",
    "korean": "documents/korean.pdf",
}

for lang, path in languages.items():
    result = pipeline.process(path, f"output/{lang}/")
    print(f"{lang}: {result['output_dir']}")
```

---

### AI/ML Pipelines

#### Example 19: RAG Document Ingestion

Prepare documents for RAG (Retrieval-Augmented Generation):

```python
import asyncio
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings
import json

async def prepare_for_rag():
    settings = Settings(
        backend="deepseek",
        openai_api_key="your-key",

        # Optimize for structured output
        dump_md=True,
        dump_content_list=True,  # Key for chunking
        dump_middle_json=True,   # Structural data
    )

    pipeline = DocumentPipeline(settings=settings)

    # Process documents
    docs = list(Path("knowledge_base/").glob("*.pdf"))
    results = await pipeline.aio_process_batch(
        docs,
        "output/rag_ready",
        session_id="rag-ingestion"
    )

    # Create chunks for RAG
    chunks = []
    for doc, result in zip(docs, results):
        content_list = result["content_list"]

        for i, block in enumerate(content_list):
            if block.get("type") == "text":
                chunks.append({
                    "doc_id": doc.stem,
                    "chunk_id": f"{doc.stem}_{i}",
                    "content": block["text"],
                    "page": block.get("page_idx", 0),
                    "bbox": block.get("bbox"),
                    "metadata": {
                        "text_level": block.get("text_level"),
                        "source": doc.name
                    }
                })

    # Save chunks for vector database ingestion
    with open("output/rag_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"Saved to output/rag_chunks.json")

asyncio.run(prepare_for_rag())
```

---

#### Example 20: Training Data Extraction

Extract structured data for ML training:

```python
import asyncio
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings
import json

async def extract_training_data():
    settings = Settings(
        backend="mineru",
        formula_enable=True,
        table_enable=True,
        dump_content_list=True,
        openai_api_key="your-key"
    )

    pipeline = DocumentPipeline(settings=settings)

    # Process scientific papers
    papers = list(Path("training_corpus/").glob("*.pdf"))
    results = await pipeline.aio_process_batch(
        papers,
        "output/training_data"
    )

    # Extract formula-context pairs for training
    training_examples = []
    for paper, result in zip(papers, results):
        content_list = result["content_list"]

        for i, block in enumerate(content_list):
            if block.get("type") == "equation":
                # Get surrounding context
                context_before = content_list[i-1]["text"] if i > 0 else ""
                context_after = content_list[i+1]["text"] if i < len(content_list)-1 else ""

                training_examples.append({
                    "formula": block.get("latex_text", ""),
                    "context_before": context_before,
                    "context_after": context_after,
                    "source": paper.name,
                    "page": block.get("page_idx", 0)
                })

    # Save training dataset
    with open("output/formula_training_data.json", "w") as f:
        json.dump(training_examples, f, indent=2)

    print(f"Extracted {len(training_examples)} formula examples")

asyncio.run(extract_training_data())
```

---

## Integration Examples

### Example 21: FastAPI Integration

Expose OCRRouter as a REST API:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ocrrouter import DocumentPipeline, Settings
import tempfile
from pathlib import Path
import asyncio

app = FastAPI(title="OCRRouter API")

# Initialize pipeline
settings = Settings(
    backend="deepseek",
    openai_api_key="your-api-key",
    max_concurrency=5
)
pipeline = DocumentPipeline(settings=settings)

@app.post("/convert")
async def convert_document(file: UploadFile = File(...)):
    """Convert uploaded PDF to Markdown"""

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files supported")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process
        result = await pipeline.aio_process(
            tmp_path,
            tempfile.mkdtemp(),
            session_id=f"api-{file.filename}"
        )

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "markdown": result["markdown"],
            "pages": len(result.get("middle_json", [])),
        })

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup
        Path(tmp_path).unlink()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn app:app --reload
```

---

### Example 22: Celery Task Integration

Use OCRRouter with Celery for background processing:

```python
from celery import Celery
from ocrrouter import DocumentPipeline, Settings
import asyncio

app = Celery('ocrrouter_tasks', broker='redis://localhost:6379/0')

settings = Settings(
    backend="deepseek",
    openai_api_key="your-key"
)

@app.task
def convert_document_task(pdf_path: str, output_dir: str):
    """Celery task for document conversion"""

    pipeline = DocumentPipeline(settings=settings)

    # Run async code in sync context
    result = asyncio.run(
        pipeline.aio_process(pdf_path, output_dir)
    )

    return {
        "pdf_path": pdf_path,
        "output_dir": result["output_dir"],
        "status": "completed"
    }

# Usage:
# result = convert_document_task.delay("document.pdf", "output/")
# result.get()  # Wait for completion
```

---

## See Also

- [Backend Guide](BACKENDS.md) — Choose the right backend
- [Configuration](CONFIGURATION.md) — Detailed settings reference
- [API Reference](API.md) — Complete API documentation
- [Output Formats](OUTPUT_FORMATS.md) — Understanding output files
