# Backend Guide

This guide helps you choose the right OCR backend for your documents and understand how to use composite mode to combine model strengths.

## Overview

OCRRouter supports 6 backends, each optimized for different use cases:

1. **MinerU** — Two-step extraction, excellent for academic papers
2. **DeepSeek** — Grounding mode, efficient for general documents
3. **DotsOCR** — Flexible one-step or two-step extraction
4. **PaddleOCR** — Fast OCR-only, strong multilingual support
5. **Hunyuan** — OCR-only, markdown-optimized output
6. **GeneralVLM** — Use GPT-4, Claude, Gemini, or custom VLMs

## Backend Profiles

### 1. MinerU

**Capabilities**: Layout Detection + OCR

**Processing Mode**: Two-step (layout → OCR)

**Strengths**:
- ✓ Excellent layout detection accuracy
- ✓ Superior formula extraction (LaTeX output)
- ✓ Cross-page table merging
- ✓ Handles complex academic documents
- ✓ Specialized handling for equations, tables, text

**Limitations**:
- ✗ Does not support `ocr_only` mode (requires layout detection)
- ✗ Slower than one-step models (two VLM calls per page)
- ✗ In complex layouts (e.g., medical reports), layout detection may miss some words, which then never get passed to OCR

**Best For**:
- Research papers and theses
- Scientific documents with formulas
- Complex multi-column layouts
- Documents requiring precise structure preservation

**Configuration**:
```python
from ocrrouter import Settings

settings = Settings(
    backend="mineru",
    mineru_model_name="mineru-2.5",
    formula_enable=True,
    table_enable=True,
    table_merge_enable=True,  # Merge tables across pages
    openai_api_key="your-key"
)
```

**Output Modes**: `all`, `layout_only`

---

### 2. DeepSeek-OCR 2

**Capabilities**: Layout Detection + OCR

**Processing Mode**: Grounding mode (layout + OCR in one call)

**Strengths**:
- ✓ Efficient grounding mode (single VLM call)
- ✓ Excellent general document accuracy
- ✓ Precise bounding box coordinates
- ✓ Good formula and table support
- ✓ 2-3x faster than two-step models
- ✓ Dynamic resolution with tiling for large images (v2)
- ✓ Visual Causal Flow encoding for improved accuracy (v2)

**Limitations**:
- ✗ May be less specialized for academic content than MinerU

**Best For**:
- General business documents
- Invoices and contracts
- Mixed-content documents
- Performance-critical applications

**Configuration**:
```python
settings = Settings(
    backend="deepseek",
    deepseek_model_name="deepseek-ocr",
    output_mode="all",  # Supports all modes
    openai_api_key="your-key"
)
```

**Output Modes**: `all`, `layout_only`, `ocr_only`

---

### 3. DotsOCR

**Capabilities**: Layout Detection + OCR

**Processing Mode**: Flexible (one-step or two-step)

**Strengths**:
- ✓ Flexible extraction modes
- ✓ Good balance of speed and accuracy
- ✓ Structured JSON output parsing
- ✓ Detailed prompting for format control
- ✓ Strong layout detection and OCR quality

**Limitations**:
- ✗ Less specialized than MinerU for academic content
- ✗ May require tuning for optimal results
- ✗ Self-hosted only (no API available). ~10-20 sec/page on L4 GPU

**Best For**:
- Documents requiring flexible processing
- Experimentation with extraction modes
- Custom prompt-based workflows

**Configuration**:
```python
settings = Settings(
    backend="dotsocr",
    dotsocr_model_name="dots-ocr",
    dotsocr_extraction_mode="one_step",  # or "two_step"
    openai_api_key="your-key"
)
```

**Output Modes**: `all`, `layout_only`, `ocr_only`

---

### 4. PaddleOCR

**Capabilities**: OCR Only (no layout detection)

**Strengths**:
- ✓ Very fast OCR processing
- ✓ Excellent multilingual support
- ✓ Lightweight and efficient
- ✓ Table structure recognition (OTSL → HTML)
- ✓ Chart and formula recognition

**Limitations**:
- ✗ No layout detection (cannot be used as layout model)
- ✗ Requires another model for layout in composite mode
- ✗ **Known issue**: Inputs outside training scope (barcodes, non-text elements) may cause infinite loops. Always use with layout detection to filter out non-text/table/chart elements.

**Best For**:
- Simple OCR tasks
- Multilingual documents
- OCR component in composite mode
- High-throughput processing

**Configuration**:
```python
settings = Settings(
    backend="paddleocr",
    paddleocr_model_name="paddle-ocr",
    output_mode="ocr_only",  # Only supports ocr_only
    openai_api_key="your-key"
)
```

**Output Modes**: `ocr_only` (or `all` when used in composite mode)

---

### 5. Hunyuan-OCR

**Capabilities**: OCR Only (no layout detection)

**Strengths**:
- ✓ Markdown-optimized output
- ✓ Clean text extraction
- ✓ Table parsing to HTML
- ✓ LaTeX formula recognition
- ✓ Content block parsing
- ✓ Output follows document layout and structure

**Limitations**:
- ✗ No layout detection (cannot be used as layout model)
- ✗ Requires another model for layout in composite mode
- ✗ Limited image/figure detection (focuses more on text content in training)

**Best For**:
- Markdown-centric workflows
- Clean text extraction
- OCR component in composite mode

**Configuration**:
```python
settings = Settings(
    backend="hunyuanocr",
    hunyuan_model_name="hunyuan-ocr",
    output_mode="ocr_only",
    openai_api_key="your-key"
)
```

**Output Modes**: `ocr_only` (or `all` when used in composite mode)

---

### 6. GeneralVLM

**Capabilities**: OCR Only (no layout detection)

**Supported Models**:
- GPT-4V, GPT-4O, GPT-4-Turbo
- Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Gemini 1.5 Pro, Gemini 2.0, Gemini 2.5 Pro
- Any OpenAI-compatible VLM API

**Strengths**:
- ✓ Use your preferred VLM
- ✓ Works with proprietary models (GPT, Claude, Gemini)
- ✓ OpenAI-compatible API support
- ✓ Flexible model selection

**Limitations**:
- ✗ No layout detection (cannot be used as layout model)
- ✗ Quality depends on chosen model
- ✗ May be more expensive than specialized OCR models

**Best For**:
- Using existing VLM subscriptions
- Testing different VLM models
- OCR component in composite mode
- Custom or proprietary VLMs

**Configuration**:
```python
# Using GPT-4V
settings = Settings(
    backend="generalvlm",
    generalvlm_model_name="gpt-4-vision-preview",
    openai_api_key="sk-...",
    openai_base_url="https://api.openai.com/v1"
)

# Using Claude via OpenRouter
settings = Settings(
    backend="generalvlm",
    generalvlm_model_name="anthropic/claude-3.5-sonnet",
    openai_api_key="sk-or-...",
    openai_base_url="https://openrouter.ai/api/v1"
)

# Using Gemini via OpenAI-compatible proxy
settings = Settings(
    backend="generalvlm",
    generalvlm_model_name="gemini-2.5-pro",
    openai_api_key="your-gemini-key",
    openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

**Output Modes**: `ocr_only` (or `all` when used in composite mode)

---

## Composite Mode

Composite mode allows you to **mix layout detection from one model with OCR from another**, combining their strengths.

### How It Works

1. **Layout Detection Phase**: Use `layout_model` to detect document structure (blocks, bounding boxes, types)
2. **Content Extraction Phase**: Use `ocr_model` to extract text from each detected region
3. **Result Assembly**: Combine layout + content into final output

### Valid Combinations

**Layout Models** (must support layout detection):
- `mineru`
- `deepseek`
- `dotsocr`

**OCR Models** (any model):
- `mineru`
- `deepseek`
- `dotsocr`
- `paddleocr`
- `hunyuanocr`
- `generalvlm`

### Why Use Composite Mode?

1. **Optimize Cost vs Quality**: Use expensive model for layout, cheaper for OCR
2. **Leverage Strengths**: MinerU's excellent layout + PaddleOCR's speed
3. **Flexibility**: Mix open-source and proprietary models
4. **Performance**: Often 2-3x faster than using one model for everything

### Common Patterns

#### Pattern 1: Quality Layout + Fast OCR
```python
settings = Settings(
    backend="composite",
    layout_model="mineru",      # Best layout detection
    ocr_model="paddleocr",      # Fast OCR extraction
)
```
**Use case**: Academic papers where structure matters but speed is important

#### Pattern 2: Layout Detection + Premium VLM
```python
settings = Settings(
    backend="composite",
    layout_model="deepseek",    # Efficient layout
    ocr_model="generalvlm",     # Use GPT-4 for OCR
    generalvlm_model_name="gpt-4-vision-preview"
)
```
**Use case**: High-quality OCR using your GPT-4 subscription

#### Pattern 3: Fast Layout + Markdown-Optimized OCR
```python
settings = Settings(
    backend="composite",
    layout_model="dotsocr",     # Fast layout detection
    ocr_model="hunyuanocr",     # Markdown-focused OCR
)
```
**Use case**: General documents with markdown output priority

#### Pattern 4: Cost Optimization
```python
settings = Settings(
    backend="composite",
    layout_model="deepseek",    # Efficient one-step for layout
    ocr_model="paddleocr",      # Lightweight OCR
    max_concurrency=20          # High parallelism
)
```
**Use case**: Batch processing large document archives

### Example: Complete Composite Configuration

```python
from ocrrouter import DocumentPipeline, Settings

settings = Settings(
    # Composite mode
    backend="composite",
    layout_model="mineru",
    ocr_model="deepseek",

    # API configuration
    openai_api_key="your-key",
    openai_base_url="https://api.example.com/v1",

    # Processing options
    formula_enable=True,
    table_enable=True,

    # Output configuration
    output_mode="all",
    dump_md=True,
    dump_content_list=True,

    # Performance tuning
    max_concurrency=10,
    http_timeout=120
)

pipeline = DocumentPipeline(settings=settings)
result = await pipeline.aio_process("document.pdf", "output/")
```

---

## Backend Comparison Matrix

| Feature | MinerU | DeepSeek | DotsOCR | PaddleOCR | Hunyuan | GeneralVLM |
|---------|:------:|:--------:|:-------:|:---------:|:-------:|:----------:|
| **Layout Detection** | ✓ | ✓ | ✓ | — | — | — |
| **OCR Extraction** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Formulas** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | — | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Tables** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Processing Mode** | Two-step | One-step | Both | N/A | N/A | N/A |
| **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡ |
| **Multilingual** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Output Modes** | all, layout | all | all | ocr | ocr | ocr |
| **Best For** | Academic | General | Quality | Fast/Multilingual | Markdown | Custom VLM |

**Speed reference** (per page):
- MinerU, PaddleOCR: ~1 sec
- DeepSeek, HunyuanOCR: ~1-3 sec
- DotsOCR: ~10-20 sec (self-hosted on L4 GPU)

---

## Quality Rankings (Experimental)

> **Disclaimer**: These rankings are based on personal experiments and may vary depending on document types and configurations. Results should be verified for your specific use case.

### Layout Detection Quality

| Rank | Backend | Notes |
|:----:|---------|-------|
| 1 | DotsOCR | Strong layout detection accuracy |
| 2 | MinerU | Excellent for academic/structured documents |
| 3 | DeepSeek | Good general-purpose layout detection |

### OCR Quality (Full-page OCR supported)

| Rank | Backend | Notes |
|:----:|---------|-------|
| 1 | DotsOCR | High OCR accuracy with structure output |
| 2 | HunyuanOCR | Good text extraction, follows document layout |
| 3 | DeepSeek | Balanced speed and accuracy |

### OCR Quality (No full-page OCR support)

These backends require layout detection first and cannot perform standalone full-page OCR:

| Rank | Backend | Notes |
|:----:|---------|-------|
| 1 | PaddleOCR | Excellent OCR within detected regions |
| 2 | MinerU | High quality but requires layout step |

---

## Decision Tree

Use this flowchart to choose the right backend:

```
START: What type of document?
│
├─ Academic/Scientific papers with formulas
│  └─ Choose: MinerU
│     └─ Need faster? → Composite: mineru (layout) + paddleocr (OCR)
│
├─ General business documents (invoices, contracts)
│  └─ Choose: DeepSeek
│     └─ Need premium quality? → Composite: deepseek (layout) + generalvlm (OCR)
│
├─ Multilingual documents
│  ├─ Need layout detection? → Composite: deepseek (layout) + paddleocr (OCR)
│  └─ OCR only? → PaddleOCR
│
├─ Large batch processing (speed critical)
│  └─ Composite: dotsocr (layout) + paddleocr (OCR)
│
├─ Want to use GPT-4/Claude/Gemini
│  ├─ Need layout? → Composite: deepseek (layout) + generalvlm (OCR)
│  └─ OCR only? → GeneralVLM
│
└─ Markdown-focused output
   ├─ Need layout? → Composite: deepseek (layout) + hunyuanocr (OCR)
   └─ OCR only? → Hunyuan
```

---

## Backend Selection by Use Case

### Academic Research
```python
# Best: MinerU for accuracy
Settings(backend="mineru", formula_enable=True, table_merge_enable=True)

# Faster alternative: Composite mode
Settings(backend="composite", layout_model="mineru", ocr_model="deepseek")
```

### Business Documents
```python
# Best: DeepSeek for efficiency
Settings(backend="deepseek", table_enable=True)

# With GPT-4 for premium quality
Settings(backend="composite", layout_model="deepseek", ocr_model="generalvlm",
         generalvlm_model_name="gpt-4-vision-preview")
```

### Document Digitization
```python
# Best: Composite for speed
Settings(backend="composite", layout_model="dotsocr", ocr_model="paddleocr",
         max_concurrency=20)
```

### AI/ML Pipelines (RAG, Training Data)
```python
# Best: DeepSeek for structured output
Settings(backend="deepseek", dump_content_list=True, dump_middle_json=True)

# With custom VLM
Settings(backend="composite", layout_model="deepseek", ocr_model="generalvlm")
```

---

## See Also

- [Examples](EXAMPLES.md) — Code examples for each backend
- [Configuration](CONFIGURATION.md) — Detailed settings reference
- [API Reference](API.md) — Complete API documentation
- [Output Formats](OUTPUT_FORMATS.md) — Understanding output files
