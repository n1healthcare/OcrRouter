# OcrRouter

Python library for converting PDFs and images to Markdown using multiple VLM backends. Used in the CHR parsing pipeline for document OCR.

## Backends

MinerU, DeepSeek-OCR, DotsOCR, PaddleOCR, Hunyuan-OCR, GeneralVLM (GPT/Claude/Gemini). Composite mode mixes layout detection + OCR from different models.

## Dev Commands

```bash
uv run pytest              # Run tests
uv run python -m ocrrouter # Run locally
uv build                   # Build package
```

## Key Entry Points

- `ocrrouter/` — main library source
- `docs/` — documentation
- `demo/` — usage examples

## Rules

- Target branch for PRs: `develop`
- Never push to `main` directly
