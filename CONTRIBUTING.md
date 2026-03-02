# Contributing to OCRRouter

Thank you for your interest in contributing to OCRRouter! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Adding New Backends](#adding-new-backends)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports:
1. Check the [existing issues](https://github.com/yourusername/ocrrouter/issues)
2. Review the [documentation](docs/)
3. Try the latest version

When filing a bug report, include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- OCRRouter version, Python version, OS
- Sample PDF (if possible) or description of the document
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear use case and motivation
- Expected behavior and interface
- Examples of how it would work
- Any alternatives you've considered

### Pull Requests

- Fix bugs
- Add new backends
- Improve documentation
- Add examples
- Enhance performance
- Improve error handling

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ocrrouter.git
cd ocrrouter
```

### 2. Install Python 3.13

This project requires **Python 3.13** or later.

```bash
# Using uv (recommended)
uv python install 3.13

# Or download from python.org
# https://www.python.org/downloads/
```

### 3. Install Development Dependencies

We use **uv** for fast, reliable dependency management:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv sync --all-extras

# This will:
# - Install Python 3.13 if needed
# - Create a virtual environment
# - Install all dependencies including dev tools
```

**Alternative using pip:**
```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 5. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API credentials
nano .env
```

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation updates
- `refactor/` — Code refactoring
- `test/` — Test additions or fixes

### 2. Make Your Changes

- Write clear, concise code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Using uv (recommended)
uv run pytest                      # Run tests
uv run ruff check .                # Run linter
uv run ruff format --check .       # Check formatting
uv run mypy ocrrouter              # Run type checker

# Or with activated virtual environment
source .venv/bin/activate
pytest                             # Run tests
ruff check .                       # Run linter
ruff format .                      # Format code (auto-fix)
mypy ocrrouter                     # Run type checker
```

**Linting Configuration:**

The project uses Ruff for linting and formatting. Configuration is in `pyproject.toml`:
- Line length: 120 characters
- Target: Python 3.13
- Auto-fixes available for many issues

**Common linting commands:**
```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check formatting without changes
uv run ruff format --check .
```

### 4. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for new backend"
```

Commit message format:
```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation changes
- `refactor` — Code refactoring
- `test` — Test changes
- `chore` — Build/tooling changes

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

In your PR description, include:
- What the change does
- Why it's needed
- How to test it
- Related issue numbers (if any)

### 6. Code Review

- Address review feedback promptly
- Be open to suggestions
- Update your PR based on feedback

---

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Use type hints where possible

### Code Organization

```python
# Good: Imports organized by standard lib, third-party, local
import os
import sys
from typing import Any

from loguru import logger
from pydantic import BaseModel

from ocrrouter.config import Settings
from ocrrouter.backends.base import BaseModelBackend


# Good: Type hints
def process_document(
    input_path: str,
    output_dir: str,
    settings: Settings | None = None,
) -> dict:
    """Process a document.

    Args:
        input_path: Path to input file
        output_dir: Output directory
        settings: Optional settings

    Returns:
        Processing results
    """
    ...


# Good: Docstrings
class DocumentPipeline:
    """Main pipeline for document processing.

    This class orchestrates the entire workflow...

    Example:
        >>> pipeline = DocumentPipeline(backend="deepseek")
        >>> result = pipeline.process("doc.pdf", "output/")
    """
```

### Error Handling

```python
# Good: Specific exceptions with helpful messages
try:
    result = backend.analyze(pdf_bytes)
except ValueError as e:
    logger.error(f"Invalid PDF format: {e}")
    raise
except ConnectionError as e:
    logger.error(f"Failed to connect to VLM server: {e}")
    raise

# Bad: Bare except
try:
    result = backend.analyze(pdf_bytes)
except:  # Don't do this
    pass
```

### Logging

```python
from loguru import logger

# Use appropriate log levels
logger.debug("Preprocessing page 5")
logger.info("Processing document.pdf")
logger.warning("API rate limit approaching")
logger.error("Failed to process page 3")
```

---

## Adding New Backends

To add support for a new OCR model:

### 1. Create Backend Directory

```
ocrrouter/backends/models/newbackend/
├── __init__.py
├── backend.py          # Main backend class
├── client.py           # VLM client
├── preprocessor.py     # Image preprocessing
└── postprocessor.py    # Output parsing
```

### 2. Implement Backend Class

```python
# backend.py
from ocrrouter.backends.base import BaseModelBackend

class NewBackend(BaseModelBackend):
    """Backend for NewModel OCR."""

    async def analyze(
        self,
        pdf_bytes: bytes,
        image_writer: Any,
        **options: Any,
    ) -> tuple[list, list]:
        """Analyze document with NewModel.

        Args:
            pdf_bytes: PDF file bytes
            image_writer: Image writer instance
            **options: Additional options

        Returns:
            Tuple of (middle_json, model_output)
        """
        # Implementation
        ...
```

### 3. Register in Factory

```python
# backends/factory.py
from .models.newbackend import NewBackend

def get_backend(backend_name: str, settings: Settings):
    if backend_name == "newbackend":
        return NewBackend(settings)
    # ...
```

### 4. Add Configuration

```python
# config/settings.py
class Settings(BaseModel):
    backend: Literal[
        ...,
        "newbackend",  # Add here
    ] = "mineru"

    newbackend_model_name: str = Field(
        default="newbackend-v1",
        description="NewBackend model name"
    )
```

### 5. Add Documentation

- Update `docs/BACKENDS.md` with backend profile
- Add examples to `docs/EXAMPLES.md`
- Update README.md backend comparison table

### 6. Add Tests

```python
# tests/test_newbackend.py
import pytest
from ocrrouter import get_backend, Settings

def test_newbackend_creation():
    settings = Settings(backend="newbackend")
    backend = get_backend("newbackend", settings)
    assert backend is not None

def test_newbackend_analyze():
    # Test backend functionality
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=ocrrouter --cov-report=html

# Run specific test
pytest tests/test_pipeline.py::test_process_document
```

### Writing Tests

```python
import pytest
from pathlib import Path
from ocrrouter import DocumentPipeline, Settings

@pytest.fixture
def sample_pdf():
    """Provide sample PDF for testing."""
    return Path("tests/fixtures/sample.pdf")

def test_process_document(sample_pdf):
    """Test basic document processing."""
    settings = Settings(backend="deepseek", openai_api_key="test")
    pipeline = DocumentPipeline(settings=settings)

    result = pipeline.process(sample_pdf, "output/test")

    assert result["markdown"] is not None
    assert len(result["middle_json"]) > 0
```

---

## Documentation

### Documentation Structure

```
docs/
├── BACKENDS.md       # Backend comparison and selection
├── EXAMPLES.md       # Code examples
├── API.md            # API reference
├── CONFIGURATION.md  # Settings reference
└── OUTPUT_FORMATS.md # Output file formats
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Show expected outputs
- Link related sections
- Use proper markdown formatting

### Example Template

````markdown
## New Feature

Description of the feature.

### Usage

```python
from ocrrouter import DocumentPipeline

# Example usage
pipeline = DocumentPipeline(new_feature=True)
result = pipeline.process("document.pdf", "output/")
```

### Parameters

- `new_feature` (bool): Enable new feature

### Returns

- `dict`: Processing results

### Example

```python
# Complete example
...
```
````

---

## Release Process

(For maintainers)

### 1. Update Version

```python
# ocrrouter/version.py
__version__ = "0.2.0"
```

### 2. Update Changelog

```markdown
# CHANGELOG.md

## [0.2.0] - 2024-01-15

### Added
- New backend support for XYZ
- Feature ABC

### Changed
- Improved performance of...

### Fixed
- Bug in...
```

### 3. Create Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Questions?

- **Issues**: [GitHub Issues](https://github.com/yourusername/ocrrouter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ocrrouter/discussions)
- **Documentation**: [docs/](docs/)

---

Thank you for contributing to OCRRouter!
