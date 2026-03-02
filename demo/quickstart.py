"""
OCRRouter Quickstart Example

This is the simplest possible example to get started with OCRRouter.
It demonstrates basic document processing with the DeepSeek backend.

Prerequisites:
1. Install ocrrouter: pip install ocrrouter
2. Set environment variables or update the code below with your API credentials
3. Place a PDF file in the same directory or update the path

Run:
    python quickstart.py
"""

import os
from pathlib import Path

from ocrrouter import DocumentPipeline, Settings


def main():
    # ========== Configuration ==========

    # Option 1: Load from environment variables (recommended for production)
    # Set these in your .env file or shell:
    #   export OPENAI_BASE_URL="https://api.example.com/v1"
    #   export OPENAI_API_KEY="your-api-key"

    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Option 2: Hardcode for testing (NOT recommended for production)
    # openai_base_url = "https://api.example.com/v1"
    # openai_api_key = "your-api-key"

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please set environment variable or update the script.")
        return

    # ========== Document Setup ==========

    # Path to your PDF file
    input_pdf = "document.pdf"  # Change this to your PDF path

    # Check if file exists
    if not Path(input_pdf).exists():
        print(f"ERROR: File not found: {input_pdf}")
        print("Please update the 'input_pdf' variable with a valid PDF path.")
        return

    # Output directory
    output_dir = "output"

    # ========== Create Pipeline ==========

    print("Creating OCRRouter pipeline...")

    settings = Settings(
        # Backend selection (deepseek is fast and accurate for general documents)
        backend="deepseek",

        # API configuration
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,

        # Output mode: "all" = layout detection + OCR
        output_mode="all",

        # Logging level
        log_level="INFO",
    )

    pipeline = DocumentPipeline(settings=settings)

    # ========== Process Document ==========

    print(f"\nProcessing document: {input_pdf}")
    print("This may take a minute depending on document size...")

    try:
        result = pipeline.process(input_pdf, output_dir)

        # ========== Results ==========

        print("\n" + "=" * 50)
        print("SUCCESS! Document processed.")
        print("=" * 50)

        print(f"\nOutput directory: {result['output_dir']}")

        print("\nGenerated files:")
        output_path = Path(result['output_dir'])
        for file in sorted(output_path.glob("*")):
            print(f"  - {file.name}")

        print("\nMarkdown preview (first 500 characters):")
        print("-" * 50)
        markdown = result["markdown"]
        print(markdown[:500])
        if len(markdown) > 500:
            print("...")

        print(f"\n\nPages processed: {len(result.get('middle_json', []))}")
        print(f"Content blocks: {len(result.get('content_list', []))}")

        print("\nTo view the full markdown:")
        print(f"  cat {output_path / Path(input_pdf).stem}.md")

    except Exception as e:
        print("\nERROR: Processing failed!")
        print(f"  {type(e).__name__}: {e}")
        print("\nPlease check:")
        print("  - API credentials are correct")
        print("  - VLM server is accessible")
        print("  - PDF file is not corrupted")
        return

    print("\n" + "=" * 50)
    print("Quickstart complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
