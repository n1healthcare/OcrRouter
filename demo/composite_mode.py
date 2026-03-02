"""
OCRRouter Composite Mode Demonstration

This script demonstrates OCRRouter's unique composite mode feature,
which allows you to mix layout detection from one model with OCR
extraction from another model.

This enables you to:
- Combine model strengths (e.g., MinerU's excellent layout + PaddleOCR's speed)
- Optimize cost vs quality
- Mix open-source and proprietary models

Prerequisites:
1. Install ocrrouter: pip install ocrrouter
2. Set environment variables with your API credentials
3. Place a PDF file in the same directory or update the path

Run:
    python composite_mode.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from ocrrouter import DocumentPipeline, Settings


async def compare_composite_configurations(pdf_path: str, output_base: str):
    """
    Compare different composite mode configurations.

    This function processes the same document with multiple composite
    configurations to demonstrate the flexibility of composite mode.
    """

    # API credentials
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        return

    # ========== Configuration 1: Quality Layout + Fast OCR ==========

    print("\n" + "=" * 70)
    print("Configuration 1: MinerU Layout + PaddleOCR")
    print("=" * 70)
    print("Use case: Academic papers where structure matters but speed is important")
    print()

    config1 = Settings(
        backend="composite",
        layout_model="mineru",  # Best layout detection
        ocr_model="paddleocr",  # Fast OCR
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        formula_enable=True,
        table_enable=True,
    )

    # ========== Configuration 2: Fast Layout + Quality OCR ==========

    print("\n" + "=" * 70)
    print("Configuration 2: DeepSeek Layout + DeepSeek OCR")
    print("=" * 70)
    print("Use case: General documents, good balance of speed and accuracy")
    print()

    config2 = Settings(
        backend="composite",
        layout_model="deepseek",  # Fast layout
        ocr_model="deepseek",  # Fast OCR
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    # ========== Configuration 3: Layout + Premium VLM OCR ==========

    print("\n" + "=" * 70)
    print("Configuration 3: DotsOCR Layout + GeneralVLM (GPT/Claude/Gemini)")
    print("=" * 70)
    print("Use case: High-quality OCR using premium VLM subscription")
    print()

    config3 = Settings(
        backend="composite",
        layout_model="dotsocr",  # Fast layout
        ocr_model="generalvlm",  # Use premium VLM
        generalvlm_model_name="gemini-2.5-pro",  # Change to your model
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    # ========== Process with all configurations ==========

    configurations = [
        ("mineru_paddleocr", config1),
        ("deepseek_deepseek", config2),
        ("dotsocr_generalvlm", config3),
    ]

    results = []

    print("\n" + "=" * 70)
    print("Processing document with all configurations...")
    print("=" * 70)

    for name, config in configurations:
        print(f"\nProcessing with {name}...")
        start_time = datetime.now()

        try:
            pipeline = DocumentPipeline(settings=config)
            result = await pipeline.aio_process(
                pdf_path,
                f"{output_base}/{name}",
            )

            duration = (datetime.now() - start_time).total_seconds()

            results.append(
                {
                    "name": name,
                    "config": config,
                    "result": result,
                    "duration": duration,
                    "success": True,
                }
            )

            print(f"  ✓ Completed in {duration:.1f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append(
                {
                    "name": name,
                    "config": config,
                    "result": None,
                    "duration": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # ========== Compare Results ==========

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for r in results:
        if r["success"]:
            print(f"\n{r['name']}:")
            print(f"  Duration: {r['duration']:.1f}s")
            print(f"  Output: {r['result']['output_dir']}")
            print(f"  Pages: {len(r['result'].get('middle_json', []))}")
            print(f"  Blocks: {len(r['result'].get('content_list', []))}")

            # Show markdown preview
            markdown = r["result"]["markdown"]
            print("  Markdown preview (first 200 chars):")
            print(f"    {markdown[:200].replace(chr(10), ' ')}...")
        else:
            print(f"\n{r['name']}: FAILED - {r.get('error', 'Unknown error')}")

    # ========== Recommendations ==========

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
Composite Mode Use Cases:

1. Academic Papers (formulas, complex layouts):
   - layout_model="mineru" (best layout detection)
   - ocr_model="deepseek" (fast, accurate)

2. Business Documents (invoices, contracts):
   - layout_model="deepseek" (fast grounding mode)
   - ocr_model="paddleocr" (fast, multilingual)

3. High-Quality Output (premium results):
   - layout_model="deepseek" (efficient)
   - ocr_model="generalvlm" (GPT-4/Claude/Gemini)

4. Batch Processing (speed critical):
   - layout_model="dotsocr" (fast)
   - ocr_model="paddleocr" (very fast, multilingual)

5. Cost Optimization:
   - Use cheaper models for layout (one call per page)
   - Use specialized models for OCR (many calls per page)
    """)

    print("=" * 70)


def simple_composite_example():
    """
    Simple synchronous example of composite mode.
    """

    print("\n" + "=" * 70)
    print("SIMPLE COMPOSITE MODE EXAMPLE")
    print("=" * 70)

    # API credentials
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        return

    # Document path
    pdf_path = "document.pdf"  # Change to your PDF

    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        return

    # Create composite pipeline
    settings = Settings(
        backend="composite",
        layout_model="mineru",  # Best layout detection
        ocr_model="paddleocr",  # Fast OCR
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
    )

    pipeline = DocumentPipeline(settings=settings)

    print(f"\nProcessing {pdf_path} with composite mode...")
    print(f"  Layout model: {settings.layout_model}")
    print(f"  OCR model: {settings.ocr_model}")

    try:
        result = pipeline.process(pdf_path, "output/composite_simple")

        print("\n✓ Success!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Pages: {len(result.get('middle_json', []))}")

    except Exception as e:
        print(f"\n✗ Failed: {e}")


async def main_async():
    """Main async function for comparisons."""

    # Document path
    pdf_path = "document.pdf"  # Change to your PDF

    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        print("Please update the 'pdf_path' variable with a valid PDF path.")
        return

    # Run comparison
    await compare_composite_configurations(pdf_path, "output/composite_comparison")


def main():
    """Main entry point."""

    print("=" * 70)
    print("OCRRouter Composite Mode Demonstration")
    print("=" * 70)

    print("""
This demo showcases OCRRouter's unique composite mode feature.

Composite mode allows you to mix layout detection from one model
with OCR extraction from another model, combining their strengths.

Choose demo mode:
  1. Simple example (single configuration)
  2. Comparison (multiple configurations in parallel)
    """)

    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        simple_composite_example()
    elif choice == "2":
        asyncio.run(main_async())
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
