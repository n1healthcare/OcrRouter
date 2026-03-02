"""PP-DocLayoutV3 client for document layout detection.

Uses the PP-DocLayoutV3 model from PaddlePaddle via HuggingFace Transformers
for detecting document regions (text, tables, images, formulas, etc.).
"""

import asyncio
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    import torch

from ocrrouter.backends.utils import ContentBlock, gather_tasks
from ocrrouter.config import Settings
from ocrrouter.observability import get_langfuse_client

from .postprocessor import PPDocLayoutPostprocessor
from .preprocessor import PPDocLayoutPreprocessor
from .utils import DEFAULT_ID2LABEL, DEFAULT_THRESHOLD


class PPDocLayoutClient:
    """PP-DocLayoutV3 client for layout detection.

    This client uses the PP-DocLayoutV3 model from PaddlePaddle (via HuggingFace
    Transformers) to detect document layout regions including:
    - Text blocks (paragraphs, titles, references)
    - Tables
    - Images and charts
    - Formulas (display and inline)
    - Headers, footers, and other structural elements

    The model runs locally on GPU (if available) or CPU.

    Example:
        >>> client = PPDocLayoutClient(settings)
        >>> blocks = await client.aio_layout_detect(image)
        >>> # blocks contains ContentBlock objects with type and bbox
    """

    _instances: dict[str, "PPDocLayoutClient"] = {}

    def __init__(
        self,
        settings: Settings,
        model_dir: str | None = None,
        threshold: float | None = None,
        batch_size: int = 1,
        device: str | None = None,
        executor: Executor | None = None,
        use_tqdm: bool | None = None,
    ) -> None:
        """Initialize the PP-DocLayoutV3 client.

        Args:
            settings: Settings object with configuration.
            model_dir: HuggingFace model ID or local path. Defaults to
                "PaddlePaddle/PP-DocLayoutV3_safetensors".
            threshold: Detection confidence threshold. Defaults to 0.3.
            batch_size: Batch size for inference.
            device: Device to run on ("cuda", "cpu", or specific like "cuda:0").
            executor: Executor for running inference in thread pool.
            use_tqdm: Whether to show progress bars.
        """
        self._settings = settings

        # Use settings.use_tqdm if not explicitly provided
        if use_tqdm is None:
            use_tqdm = settings.use_tqdm

        # Model configuration
        self.model_dir = model_dir or settings.ppdoclayout_model_dir
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self.batch_size = batch_size

        # Device selection (lazy import torch to avoid requiring it at module load)
        if device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        # Initialize preprocessor and postprocessor
        self.preprocessor = PPDocLayoutPreprocessor()
        self.postprocessor = PPDocLayoutPostprocessor(
            id2label=DEFAULT_ID2LABEL,
            layout_nms=True,
            layout_merge_mode="large",
            debug=settings.debug,
        )

        # Executor for thread pool inference
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.max_concurrency = settings.max_concurrency
        self.use_tqdm = use_tqdm
        self.debug = settings.debug

        # Lazy-loaded model and processor
        self._model = None
        self._image_processor = None
        self._model_lock = asyncio.Lock()

    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded (lazy initialization)."""
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:
                return

            logger.debug(f"Loading PP-DocLayoutV3 from {self.model_dir}")

            # Load in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            logger.debug(f"PP-DocLayoutV3 loaded on device: {self.device}")

    def _load_model(self) -> None:
        """Load the PP-DocLayoutV3 model (runs in thread pool).

        Tries multiple loading strategies:
        1. PP-DocLayoutV3 specific classes from transformers
        2. Dynamic import from model's remote code
        3. RT-DETR fallback for similar architecture
        """
        model_loaded = False
        last_error = None

        # Strategy 1: Try PP-DocLayoutV3 specific classes from transformers
        try:
            from transformers import (
                PPDocLayoutV3ForObjectDetection,
                PPDocLayoutV3ImageProcessorFast,
            )

            self._image_processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(self.model_dir)
            self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
            model_loaded = True
            logger.debug("Loaded PP-DocLayoutV3 using native transformers classes")
        except ImportError as e:
            last_error = e
            logger.debug(f"PP-DocLayoutV3 classes not in transformers: {e}")

        # Strategy 2: Try loading via huggingface_hub dynamic import
        if not model_loaded:
            try:
                import importlib.util
                import sys

                from huggingface_hub import hf_hub_download

                # Download and import the model's custom code
                model_file = hf_hub_download(
                    repo_id=self.model_dir,
                    filename="modeling_ppdoclayoutv3.py",
                )
                config_file = hf_hub_download(
                    repo_id=self.model_dir,
                    filename="configuration_ppdoclayoutv3.py",
                )
                processor_file = hf_hub_download(
                    repo_id=self.model_dir,
                    filename="image_processing_ppdoclayoutv3.py",
                )

                # Load the modules dynamically
                def load_module(name, path):
                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    return module

                config_module = load_module("configuration_ppdoclayoutv3", config_file)
                model_module = load_module("modeling_ppdoclayoutv3", model_file)
                processor_module = load_module("image_processing_ppdoclayoutv3", processor_file)

                # Get the classes
                ModelClass = model_module.PPDocLayoutV3ForObjectDetection
                ProcessorClass = processor_module.PPDocLayoutV3ImageProcessorFast

                self._image_processor = ProcessorClass.from_pretrained(self.model_dir)
                self._model = ModelClass.from_pretrained(self.model_dir)
                model_loaded = True
                logger.debug("Loaded PP-DocLayoutV3 using dynamic import from HuggingFace Hub")
            except Exception as e:
                last_error = e
                logger.debug(f"Dynamic import failed: {e}")

        # Strategy 3: Try using RT-DETR as base (PP-DocLayoutV3 uses similar architecture)
        if not model_loaded:
            try:
                from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

                self._image_processor = RTDetrImageProcessor.from_pretrained(
                    self.model_dir,
                    trust_remote_code=True,
                )
                self._model = RTDetrForObjectDetection.from_pretrained(
                    self.model_dir,
                    trust_remote_code=True,
                )
                model_loaded = True
                logger.debug("Loaded PP-DocLayoutV3 using RT-DETR classes")
            except Exception as e:
                last_error = e
                logger.debug(f"RT-DETR fallback failed: {e}")

        # Final error if nothing worked
        if not model_loaded:
            raise ImportError(
                f"Failed to load PP-DocLayoutV3 model from '{self.model_dir}'.\n"
                f"Last error: {last_error}\n\n"
                "The 'pp_doclayout_v3' model type is not yet in your transformers version.\n\n"
                "Solutions:\n"
                "1. Try installing transformers from GitHub main branch:\n"
                "   pip install git+https://github.com/huggingface/transformers.git\n\n"
                "2. Use a different layout model instead:\n"
                "   - layout_model='mineru' (MinerU VLM layout detection)\n"
                "   - layout_model='deepseek' (DeepSeek-OCR grounding mode)\n"
                "   - layout_model='dotsocr' (DotsOCR layout detection)"
            )

        self._model.eval()
        self._model = self._model.to(self.device)

        # Update id2label from model config if available
        if hasattr(self._model.config, "id2label"):
            self.postprocessor.id2label = self._model.config.id2label

    def _run_inference(
        self,
        images: list[Image.Image],
    ) -> list[dict]:
        """Run inference on a batch of images (runs in thread pool).

        Args:
            images: List of PIL Images.

        Returns:
            List of detection result dicts.
        """
        if self._model is None or self._image_processor is None:
            raise RuntimeError("Model not loaded. Call _ensure_model_loaded() first.")

        import torch

        # Preprocess images
        pil_images = [self.preprocessor.prepare_for_layout(img) for img in images]
        inputs = self._image_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        target_sizes = torch.tensor(
            [img.size[::-1] for img in pil_images],  # (height, width)
            device=self.device,
        )

        # Try post_process_object_detection, fall back to manual processing
        if hasattr(self._image_processor, "post_process_object_detection"):
            results = self._image_processor.post_process_object_detection(
                outputs,
                threshold=self.threshold,
                target_sizes=target_sizes,
            )
        else:
            # Manual post-processing for models without the method
            results = self._manual_post_process(outputs, target_sizes)

        return results

    def _manual_post_process(
        self,
        outputs,
        target_sizes: "torch.Tensor",
    ) -> list[dict]:
        """Manual post-processing fallback.

        Args:
            outputs: Model outputs with logits and pred_boxes.
            target_sizes: Tensor of (height, width) for each image.

        Returns:
            List of detection result dicts.
        """
        import torch
        from torch.nn.functional import softmax

        results = []
        logits = outputs.logits  # (batch, num_queries, num_classes)
        boxes = outputs.pred_boxes  # (batch, num_queries, 4) in cxcywh format

        for idx in range(len(target_sizes)):
            height, width = target_sizes[idx].tolist()

            # Get scores and labels
            probs = softmax(logits[idx], dim=-1)
            scores, labels = probs.max(dim=-1)

            # Filter by threshold (exclude background class if present)
            mask = scores > self.threshold
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            filtered_boxes = boxes[idx][mask]

            # Convert cxcywh to xyxy and scale to image size
            cx, cy, w, h = filtered_boxes.unbind(-1)
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            scaled_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            results.append(
                {
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                    "boxes": scaled_boxes,
                }
            )

        return results

    async def _do_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """Core layout detection logic.

        Args:
            image: PIL Image of document page.
            priority: Optional priority (unused, for interface compatibility).
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock objects with type and bbox (no content).
        """
        await self._ensure_model_loaded()

        # Run inference in thread pool
        loop = asyncio.get_event_loop()

        if semaphore is None:
            results = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                [image],
            )
        else:
            async with semaphore:
                results = await loop.run_in_executor(
                    self.executor,
                    self._run_inference,
                    [image],
                )

        if not results:
            return []

        # Parse detection output
        img_size = image.size  # (width, height)
        blocks = self.postprocessor.parse_layout_output(
            results[0],
            img_size=img_size,
        )

        return blocks

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        page_idx: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout blocks in a document page image.

        Args:
            image: PIL Image of document page.
            priority: Optional priority for request ordering.
            semaphore: Optional semaphore for concurrency control.
            page_idx: Optional page index for tracing.

        Returns:
            List of ContentBlock objects with type and bbox.
        """
        langfuse = get_langfuse_client()

        if langfuse and page_idx is not None:
            with langfuse.start_as_current_span(name=f"page-{page_idx}"):
                with langfuse.start_as_current_span(name="layout-ppdoclayout-detection"):
                    return await self._do_layout_detect(image, priority, semaphore)
        else:
            return await self._do_layout_detect(image, priority, semaphore)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images.

        Args:
            images: List of PIL Images.
            priority: Priority value(s) for request ordering.
            semaphore: Optional semaphore for concurrency control.

        Returns:
            List of ContentBlock lists, one per image.
        """
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)

        total_pages = len(images)
        return await gather_tasks(
            tasks=[
                self.aio_layout_detect(img, p, semaphore, page_idx=idx if total_pages > 1 else None)
                for idx, (img, p) in enumerate(zip(images, priority))
            ],
            use_tqdm=self.use_tqdm,
            tqdm_desc="PP-DocLayout Detection",
        )

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        """Content extraction (not supported - layout only).

        Raises:
            NotImplementedError: PP-DocLayoutV3 is layout detection only.
        """
        raise NotImplementedError(
            "PP-DocLayoutV3 is a layout detection model only. Use a different backend for OCR extraction."
        )
