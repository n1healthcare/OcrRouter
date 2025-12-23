"""Backend factory for document processing."""

from ocrrouter.config import Settings
from .models.base import BaseModelBackend


def get_backend(backend_name: str, settings: Settings) -> BaseModelBackend:
    """Get a backend instance by name.

    This is a simple factory function using if/elif routing.
    When adding new backends, simply add another elif clause.

    Args:
        backend_name: Name of the backend to instantiate.
            Currently supported: "mineru", "deepseek", "dotsocr", "composite",
            "hunyuanocr", "generalvlm"
        settings: Settings object with configuration.

    Returns:
        Backend instance.

    Raises:
        ValueError: If the backend name is not recognized.

    Example:
        >>> from ocrrouter import get_backend, Settings
        >>> settings = Settings(openai_api_key="sk-...")
        >>> backend = get_backend("mineru", settings=settings)
        >>> middle_json, results = await backend.analyze(pdf_bytes, image_writer)
    """
    if backend_name == "mineru":
        from .models.mineru.backend import MinerUBackend

        return MinerUBackend(settings)
    elif backend_name == "deepseek":
        from .models.deepseek.backend import DeepSeekBackend

        return DeepSeekBackend(settings)
    elif backend_name == "dotsocr":
        from .models.dotsocr.backend import DotsOCRBackend

        return DotsOCRBackend(settings)
    elif backend_name == "composite":
        from .models.composite.backend import CompositeBackend

        return CompositeBackend(settings)
    elif backend_name == "hunyuanocr":
        from .models.hunyuanocr.backend import HunyuanOCRBackend

        return HunyuanOCRBackend(settings)
    elif backend_name == "generalvlm":
        from .models.generalvlm.backend import GeneralVLMBackend

        return GeneralVLMBackend(settings)
    else:
        available_backends = [
            "mineru",
            "deepseek",
            "dotsocr",
            "composite",
            "hunyuanocr",
            "generalvlm",
        ]
        raise ValueError(
            f"Unknown backend: '{backend_name}'. Available backends: {available_backends}"
        )
