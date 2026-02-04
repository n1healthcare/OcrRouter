"""PP-DocLayoutV3 backend for document layout detection."""

from .backend import PPDocLayoutBackend
from .client import PPDocLayoutClient

__all__ = ["PPDocLayoutBackend", "PPDocLayoutClient"]
