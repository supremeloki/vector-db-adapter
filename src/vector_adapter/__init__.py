from .core import (
    CollectionNotFoundError,
    DimensionMismatchError,
    InMemoryBackend,
    SearchHit,
    SqliteBackend,
    VectorAdapter,
    VectorBackend,
    VectorRecord,
    VectorStoreError,
    cosine_distance,
)

__all__ = [
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "InMemoryBackend",
    "SearchHit",
    "SqliteBackend",
    "VectorAdapter",
    "VectorBackend",
    "VectorRecord",
    "VectorStoreError",
    "cosine_distance",
]

__version__ = "0.1.0"
