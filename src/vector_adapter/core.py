from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


class VectorStoreError(Exception):
    pass


class CollectionNotFoundError(VectorStoreError):
    def __init__(self, name: str) -> None:
        super().__init__(f"collection not found: {name!r}")


class DimensionMismatchError(VectorStoreError):
    def __init__(self, collection: str, expected: int, actual: int) -> None:
        super().__init__(
            f"collection {collection!r} expects {expected} dims, got {actual}"
        )


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class SearchHit:
    record_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorBackend(Protocol):
    backend_name: str

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int: ...

    def query(self, collection: str, vector: Sequence[float],
              top_k: int) -> list[SearchHit]: ...

    def delete(self, collection: str, record_ids: Sequence[str]) -> int: ...

    def count(self, collection: str) -> int: ...


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))


class InMemoryBackend:
    backend_name = "memory"

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorRecord]] = {}

    def _require(self, collection: str) -> dict[str, VectorRecord]:
        store = self._collections.get(collection)
        if store is None:
            raise CollectionNotFoundError(collection)
        return store

    def create_collection(self, name: str, dimension: int | None = None) -> None:
        if name in self._collections:
            raise VectorStoreError(f"collection exists: {name!r}")
        self._collections[name] = {}
        self._dimensions[name] = dimension

    _dimensions: dict[str, int | None] = {}

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        store = self._collections.setdefault(collection, {})
        declared = self._dimensions.get(collection)
        for record in records:
            if declared is not None and record.dimension != declared:
                raise DimensionMismatchError(collection, declared, record.dimension)
            store[record.record_id] = record
        return len(records)

    def query(self, collection: str, vector: Sequence[float],
              top_k: int = 5) -> list[SearchHit]:
        store = self._require(collection)
        scored = [
            SearchHit(record_id=rid,
                      score=round(cosine_distance(vector, rec.vector), 6),
                      metadata=dict(rec.metadata))
            for rid, rec in store.items()
        ]
        scored.sort(key=lambda hit: hit.score)
        return scored[:top_k]

    def delete(self, collection: str, record_ids: Sequence[str]) -> int:
        store = self._require(collection)
        removed = 0
        for rid in record_ids:
            if store.pop(rid, None) is not None:
                removed += 1
        return removed

    def count(self, collection: str) -> int:
        return len(self._require(collection))


InMemoryBackend._dimensions = {}


class SqliteBackend:
    backend_name = "sqlite"

    def __init__(self, path: Path | None = None) -> None:
        self._conn = sqlite3.connect(str(path or ":memory:"))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT NOT NULL,
                record_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (collection, record_id)
            )
        """)
        self._conn.commit()

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> int:
        written = 0
        for record in records:
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors "
                "(collection, record_id, dimension, vector, metadata) VALUES (?, ?, ?, ?, ?)",
                (collection, record.record_id, record.dimension,
                 json.dumps(record.vector), json.dumps(record.metadata)),
            )
            written += 1
        self._conn.commit()
        return written

    def query(self, collection: str, vector: Sequence[float],
              top_k: int = 5) -> list[SearchHit]:
        rows = self._conn.execute(
            "SELECT record_id, vector, metadata FROM vectors WHERE collection = ?",
            (collection,),
        ).fetchall()
        hits = []
        for record_id, raw_vector, raw_meta in rows:
            stored = tuple(float(v) for v in json.loads(raw_vector))
            hits.append(SearchHit(
                record_id=record_id,
                score=round(cosine_distance(vector, stored), 6),
                metadata=json.loads(raw_meta),
            ))
        hits.sort(key=lambda h: h.score)
        return hits[:top_k]

    def delete(self, collection: str, record_ids: Sequence[str]) -> int:
        removed = 0
        for rid in record_ids:
            cursor = self._conn.execute(
                "DELETE FROM vectors WHERE collection = ? AND record_id = ?",
                (collection, rid),
            )
            removed += cursor.rowcount
        self._conn.commit()
        return removed

    def count(self, collection: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)
        ).fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()


class VectorAdapter:
    def __init__(self, backend: VectorBackend) -> None:
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return self.backend.backend_name

    def add_documents(self, collection: str,
                      ids: Sequence[str], vectors: Sequence[Sequence[float]],
                      metadatas: Sequence[dict[str, Any]] | None = None) -> int:
        if not (len(ids) == len(vectors)):
            raise VectorStoreError("ids/vectors length mismatch")
        metas = metadatas or [{} for _ in ids]
        records = [
            VectorRecord(record_id=rid, vector=tuple(float(x) for x in vec), metadata=meta)
            for rid, vec, meta in zip(ids, vectors, metas)
        ]
        return self.backend.upsert(collection, records)

    def search(self, collection: str, query_vector: Sequence[float],
               top_k: int = 5) -> list[SearchHit]:
        return self.backend.query(collection, query_vector, top_k)

    def remove(self, collection: str, ids: Sequence[str]) -> int:
        return self.backend.delete(collection, ids)

    def count(self, collection: str) -> int:
        return self.backend.count(collection)
