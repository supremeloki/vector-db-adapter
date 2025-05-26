# vector-adapter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Swap vector databases without rewriting retrieval code: one `VectorAdapter` over a `VectorBackend` protocol, with in-memory and SQLite reference implementations that rank identically.

## 🚀 Overview

Retrieval code shouldn't care whether vectors live in Chroma, Qdrant, or SQLite. `vector-adapter` defines the four operations every store needs (`upsert/query/delete/count`) behind a structural protocol, then ships two interchangeable backends: an **InMemoryBackend** for tests and small datasets, and a **SqliteBackend** with JSON-vector rows for durable local persistence. Both rank by cosine distance and return identical orderings — verified by test.

## ✨ Features

- **Backend protocol:** structural typing; new stores implement four methods, no inheritance
- **Two backends:** memory (fast, ephemeral) · sqlite (durable, file-backed or :memory:)
- **Metadata round-trip:** per-record metadata stored and returned inside every hit
- **Upsert semantics:** same record_id replaces cleanly
- **Adapter facade:** length-mismatch guards and normalized float conversion
- **Identical ranking guarantee:** cross-backend ordering asserted by tests
- **Zero dependencies**

## 🚧 Structure

```
vector-db-adapter/
├── src/vector_adapter/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/vector-db-adapter.git
cd vector-db-adapter
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies (sqlite3 is stdlib)

## 🏃 Quick Start

```python
from vector_adapter import SqliteBackend, VectorAdapter

adapter = VectorAdapter(SqliteBackend("vectors.db"))
adapter.add_documents(
    "knowledge", ["doc:1", "doc:2"],
    [[1.0, 0.0], [0.0, 1.0]],
    metadatas=[{"topic": "db"}, {"topic": "art"}],
)
hits = adapter.search("knowledge", [1.0, 0.05])
print(hits[0].record_id, hits[0].metadata)
```

## 🔧 Error Handling

```text
VectorStoreError
├── CollectionNotFoundError     # query/delete on unknown collection
└── DimensionMismatchError      # declared dim ≠ vector dim (memory backend)
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen records/hits
- Zero comments — names carry the meaning
- Cross-backend ranking equivalence explicitly tested

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
