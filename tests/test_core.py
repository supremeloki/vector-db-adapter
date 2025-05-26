import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vector_adapter import (
    CollectionNotFoundError,
    InMemoryBackend,
    SqliteBackend,
    VectorAdapter,
    VectorRecord,
)


@pytest.fixture
def memory_adapter():
    adapter = VectorAdapter(InMemoryBackend())
    adapter.backend.create_collection("docs")
    return adapter


@pytest.fixture
def sqlite_adapter():
    adapter = VectorAdapter(SqliteBackend())
    return adapter


VECTORS = {
    "doc:1": [1.0, 0.0, 0.0],
    "doc:2": [0.9, 0.1, 0.0],
    "doc:3": [0.0, 1.0, 0.0],
}


def test_memory_add_and_count(memory_adapter):
    written = memory_adapter.add_documents("docs", list(VECTORS), list(VECTORS.values()))
    assert written == 3
    assert memory_adapter.count("docs") == 3


def test_memory_search_ranks_by_similarity(memory_adapter):
    ids = list(VECTORS)
    memory_adapter.add_documents("docs", ids, list(VECTORS.values()))
    hits = memory_adapter.search("docs", VECTORS["doc:1"], top_k=2)
    assert hits[0].record_id in {"doc:1", "doc:2"}
    assert hits[0].score <= hits[1].score


def test_memory_delete(memory_adapter):
    memory_adapter.add_documents("docs", ["a", "b"], [[1.0], [2.0]])
    assert memory_adapter.remove("docs", ["a"]) == 1
    assert memory_adapter.count("docs") == 1


def test_unknown_collection_raises(memory_adapter):
    with pytest.raises(CollectionNotFoundError):
        memory_adapter.search("ghost", [1.0])


def test_length_mismatch_rejected(memory_adapter):
    with pytest.raises(Exception):
        memory_adapter.add_documents("docs", ["only-id"], [[1.0], [2.0]])


def test_sqlite_backend_roundtrip(sqlite_adapter):
    sqlite_adapter.add_documents("kb", ["k1", "k2"], [[1.0, 0.0], [0.0, 1.0]],
                                 metadatas=[{"tag": "a"}, {"tag": "b"}])
    assert sqlite_adapter.count("kb") == 2
    hits = sqlite_adapter.search("kb", [1.0, 0.0])
    assert hits[0].record_id == "k1"
    assert hits[0].metadata["tag"] == "a"


def test_sqlite_upsert_overwrites(sqlite_adapter):
    sqlite_adapter.add_documents("c", ["same"], [[1.0, 0.0]])
    sqlite_adapter.add_documents("c", ["same"], [[0.0, 1.0]])
    hits = sqlite_adapter.search("c", [0.0, 1.0])
    assert hits[0].record_id == "same"


def test_memory_and_sqlite_agree_on_ranking():
    ids = ["near", "far"]
    vectors = [[1.0, 0.1], [0.0, 1.0]]
    query = [1.0, 0.0]

    mem = VectorAdapter(InMemoryBackend())
    sql = VectorAdapter(SqliteBackend())
    for backend in (mem, sql):
        backend.add_documents("cmp", ids, vectors)

    mem_top = mem.search("cmp", query, top_k=1)[0]
    sql_top = sql.search("cmp", query, top_k=1)[0]
    assert mem_top.record_id == sql_top.record_id


def test_metadata_preserved_in_hits(memory_adapter):
    memory_adapter.add_documents(
        "meta-docs", ["m1"], [[1.0]], metadatas=[{"chapter": 7}],
    )
    hit = memory_adapter.search("meta-docs", [1.0])[0]
    assert hit.metadata["chapter"] == 7
