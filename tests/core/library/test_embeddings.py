# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


import re
from pathlib import Path

import numpy as np
import pytest

from tagstudio.core.library.alchemy.embeddings import (
    EmbeddingConfig,
    EmbeddingService,
    EmbeddingUnavailableError,
    _job_batches,
)
from tagstudio.core.library.alchemy.enums import BrowsingState
from tagstudio.core.library.alchemy.fields import TextField
from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry

KEYWORDS = {
    "kitten": 0,
    "cat": 0,
    "foo": 1,
    "bar": 2,
}


class FakeEmbeddingBackend:
    """Small deterministic backend used to test the semantic index without a model."""

    def encode(self, inputs):
        vectors = []
        for item in inputs:
            text = item["text"] if isinstance(item, dict) else item
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            vector = np.zeros(8, dtype=np.float32)
            for keyword, index in KEYWORDS.items():
                if keyword in words:
                    vector[index] += 1
            if not vector.any():
                vector[7] = 1
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors)


class UnavailableEmbeddingBackend:
    """Backend that behaves like missing model dependencies."""

    def encode(self, inputs):
        raise EmbeddingUnavailableError("The embedding model could not be loaded.")


@pytest.fixture
def embedding_library(library: Library) -> Library:
    entry = Entry(
        id=3,
        path=Path("foo_kitten.txt"),
        fields=[TextField(name="Description", value="a fluffy kitten playing outside")],
    )
    assert library.add_entries([entry])

    library.configure_semantic_index(
        EmbeddingConfig(model="test-model", dimensions=8, use_images=False),
        backend=FakeEmbeddingBackend(),
    )
    return library


def test_index_entries_is_incremental(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None
    assert index.has_index() is False

    report = index.index_entries()
    assert report.indexed == 3
    assert report.skipped == 0
    assert report.failed == 0
    assert index.has_index() is True

    # Running it again should not re-embed unchanged entries.
    report = index.index_entries()
    assert report.indexed == 0
    assert report.skipped == 3

    # Changing an entry invalidates only that entry.
    assert embedding_library.update_entry_path(3, Path("foo_kitten_moved.txt"))
    report = index.index_entries()
    assert report.indexed == 1
    assert report.skipped == 2


def test_job_batches_isolates_images_and_limits_text_batches():
    text_job = (Entry(path=Path("example.txt"), fields=[]), "title: example", "")
    image_job = (
        Entry(path=Path("example.png"), fields=[]),
        {"text": "title: example <|image|>", "image": "example.png"},
        "",
    )

    batches = list(_job_batches([image_job, *[text_job] * 9, image_job]))

    assert [len(batch) for batch in batches] == [1, 8, 1, 1]
    assert batches[0] == [image_job]
    assert batches[-1] == [image_job]


def test_resolve_device_falls_back_to_available_accelerator():
    torch = pytest.importorskip("torch")

    resolved = EmbeddingService.resolve_device("cuda")
    assert resolved == ("cuda" if torch.cuda.is_available() else "cpu")
    assert EmbeddingService.resolve_dtype("cpu") == torch.float32


def test_index_entries_reports_progress(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None
    calls: list[tuple[int, int, str]] = []

    index.index_entries(progress=lambda done, total, label: calls.append((done, total, label)))

    assert calls
    assert calls[-1][0] == calls[-1][1]


def test_semantic_search_ranks_by_similarity(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None
    index.index_entries()

    results = embedding_library.search_library(
        BrowsingState(query="kitten", semantic=True), page_size=0
    )
    assert results.total_count == 3
    assert results.ids[0] == 3

    paged = embedding_library.search_library(
        BrowsingState(query="kitten", semantic=True), page_size=1
    )
    assert paged.total_count == 3
    assert paged.ids == [3]


def test_semantic_search_without_index_returns_nothing(embedding_library: Library):
    results = embedding_library.search_library(
        BrowsingState(query="kitten", semantic=True), page_size=0
    )
    assert results.total_count == 0
    assert results.ids == []


def test_suggest_tags_uses_semantic_similarity(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None

    suggestions = index.suggest_tags([3], threshold=0.6, max_per_entry=5)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.entry_id == 3
    assert suggestion.tag_name == "foo"
    assert suggestion.score == pytest.approx(0.707, abs=0.01)


def test_suggest_tags_skips_tags_already_applied(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None

    # Entry 1 ("foo") is already tagged with the "foo" tag.
    suggestions = index.suggest_tags([1], threshold=0.6, max_per_entry=5)

    assert all(suggestion.tag_name != "foo" for suggestion in suggestions)


def test_clear_entry_embeddings(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None
    index.index_entries()

    assert embedding_library.clear_entry_embeddings() == 3
    assert index.has_index() is False


def test_index_entries_propagates_unavailable_backend(embedding_library: Library):
    embedding_library.configure_semantic_index(
        EmbeddingConfig(model="test-model", dimensions=8, use_images=False),
        backend=UnavailableEmbeddingBackend(),
    )
    index = embedding_library.semantic_index
    assert index is not None

    with pytest.raises(EmbeddingUnavailableError):
        index.index_entries()
