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
    describe_title,
    entry_title,
    normalize_terms,
    select_tag_suggestions,
    tag_hubness,
    tag_label,
    title_is_informative,
)
from tagstudio.core.library.alchemy.enums import BrowsingState
from tagstudio.core.library.alchemy.fields import TextField
from tagstudio.core.library.alchemy.library import Library
from tagstudio.core.library.alchemy.models import Entry, Tag
from tagstudio.core.library.alchemy.tag_verifier import VerificationConfig, VerificationOutcome

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

    suggestions = index.suggest_tags([3], threshold=0.6, max_per_entry=5).suggestions

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.entry_id == 3
    assert suggestion.tag_name == "foo"
    assert suggestion.score == pytest.approx(0.707, abs=0.01)


def test_suggest_tags_skips_tags_already_applied(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None

    # Entry 1 ("foo") is already tagged with the "foo" tag.
    suggestions = index.suggest_tags([1], threshold=0.6, max_per_entry=5).suggestions

    assert all(suggestion.tag_name != "foo" for suggestion in suggestions)


def test_tag_label_omits_parent_names(library: Library):
    """Siblings must not share their parent's name inside the embedded label."""
    tag = library.get_tag_by_name("bar")
    assert tag is not None

    label = tag_label(tag)

    assert label.endswith("bar")
    assert "subbar" not in label
    assert "Category" not in label


def test_suggest_tags_keeps_one_tag_per_facet(embedding_library: Library):
    """Two tags under the same parent must never both be suggested."""
    index = embedding_library.semantic_index
    assert index is not None

    parent = embedding_library.get_tag_by_name("subbar")
    assert parent is not None
    sibling = Tag(
        name="bar two",
        parent_tags={parent},
        color_namespace="tagstudio-standard",
        color_slug="blue",
    )
    assert embedding_library.add_tag(sibling)

    entry = Entry(
        id=4,
        path=Path("bar_kitten.txt"),
        fields=[TextField(name="Description", value="a bar kitten")],
    )
    assert embedding_library.add_entries([entry])

    suggestions = index.suggest_tags(
        [4], threshold=0.5, max_per_entry=5, one_per_facet=True
    ).suggestions
    assert len(suggestions) == 1
    assert suggestions[0].tag_name in {"bar", "bar two"}

    # Disabling the facet rule restores the previous one-vs-all behaviour.
    suggestions = index.suggest_tags(
        [4], threshold=0.5, max_per_entry=5, one_per_facet=False
    ).suggestions
    assert {suggestion.tag_name for suggestion in suggestions} == {"bar", "bar two"}


def test_suggest_tags_skips_facet_filled_by_existing_tag(embedding_library: Library):
    """An entry tagged with one sibling must not be offered the other."""
    index = embedding_library.semantic_index
    assert index is not None

    parent = embedding_library.get_tag_by_name("subbar")
    assert parent is not None
    sibling = Tag(
        name="bar two",
        parent_tags={parent},
        color_namespace="tagstudio-standard",
        color_slug="blue",
    )
    assert embedding_library.add_tag(sibling)

    # Entry 2 is tagged with "bar", a sibling of "bar two".
    suggestions = index.suggest_tags(
        [2], threshold=0.5, max_per_entry=5, one_per_facet=True
    ).suggestions
    assert all(suggestion.tag_name != "bar two" for suggestion in suggestions)

    suggestions = index.suggest_tags(
        [2], threshold=0.5, max_per_entry=5, one_per_facet=False
    ).suggestions
    assert any(suggestion.tag_name == "bar two" for suggestion in suggestions)


@pytest.mark.parametrize(
    "title",
    [
        "PXL_20241212_183835927",
        "IMG_1234",
        "DSC_0001",
        "VID_20241212_184108801",
        "20241212_183835927",
        "WhatsApp Video 2024-12-12 at 18.35.92",
        "Screen Recording 2024-12-12 at 18.35.92",
        "untitled",
    ],
)
def test_camera_filenames_are_not_informative(title: str):
    assert not title_is_informative(title)


@pytest.mark.parametrize(
    "title",
    ["beach sunset", "foo kitten", "walk in the park with my dog", "birthday party"],
)
def test_descriptive_filenames_are_informative(title: str):
    assert title_is_informative(title)


def test_entry_title_replaces_camera_names_with_the_file_kind():
    """A timestamp name says nothing, so the model must not read it as content."""
    video = Entry(path=Path("PXL_20241212_183835927.mp4"), fields=[])
    image = Entry(path=Path("IMG_1234.jpg"), fields=[])
    named = Entry(path=Path("beach_sunset.mp4"), fields=[])

    assert entry_title(video) == "video"
    assert entry_title(image) == "image"
    assert entry_title(named) == "beach sunset"


def test_tag_hubness_penalises_tags_close_to_every_file():
    """A tag near every file must score higher than one that only fits a few."""
    broad = np.ones(4, dtype=np.float32)
    narrow = np.zeros(4, dtype=np.float32)
    narrow[0] = 1.0

    sample = np.vstack(
        [
            *(np.ones(4, dtype=np.float32) / 2.0 for _ in range(6)),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.6, 0.4, 0.0, 0.0], dtype=np.float32),
        ]
    )
    tag_vectors = np.vstack([broad, narrow])

    hubness = tag_hubness(tag_vectors, sample)

    assert hubness is not None
    assert hubness[0] > hubness[1]


def test_tag_hubness_needs_enough_files():
    tag_vectors = np.ones((1, 4), dtype=np.float32)
    assert tag_hubness(tag_vectors, np.ones((2, 4), dtype=np.float32)) is None


def test_select_tag_suggestions_drops_tags_below_the_margin(library: Library):
    """A hub tag can clear the threshold yet still be excluded by the margin."""
    tags = [
        Tag(id=1200, name="outdoors"),
        Tag(id=1201, name="nsfw"),
        Tag(id=1202, name="beach"),
    ]
    raw = np.array([0.62, 0.90, 0.83], dtype=np.float32)
    # "nsfw" is close to everything, so its corrected score falls below zero even
    # though its raw similarity is the highest of the three.
    scores = np.array([0.05, -0.04, 0.11], dtype=np.float32)

    selected = select_tag_suggestions(
        scores,
        tags,
        threshold=0.6,
        max_per_entry=5,
        raw_scores=raw,
        min_margin=0.0,
    )

    assert [tag.name for tag, _score in selected] == ["beach", "outdoors"]
    assert all(0.6 <= score <= 1.0 for _tag, score in selected)


def test_suggest_tags_reports_files_without_suggestions(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None

    result = index.suggest_tags([3], threshold=0.99, max_per_entry=5)

    assert result.suggestions == []
    assert [entry_id for entry_id, _name in result.unsuggested] == [3]


def test_capture_noise_is_not_read_as_content():
    """A render flag, a clock time and a resolution describe the capture, not the frame."""
    title = describe_title(normalize_terms("VRChat_DLSS_2026-09-04_17-37-48.282_1280x720"))

    assert title == "VRChat"
    assert title_is_informative(title)


def test_a_lone_number_stays_part_of_the_name():
    assert describe_title(normalize_terms("2001_a_space_odyssey")) == "2001 a space odyssey"
    assert describe_title(normalize_terms("boeing_747")) == "boeing 747"
    # Two or more numbers in a row are a capture timestamp, not a name.
    assert describe_title(normalize_terms("IMG_20241212_183835927")) == "IMG"


def test_resolutions_and_pipeline_flags_alone_are_not_informative():
    assert not title_is_informative("DLSS 1280x720")
    assert not title_is_informative("1920x1080 60fps")


class StubVerifier:
    """Stands in for the vision-language check without any HTTP."""

    def __init__(self, keep: set[int] | None, error: str = ""):
        self.config = VerificationConfig(base_url="http://localhost:1234/v1", model="stub-vl")
        self.keep = keep
        self.error = error
        self.calls: list[list[str]] = []

    def verify(self, image_path, labels):
        self.calls.append(list(labels))
        kept = None if self.keep is None else frozenset(self.keep)
        return VerificationOutcome(kept=kept, error=self.error)


def _add_bar_kitten(embedding_library: Library) -> int:
    entry = Entry(
        id=4,
        path=Path("bar_kitten.txt"),
        fields=[TextField(name="Description", value="a bar kitten")],
    )
    assert embedding_library.add_entries([entry])
    return 4


def _pretend_files_have_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tagstudio.core.library.alchemy.embeddings.media_path_for_entry",
        lambda *args, **kwargs: Path("bar_kitten.png"),
    )


def test_suggest_tags_drops_every_tag_the_verifier_rejects(
    embedding_library: Library, monkeypatch: pytest.MonkeyPatch
):
    index = embedding_library.semantic_index
    assert index is not None
    entry_id = _add_bar_kitten(embedding_library)
    _pretend_files_have_images(monkeypatch)

    baseline = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5)
    assert baseline.suggestions
    assert baseline.verified == 0

    verifier = StubVerifier(keep=set())
    result = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5, verifier=verifier)

    assert verifier.calls == [[suggestion.tag_name for suggestion in baseline.suggestions]]
    assert result.suggestions == []
    assert result.verified == 1
    assert result.dropped == len(baseline.suggestions)
    assert [unsuggested_id for unsuggested_id, _name in result.unsuggested] == [entry_id]

    # What the model was asked is kept so the modal can show it for debugging.
    assert len(result.verify_notes) == 1
    note = result.verify_notes[0]
    assert note.entry_name == "bar_kitten.txt"
    assert list(note.labels) == verifier.calls[0]
    assert note.kept == 0


def test_suggest_tags_keeps_only_accepted_tags(
    embedding_library: Library, monkeypatch: pytest.MonkeyPatch
):
    index = embedding_library.semantic_index
    assert index is not None
    entry_id = _add_bar_kitten(embedding_library)
    _pretend_files_have_images(monkeypatch)

    verifier = StubVerifier(keep={0})
    result = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5, verifier=verifier)

    offered = verifier.calls[0]
    assert [suggestion.tag_name for suggestion in result.suggestions] == [offered[0]]
    assert result.dropped == len(offered) - 1
    assert result.verify_model == "stub-vl"


def test_suggest_tags_keeps_suggestions_when_verification_fails(
    embedding_library: Library, monkeypatch: pytest.MonkeyPatch
):
    """A failed request must never look like the model rejecting every tag."""
    index = embedding_library.semantic_index
    assert index is not None
    entry_id = _add_bar_kitten(embedding_library)
    _pretend_files_have_images(monkeypatch)

    baseline = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5)
    verifier = StubVerifier(keep=None, error="connection refused")
    result = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5, verifier=verifier)

    assert [suggestion.tag_id for suggestion in result.suggestions] == [
        suggestion.tag_id for suggestion in baseline.suggestions
    ]
    assert result.verified == 0
    assert result.unverified == 1
    assert "connection refused" in result.verify_error


def test_suggest_tags_does_not_verify_files_without_an_image(embedding_library: Library):
    index = embedding_library.semantic_index
    assert index is not None
    entry_id = _add_bar_kitten(embedding_library)

    verifier = StubVerifier(keep=set())
    result = index.suggest_tags([entry_id], threshold=0.5, max_per_entry=5, verifier=verifier)

    assert verifier.calls == []
    assert result.unverified == 1
    assert result.suggestions


def test_suggest_tags_announces_the_verifying_stage(
    embedding_library: Library, monkeypatch: pytest.MonkeyPatch
):
    index = embedding_library.semantic_index
    assert index is not None
    entry_id = _add_bar_kitten(embedding_library)
    _pretend_files_have_images(monkeypatch)
    stages: list[str] = []

    index.suggest_tags(
        [entry_id],
        threshold=0.5,
        verifier=StubVerifier(keep={0}),
        stage=stages.append,
    )

    assert stages == ["verifying"]


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
