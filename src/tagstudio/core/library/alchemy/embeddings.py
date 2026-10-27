# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Optional semantic search and auto-tagging powered by EmbeddingGemma 2.

The heavy model dependencies are imported lazily so that TagStudio can run without them.
Install the optional dependencies with::

    pip install "tagstudio[embeddings]"
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import structlog

from tagstudio.core.constants import RESERVED_TAG_END
from tagstudio.core.library.alchemy.models import Entry, Tag

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.library import Library


logger = structlog.get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2"
SUPPORTED_EMBEDDING_DIMENSIONS = (128, 256, 512, 768)

TASK_SEARCH = "search"
TASK_TAGGING = "tagging"

SEARCH_QUERY_PREFIX = "task: search result | query: "
CLASSIFICATION_PREFIX = "task: classification | query: "

# Keep the text sent to the model well below its 8K token context window.
MAX_ENTRY_TEXT_LENGTH = 4000

# Formats that the EmbeddingGemma 2 vision encoder can read directly through Pillow.
IMAGE_EXTENSIONS = frozenset(
    {
        ".apng",
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".heif",
        ".ico",
        ".j2k",
        ".jp2",
        ".jpeg",
        ".jpg",
        ".jxl",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)

EntryEmbeddingInput = str | dict[str, Any]
EntryJob = tuple[Entry, EntryEmbeddingInput, str]
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the optional embedding model dependencies are unavailable."""


class EmbeddingBackend(Protocol):
    """Minimal interface required by :class:`SemanticIndex` (also used by tests)."""

    def encode(self, inputs: Sequence[EntryEmbeddingInput]) -> np.ndarray:
        """Return L2-normalized float32 vectors, one per input."""
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = 256
    device: str = "auto"
    use_images: bool = True

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")


@dataclass(frozen=True)
class EntryEmbeddingRecord:
    entry_id: int
    model: str
    task: str
    dimensions: int
    vector: bytes
    content_hash: str


@dataclass(frozen=True)
class IndexReport:
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass(frozen=True)
class AutoTagSuggestion:
    entry_id: int
    entry_name: str
    tag_id: int
    tag_name: str
    score: float


def embedding_dependency_hint() -> str:
    """Return installation instructions that match how TagStudio is running."""
    if getattr(sys, "frozen", False):
        return (
            "This TagStudio build does not include the optional embedding runtime "
            '(sentence-transformers and PyTorch). Rebuild it with the "embeddings" '
            "extra installed; see the Building section of the documentation."
        )
    return 'Install the optional dependencies with: pip install "tagstudio[embeddings]"'


class EmbeddingService:
    """Lazily-loaded EmbeddingGemma 2 model wrapper.

    The model is only loaded the first time an embedding is needed. This keeps startup
    fast and lets TagStudio work normally without the optional dependencies installed.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        """Load the model if it is not already in memory.

        Callers can invoke this ahead of encoding to distinguish model loading from
        inference when reporting progress.
        """
        return self._load()

    @staticmethod
    def resolve_device(device: str) -> str:
        """Resolve a device preference to one torch can actually use.

        If the requested accelerator is not available (for example when a CPU-only
        PyTorch build is installed), fall back to CPU instead of failing to load.
        """
        requested = (device or "auto").strip()
        lowered = requested.lower()

        import torch  # pyright: ignore[reportMissingImports]

        if lowered == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"

        if lowered.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "[Embeddings] CUDA was requested but is not available, using CPU",
                torch_version=torch.__version__,
                cuda_build=torch.version.cuda,
            )
            return "cpu"

        if lowered.startswith("mps"):
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                logger.warning(
                    "[Embeddings] MPS was requested but is not available, using CPU",
                    torch_version=torch.__version__,
                )
                return "cpu"

        return requested

    @staticmethod
    def resolve_dtype(device: str) -> Any:
        """Pick a dtype that is fast on the target device.

        EmbeddingGemma 2 was trained in bfloat16, but torch's CPU bfloat16 kernels are
        several times slower than float32. Bfloat16 is only worth enabling on CUDA.
        """
        import torch  # pyright: ignore[reportMissingImports]

        if device.lower().startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def _load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                    SentenceTransformer,
                )
            except ImportError as e:
                raise EmbeddingUnavailableError(
                    f"EmbeddingGemma 2 requires the optional 'sentence-transformers' package. "
                    f"{embedding_dependency_hint()}"
                ) from e

            config_kwargs: dict[str, Any] = {"audio_config": None}
            if not self.config.use_images:
                config_kwargs["vision_config"] = None

            device = self.resolve_device(self.config.device)
            kwargs: dict[str, Any] = {
                "config_kwargs": config_kwargs,
                "truncate_dim": self.config.dimensions,
                "device": device,
                "model_kwargs": {"dtype": self.resolve_dtype(device)},
            }

            logger.info(
                "[Embeddings] Loading model",
                model=self.config.model,
                dimensions=self.config.dimensions,
                device=device,
                use_images=self.config.use_images,
            )
            try:
                self._model = SentenceTransformer(self.config.model, **kwargs)
            except Exception as e:
                raise EmbeddingUnavailableError(
                    f"Could not load the embedding model '{self.config.model}': {e}"
                ) from e
            return self._model

    def encode(self, inputs: Sequence[EntryEmbeddingInput]) -> np.ndarray:
        """Encode a batch of text and/or interleaved media inputs."""
        if not inputs:
            return np.empty((0, self.config.dimensions), dtype=np.float32)

        with self._lock:
            model = self._load()
            raw: Any
            try:
                raw = model.encode(
                    list(inputs),
                    batch_size=8,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                if len(inputs) == 1:
                    raise
                # Some model/provider combinations do not support batching mixed
                # multimodal inputs. Fall back to encoding each input individually.
                raw = [model.encode([item], normalize_embeddings=True)[0] for item in inputs]

        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] > self.config.dimensions:
            # Matryoshka truncation fallback if the loaded model did not apply it.
            vectors = vectors[:, : self.config.dimensions]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        elif vectors.shape[1] != self.config.dimensions:
            raise EmbeddingUnavailableError(
                f"The embedding model returned {vectors.shape[1]} dimensions, "
                f"but {self.config.dimensions} were requested."
            )
        return np.asarray(vectors, dtype=np.float32)


def normalize_terms(text: str) -> str:
    """Make filenames and folder names more natural for a text embedding model."""
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def entry_title(entry: Entry) -> str:
    return normalize_terms(entry.path.stem) or entry.path.name


def entry_content(entry: Entry) -> str:
    """Build the descriptive text used to embed an entry."""
    parts: list[str] = []

    parent = entry.path.parent
    if parent.parts and str(parent) not in ("", "."):
        parts.append(f"Folder: {normalize_terms(parent.as_posix())}")

    tag_names = sorted(tag.name for tag in entry.tags if tag.name)
    if tag_names:
        parts.append(f"Tags: {', '.join(tag_names)}")

    for field in entry.text_fields:
        if field.value and field.value.strip():
            parts.append(f"{field.name}: {field.value.strip()}")

    text = "\n".join(parts)
    return text[:MAX_ENTRY_TEXT_LENGTH]


def entry_search_text(entry: Entry) -> str:
    text = entry_content(entry)
    return f"title: {entry_title(entry)} | text: {text or entry_title(entry)}"


def entry_tagging_text(entry: Entry) -> str:
    content = f"File: {entry_title(entry)}"
    extra = entry_content(entry)
    if extra:
        content = f"{content}\n{extra}"
    return CLASSIFICATION_PREFIX + content[:MAX_ENTRY_TEXT_LENGTH]


def tag_label(tag: Tag) -> str:
    """Build the classification label text for a tag."""
    aliases = sorted({alias.name for alias in tag.aliases if alias.name})
    parents = sorted({parent.name for parent in tag.parent_tags if parent.name})

    label = tag.name
    if aliases:
        label += f" (also known as: {', '.join(aliases)})"
    if parents:
        label += f". Category: {', '.join(parents)}"
    return CLASSIFICATION_PREFIX + label


def image_path_for_entry(entry: Entry, library_dir: Path | None) -> Path | None:
    if library_dir is None:
        return None
    ext = entry.path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        return None
    path = library_dir / entry.path
    if not path.is_file():
        return None
    return path


def build_entry_input(
    entry: Entry,
    task: str,
    library_dir: Path | None,
    use_images: bool,
) -> tuple[EntryEmbeddingInput, str]:
    """Return the model input and a content hash for an entry."""
    text = entry_search_text(entry) if task == TASK_SEARCH else entry_tagging_text(entry)
    image_path = image_path_for_entry(entry, library_dir) if use_images else None

    image_fingerprint = ""
    if image_path is not None:
        stat = image_path.stat()
        image_fingerprint = f"{image_path.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}"
        model_input: EntryEmbeddingInput = {
            "text": f"{text} <|image|>",
            "image": str(image_path),
        }
    else:
        model_input = text

    digest = hashlib.sha256()
    digest.update(task.encode())
    digest.update(text.encode())
    digest.update(image_fingerprint.encode())
    return model_input, digest.hexdigest()


def _job_batches(
    jobs: Sequence[EntryJob], text_batch_size: int = 8
) -> Iterable[Sequence[EntryJob]]:
    """Group jobs into model batches, isolating images so progress advances per file.

    Batching image inputs does not speed up encoding on CPU, and a whole batch would
    otherwise leave the progress indicator frozen until every image finished.
    """
    text_batch: list[EntryJob] = []
    for job in jobs:
        model_input = job[1]
        if isinstance(model_input, dict) and model_input.get("image"):
            if text_batch:
                yield text_batch
                text_batch = []
            yield [job]
            continue

        text_batch.append(job)
        if len(text_batch) >= text_batch_size:
            yield text_batch
            text_batch = []
    if text_batch:
        yield text_batch


class SemanticIndex:
    """Stores and queries EmbeddingGemma 2 vectors for a :class:`Library`."""

    def __init__(
        self,
        library: Library,
        config: EmbeddingConfig,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self.library = library
        self.config = config
        self._backend: EmbeddingBackend = backend or EmbeddingService(config)
        self._tag_cache_key: tuple[Any, ...] | None = None
        self._tag_cache: tuple[list[Tag], np.ndarray] | None = None

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    def has_index(self) -> bool:
        return (
            self.library.count_entry_embeddings(
                self.config.model, TASK_SEARCH, self.config.dimensions
            )
            > 0
        )

    def prepare(self) -> None:
        """Load the embedding model ahead of indexing, if it is not loaded yet."""
        load = getattr(self._backend, "load", None)
        if callable(load):
            load()

    def index_entries(
        self,
        entry_ids: Sequence[int] | None = None,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> IndexReport:
        """Create or refresh search embeddings for the given (or all) entries."""
        entries = self.library.get_entries_for_embedding(entry_ids)
        existing = self.library.get_entry_embedding_hashes(
            self.config.model, TASK_SEARCH, self.config.dimensions
        )

        jobs: list[EntryJob] = []
        skipped = 0
        failed = 0
        for entry in entries:
            try:
                model_input, content_hash = build_entry_input(
                    entry, TASK_SEARCH, self.library.library_dir, self.config.use_images
                )
            except OSError as e:
                logger.warning(
                    "[Embeddings] Could not read entry for embedding",
                    entry_id=entry.id,
                    path=entry.path,
                    error=e,
                )
                failed += 1
                continue
            if existing.get(entry.id) == content_hash:
                skipped += 1
                continue
            jobs.append((entry, model_input, content_hash))

        total = len(jobs)
        indexed = 0
        processed = 0
        if progress:
            progress(0, total, "")

        for batch in _job_batches(jobs):
            if should_cancel and should_cancel():
                break

            inputs = [job[1] for job in batch]
            try:
                vectors = self._backend.encode(inputs)
            except EmbeddingUnavailableError:
                raise
            except Exception as e:
                logger.warning("[Embeddings] Batch encode failed, retrying individually", error=e)
                vectors = np.empty((0, self.config.dimensions), dtype=np.float32)

            records: list[EntryEmbeddingRecord] = []
            if len(vectors) == len(batch):
                for (entry, _model_input, content_hash), vector in zip(batch, vectors, strict=True):
                    records.append(self._record(entry.id, TASK_SEARCH, vector, content_hash))
            else:
                for entry, model_input, content_hash in batch:
                    if should_cancel and should_cancel():
                        break
                    try:
                        vector = self._backend.encode([model_input])[0]
                        records.append(self._record(entry.id, TASK_SEARCH, vector, content_hash))
                    except EmbeddingUnavailableError:
                        raise
                    except Exception as e:
                        logger.warning(
                            "[Embeddings] Could not embed entry",
                            entry_id=entry.id,
                            path=entry.path,
                            error=e,
                        )
                        failed += 1

            if records:
                self.library.save_entry_embeddings(records)
                indexed += len(records)

            processed += len(batch)
            if progress:
                progress(processed, total, batch[-1][0].path.name)

        return IndexReport(total=total, indexed=indexed, skipped=skipped, failed=failed)

    def search_entry_ids(
        self,
        query: str,
        *,
        show_hidden: bool = False,
    ) -> list[int]:
        """Return entry IDs ranked by semantic similarity to ``query``."""
        query = query.strip()
        if not query or not self.has_index():
            return []

        query_vector = self._backend.encode([SEARCH_QUERY_PREFIX + query])[0]
        ids, vectors = self.library.load_entry_embedding_vectors(
            self.config.model,
            TASK_SEARCH,
            self.config.dimensions,
            include_hidden=show_hidden,
        )
        if not ids or vectors.size == 0:
            return []

        scores = vectors @ query_vector
        order = np.argsort(-scores)
        return [ids[int(i)] for i in order]

    def _candidate_tags(self) -> list[Tag]:
        tags = []
        for tag in self.library.tags:
            if tag.id is None or tag.id <= RESERVED_TAG_END:
                continue
            if tag.is_category or tag.is_hidden or not tag.name:
                continue
            tags.append(tag)
        tags.sort(key=lambda tag: tag.name.lower())
        return tags

    def _tag_embeddings(self, tags: Sequence[Tag]) -> tuple[list[Tag], np.ndarray]:
        signature: tuple[Any, ...] = (
            self.config.model,
            self.config.dimensions,
            tuple(
                (
                    tag.id,
                    tag.name,
                    tuple(sorted(alias.name for alias in tag.aliases if alias.name)),
                    tuple(sorted(parent.name for parent in tag.parent_tags if parent.name)),
                )
                for tag in tags
            ),
        )
        if self._tag_cache_key == signature and self._tag_cache is not None:
            return self._tag_cache

        vectors = self._backend.encode([tag_label(tag) for tag in tags])
        if len(vectors) != len(tags):
            raise EmbeddingUnavailableError("The embedding backend returned an unexpected shape.")
        self._tag_cache_key = signature
        self._tag_cache = (list(tags), vectors)
        return self._tag_cache

    def _ensure_tagging_embeddings(
        self,
        entries: Sequence[Entry],
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> dict[int, np.ndarray]:
        existing_hashes = self.library.get_entry_embedding_hashes(
            self.config.model, TASK_TAGGING, self.config.dimensions
        )
        records = self.library.load_entry_embedding_records(
            [entry.id for entry in entries],
            self.config.model,
            TASK_TAGGING,
            self.config.dimensions,
        )

        vectors_by_id: dict[int, np.ndarray] = {
            entry_id: np.frombuffer(vector, dtype=np.float32)
            for entry_id, (_content_hash, vector) in records.items()
        }

        jobs: list[EntryJob] = []
        for entry in entries:
            try:
                model_input, content_hash = build_entry_input(
                    entry, TASK_TAGGING, self.library.library_dir, self.config.use_images
                )
            except OSError as e:
                logger.warning(
                    "[Embeddings] Could not read entry for tagging",
                    entry_id=entry.id,
                    path=entry.path,
                    error=e,
                )
                continue
            if existing_hashes.get(entry.id) == content_hash and entry.id in vectors_by_id:
                continue
            jobs.append((entry, model_input, content_hash))

        total = len(jobs)
        done = 0
        if progress:
            progress(0, total, "")
        for batch in _job_batches(jobs):
            if should_cancel and should_cancel():
                break
            try:
                vectors = self._backend.encode([job[1] for job in batch])
            except EmbeddingUnavailableError:
                raise
            except Exception:
                vectors = np.empty((0, self.config.dimensions), dtype=np.float32)

            new_records: list[EntryEmbeddingRecord] = []
            if len(vectors) == len(batch):
                for (entry, _model_input, content_hash), vector in zip(batch, vectors, strict=True):
                    vectors_by_id[entry.id] = vector
                    new_records.append(self._record(entry.id, TASK_TAGGING, vector, content_hash))
            else:
                for entry, model_input, content_hash in batch:
                    try:
                        vector = self._backend.encode([model_input])[0]
                    except EmbeddingUnavailableError:
                        raise
                    except Exception as e:
                        logger.warning(
                            "[Embeddings] Could not embed entry for tagging",
                            entry_id=entry.id,
                            error=e,
                        )
                        continue
                    vectors_by_id[entry.id] = vector
                    new_records.append(self._record(entry.id, TASK_TAGGING, vector, content_hash))

            if new_records:
                self.library.save_entry_embeddings(new_records)
            done += len(batch)
            if progress:
                progress(done, total, batch[-1][0].path.name)

        return vectors_by_id

    def suggest_tags(
        self,
        entry_ids: Sequence[int],
        *,
        threshold: float = 0.6,
        max_per_entry: int = 5,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> list[AutoTagSuggestion]:
        """Suggest tags for one or more entries using semantic similarity."""
        if not entry_ids:
            return []

        entries = self.library.get_entries_for_embedding(entry_ids)
        if not entries:
            return []

        tags = self._candidate_tags()
        if not tags:
            return []

        _tags, tag_vectors = self._tag_embeddings(tags)
        entry_vectors = self._ensure_tagging_embeddings(entries, progress, should_cancel)

        suggestions: list[AutoTagSuggestion] = []
        for entry in entries:
            if should_cancel and should_cancel():
                break
            entry_vector = entry_vectors.get(entry.id)
            if entry_vector is None:
                continue

            scores = tag_vectors @ entry_vector
            existing_tag_ids = {tag.id for tag in entry.tags}
            added = 0
            for index in np.argsort(-scores):
                score = float(scores[int(index)])
                if score < threshold or added >= max_per_entry:
                    break
                tag = tags[int(index)]
                if tag.id in existing_tag_ids:
                    continue
                suggestions.append(
                    AutoTagSuggestion(
                        entry_id=entry.id,
                        entry_name=entry.path.name,
                        tag_id=int(tag.id),
                        tag_name=tag.name,
                        score=score,
                    )
                )
                added += 1

        suggestions.sort(key=lambda item: (item.entry_name.lower(), -item.score))
        return suggestions

    def _record(
        self,
        entry_id: int,
        task: str,
        vector: np.ndarray,
        content_hash: str,
    ) -> EntryEmbeddingRecord:
        return EntryEmbeddingRecord(
            entry_id=entry_id,
            model=self.config.model,
            task=task,
            dimensions=self.config.dimensions,
            vector=np.asarray(vector, dtype=np.float32).tobytes(),
            content_hash=content_hash,
        )
