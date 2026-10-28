# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Optional semantic search and auto-tagging powered by EmbeddingGemma 2.

The heavy model dependencies are imported lazily so that TagStudio can run without them.
Install the optional dependencies with::

    pip install "tagstudio[embeddings]"
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import structlog

from tagstudio.core.constants import RESERVED_TAG_END, TS_FOLDER_NAME
from tagstudio.core.library.alchemy.models import Entry, Tag

if TYPE_CHECKING:
    from tagstudio.core.library.alchemy.library import Library
    from tagstudio.core.library.alchemy.tag_verifier import TagVerifier


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

# Video containers a preview frame can be decoded from (OpenCV with FFmpeg).
VIDEO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".avi",
        ".flv",
        ".gifv",
        ".hevc",
        ".m4p",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".webm",
        ".wmv",
    }
)

# Extracted video frames are cached inside the library so they are only decoded once.
FRAME_CACHE_FOLDER_NAME = "embedding_frames"
FRAME_LONGEST_SIDE = 512

# Filename words that describe the file itself rather than what it shows. Camera and
# phone exporters generate names built only from these plus digits, which say nothing
# about the content and make every such file look identical to the model.
_NOISE_WORDS = frozenset(
    {
        "clip",
        "copy",
        "dji",
        "dsc",
        "dscf",
        "dscn",
        "dscq",
        "edit",
        "export",
        "file",
        "final",
        "frame",
        "gop",
        "gopr",
        "gx",
        "image",
        "img",
        "movie",
        "mp4",
        "mvi",
        "new",
        "photo",
        "pic",
        "picture",
        "pxl",
        "recording",
        "render",
        "screen",
        "screenshot",
        "shot",
        "untitled",
        "vid",
        "video",
        "videos",
        "whatsapp",
        # Glue words that never describe content on their own. "WhatsApp Video 2024-12-12
        # at 18.35.92" is only noise plus "at", and must not count as a description.
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)

# Tokens that describe how a file was captured rather than what it shows:
# "VRChat_DLSS_2026-09-04_17-37-48.282_1280x720" is a game screenshot, and none of
# the pipeline flags, the clock time or the resolution say anything about the frame.
_CAPTURE_WORDS = frozenset({"dlss", "fhd", "fps", "fsr", "hdr", "qhd", "taa", "uhd", "xess"})
_RESOLUTION_RE = re.compile(r"^\d{2,5}[x×]\d{2,5}$")
_RESOLUTION_SUFFIX_RE = re.compile(r"^\d{3,4}[pk]$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Hubness correction. A tag that is close to *everything* (NSFW tag names, very broad
# categories) scores above the threshold for every file while saying nothing about any
# of them. Measuring how close each tag generally is to the library's files lets that
# popularity be subtracted: only tags this file is more similar to than usual survive.
HUB_SAMPLE_LIMIT = 512
HUB_NEIGHBOURS = 20
HUB_MIN_SAMPLE = 8

EntryEmbeddingInput = str | dict[str, Any]
EntryJob = tuple[Entry, EntryEmbeddingInput, str]
ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]
StageCallback = Callable[[str], None]


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


@dataclass(frozen=True)
class TagVerificationNote:
    """What the vision model was asked about one file, and what it answered.

    Kept for every file the check looked at so the auto-tag modal can show the exact
    reply. ``kept`` is ``None`` when the file could not be checked, which is also
    when ``error`` says why.
    """

    entry_name: str
    labels: tuple[str, ...]
    reply: str = ""
    error: str = ""
    kept: int | None = None


@dataclass(frozen=True)
class AutoTagResult:
    """Suggestions for a run, plus the ``(entry_id, name)`` files left without any.

    A file ends up in ``unsuggested`` when nothing readable could be embedded or
    when no tag stood out for it, so the UI can say so instead of silently
    dropping it from the list.

    The verification counts describe the optional vision-language check: ``verified``
    files had their suggestions filtered by the model, ``unverified`` files were left
    alone (no image, no endpoint, or an answer that could not be read), and
    ``dropped`` counts the suggestions the model rejected.
    """

    suggestions: list[AutoTagSuggestion] = field(default_factory=list)
    unsuggested: list[tuple[int, str]] = field(default_factory=list)
    verified: int = 0
    unverified: int = 0
    dropped: int = 0
    verify_model: str = ""
    verify_error: str = ""
    verify_notes: list[TagVerificationNote] = field(default_factory=list)


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


def describe_title(title: str) -> str:
    """Remove capture noise from a file title, keeping any describing words.

    A lone number is usually part of the name (``boeing 747``, ``2001 a space
    odyssey``) and is kept; a run of them is a capture timestamp and says nothing.
    """
    words = title.split()
    numeric = [word for word in words if _NUMBER_RE.fullmatch(word)]
    keep_numbers = len(numeric) <= 1

    kept: list[str] = []
    for word in words:
        if (
            _RESOLUTION_RE.fullmatch(word)
            or _RESOLUTION_SUFFIX_RE.fullmatch(word)
            or word.lower() in _CAPTURE_WORDS
        ):
            continue
        if not keep_numbers and _NUMBER_RE.fullmatch(word):
            continue
        kept.append(word)
    return " ".join(kept)


def title_is_informative(title: str) -> bool:
    """Return whether a filename says anything about the file's content.

    ``PXL_20241212_183835927``, ``IMG_1234`` and ``DSC_0001`` are camera-generated
    and carry no description. Embedding them as if they were content makes every
    such file look the same to the model, which is why a single set of unrelated
    tags kept being suggested for whole batches of files.
    """
    words = [
        word
        for word in re.findall(r"[a-z]+", describe_title(title).lower())
        if word not in _NOISE_WORDS and word not in _CAPTURE_WORDS
    ]
    return any(len(word) > 1 for word in words)


def entry_kind_label(path: Path) -> str:
    """Describe a file by type, used when its name carries no description."""
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "file"


def entry_title(entry: Entry) -> str:
    title = describe_title(normalize_terms(entry.path.stem))
    if title_is_informative(title):
        return title
    return entry_kind_label(entry.path)


def entry_content(entry: Entry, exclude_tag_ids: Collection[int] = ()) -> str:
    """Build the descriptive text used to embed an entry.

    ``exclude_tag_ids`` hides tags from the embedded text. Evaluation uses it to
    measure how well an entry is scored without the tag it should recover.
    """
    parts: list[str] = []

    parent = entry.path.parent
    if parent.parts and str(parent) not in ("", "."):
        parts.append(f"Folder: {normalize_terms(parent.as_posix())}")

    excluded = set(exclude_tag_ids)
    tag_names = sorted(tag.name for tag in entry.tags if tag.name and tag.id not in excluded)
    if tag_names:
        parts.append(f"Tags: {', '.join(tag_names)}")

    for text_field in entry.text_fields:
        if text_field.value and text_field.value.strip():
            parts.append(f"{text_field.name}: {text_field.value.strip()}")

    text = "\n".join(parts)
    return text[:MAX_ENTRY_TEXT_LENGTH]


def entry_search_text(entry: Entry) -> str:
    text = entry_content(entry)
    return f"title: {entry_title(entry)} | text: {text or entry_title(entry)}"


def entry_tagging_text(entry: Entry, exclude_tag_ids: Collection[int] = ()) -> str:
    content = f"File: {entry_title(entry)}"
    extra = entry_content(entry, exclude_tag_ids)
    if extra:
        content = f"{content}\n{extra}"
    return CLASSIFICATION_PREFIX + content[:MAX_ENTRY_TEXT_LENGTH]


def tag_label(tag: Tag) -> str:
    """Build the classification label text for a tag.

    Parent tags are deliberately left out: siblings such as ``Indoors`` and
    ``Outdoors`` under the same category would otherwise share almost all of
    their label text and become impossible to tell apart. Hierarchy is used when
    selecting suggestions instead, see :func:`tag_facets`.
    """
    aliases = sorted({alias.name for alias in tag.aliases if alias.name})

    label = tag.name
    if aliases:
        label += f" (also known as: {', '.join(aliases)})"
    return CLASSIFICATION_PREFIX + label


def tag_facets(tags: Iterable[Tag]) -> dict[int, frozenset[int]]:
    """Map each tag id to the ids of its parent tags.

    Tags sharing a parent describe the same facet of an entry (``Indoors`` and
    ``Outdoors`` under ``Setting``, for example), so they compete for one slot.
    Tags without parents belong to no facet and never compete.
    """
    facets: dict[int, frozenset[int]] = {}
    for tag in tags:
        if tag.id is None:
            continue
        facets[int(tag.id)] = frozenset(
            parent.id for parent in tag.parent_tags if parent.id is not None
        )
    return facets


def tag_hubness(tag_vectors: np.ndarray, sample: np.ndarray) -> np.ndarray | None:
    """Return how similar each candidate tag typically is to the library's files.

    This is the cross-domain term of CSLS (cross-domain similarity local scaling),
    the standard remedy for hubness in image-to-text retrieval. A tag that sits in a
    dense region of the embedding space — a long alias list, a very broad concept —
    is close to almost every file, and subtracting its own typical similarity stops
    it from winning every comparison.
    """
    if sample.ndim != 2 or sample.shape[0] < HUB_MIN_SAMPLE:
        return None
    if sample.shape[0] > HUB_SAMPLE_LIMIT:
        step = max(1, sample.shape[0] // HUB_SAMPLE_LIMIT)
        sample = sample[::step][:HUB_SAMPLE_LIMIT]

    similarities = sample @ tag_vectors.T
    neighbours = min(HUB_NEIGHBOURS, similarities.shape[0])
    return np.sort(similarities, axis=0)[-neighbours:].mean(axis=0)


def select_tag_suggestions(
    scores: np.ndarray,
    tags: Sequence[Tag],
    *,
    threshold: float,
    max_per_entry: int,
    existing_tag_ids: Collection[int] = (),
    facets: Mapping[int, frozenset[int]] | None = None,
    one_per_facet: bool = True,
    raw_scores: np.ndarray | None = None,
    min_margin: float | None = None,
) -> list[tuple[Tag, float]]:
    """Pick the best-scoring tags for a single entry.

    ``scores`` holds one value per tag, parallel to ``tags``, and decides the order.
    When ``one_per_facet`` is set, at most one tag is kept per parent tag, and facets
    already filled by the entry's existing tags are skipped entirely. This keeps
    mutually exclusive siblings from being suggested together.

    ``raw_scores`` are the plain similarities used for the ``threshold`` check and for
    the reported score, so ``scores`` can hold a hubness-corrected ranking without
    changing what the numbers in the UI mean. ``min_margin`` additionally drops tags
    whose ranking value falls below it, which is how hub tags are filtered out.
    """
    if len(scores) != len(tags):
        raise ValueError("Expected one score per candidate tag.")
    display = scores if raw_scores is None else raw_scores
    if len(display) != len(tags):
        raise ValueError("Expected one raw score per candidate tag.")

    existing = {int(tag_id) for tag_id in existing_tag_ids}
    facets = facets if facets is not None else {}

    claimed: set[int] = set()
    if one_per_facet:
        for tag_id in existing:
            claimed |= facets.get(tag_id, frozenset())

    selected: list[tuple[Tag, float]] = []
    for index in np.argsort(-scores):
        position = int(index)
        if len(selected) >= max_per_entry:
            break
        if min_margin is not None and float(scores[position]) < min_margin:
            break
        if float(display[position]) < threshold:
            continue

        tag = tags[position]
        if tag.id is None or int(tag.id) in existing:
            continue

        if one_per_facet:
            tag_facet_ids = facets.get(int(tag.id), frozenset())
            if tag_facet_ids & claimed:
                continue
            claimed |= tag_facet_ids

        selected.append((tag, float(display[position])))

    return selected


def frame_cache_dir(library_dir: Path | None) -> Path | None:
    """Return the folder where extracted video frames are cached."""
    if library_dir is None:
        return None
    return Path(library_dir) / TS_FOLDER_NAME / FRAME_CACHE_FOLDER_NAME


def _extract_video_frame(path: Path, cache_dir: Path) -> Path | None:
    """Decode and cache one representative frame from a video."""
    try:
        stat = path.stat()
    except OSError:
        return None

    digest = hashlib.sha1(
        f"{path.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    target = cache_dir / f"{digest}.png"
    if target.is_file():
        return target

    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.info("[Embeddings] OpenCV is unavailable, using text only for videos")
        return None

    capture = None
    frame = None
    try:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames and total_frames > 0:
            # Seek to the middle; that frame best represents the whole clip.
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames // 2))
            read_ok, frame = capture.read()
            if not read_ok or frame is None:
                frame = None
                for index in range(min(10, int(total_frames))):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    read_ok, frame = capture.read()
                    if read_ok and frame is not None:
                        break
        else:
            read_ok, frame = capture.read()
            if not read_ok:
                frame = None

        if frame is None:
            return None

        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > FRAME_LONGEST_SIDE:
            scale = FRAME_LONGEST_SIDE / longest
            frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))

        encoded_ok, buffer = cv2.imencode(".png", frame)
        if not encoded_ok:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(buffer.tobytes())
    except Exception as e:
        logger.warning("[Embeddings] Could not extract a video frame", path=path, error=e)
        return None
    finally:
        if capture is not None:
            capture.release()

    return target


def media_path_for_entry(
    entry: Entry, library_dir: Path | None, cache_dir: Path | None = None
) -> Path | None:
    """Return an image file the vision encoder can read for this entry.

    Images are used as-is. Videos previously had no visual input at all, so a clip
    was embedded from its filename alone and looked identical to every other clip;
    its middle frame is extracted and cached instead.
    """
    if library_dir is None:
        return None
    path = Path(library_dir) / entry.path
    if not path.is_file():
        return None

    ext = entry.path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return path
    if ext in VIDEO_EXTENSIONS:
        if cache_dir is None:
            return None
        return _extract_video_frame(path, cache_dir)
    return None


def build_entry_input(
    entry: Entry,
    task: str,
    library_dir: Path | None,
    use_images: bool,
    exclude_tag_ids: Collection[int] = (),
    cache_dir: Path | None = None,
) -> tuple[EntryEmbeddingInput, str]:
    """Return the model input and a content hash for an entry."""
    text = (
        entry_search_text(entry)
        if task == TASK_SEARCH
        else entry_tagging_text(entry, exclude_tag_ids)
    )
    image_path = media_path_for_entry(entry, library_dir, cache_dir) if use_images else None

    image_fingerprint = ""
    if image_path is not None:
        # Fingerprint the source file, not a cached frame, so regenerating frames
        # never looks like the entry itself changed.
        try:
            source = Path(library_dir) / entry.path if library_dir is not None else image_path
            stat = source.stat()
            image_fingerprint = f"{source.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            image_fingerprint = image_path.as_posix()
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
                    entry,
                    TASK_SEARCH,
                    self.library.library_dir,
                    self.config.use_images,
                    cache_dir=self._frame_cache_dir(),
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

    def candidate_tags(self, all_tags: Sequence[Tag] | None = None) -> list[Tag]:
        """Return the tags that auto-tagging may suggest.

        Categories, hidden tags and reserved tags are excluded. Passing
        ``all_tags`` reuses an already loaded tag list.
        """
        tags = []
        for tag in self.library.tags if all_tags is None else all_tags:
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

    def _frame_cache_dir(self) -> Path | None:
        return frame_cache_dir(self.library.library_dir)

    def _hubness(self, tag_vectors: np.ndarray) -> np.ndarray | None:
        """How close each tag generally is to this library's files."""
        _ids, sample = self.library.load_entry_embedding_vectors(
            self.config.model,
            TASK_TAGGING,
            self.config.dimensions,
            include_hidden=True,
        )
        return tag_hubness(tag_vectors, sample)

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
                    entry,
                    TASK_TAGGING,
                    self.library.library_dir,
                    self.config.use_images,
                    cache_dir=self._frame_cache_dir(),
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
        one_per_facet: bool = True,
        use_hubness: bool = True,
        verifier: TagVerifier | None = None,
        stage: StageCallback | None = None,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCallback | None = None,
    ) -> AutoTagResult:
        """Suggest tags for one or more entries using semantic similarity.

        Files with nothing to look at — an unreadable video with a camera-generated
        name — are reported in :attr:`AutoTagResult.unsuggested` instead of being
        given tags chosen from noise.

        When a ``verifier`` is given, its model is shown each file's image together
        with the suggestions that survived ranking, and may reject any of them. A
        file the verifier could not check keeps every suggestion: a failed request
        must never look like a rejection.
        """
        if not entry_ids:
            return AutoTagResult()

        entries = self.library.get_entries_for_embedding(entry_ids)
        if not entries:
            return AutoTagResult()

        all_tags = self.library.tags
        tags = self.candidate_tags(all_tags)
        if not tags:
            return AutoTagResult()

        facets = tag_facets(all_tags)
        _tags, tag_vectors = self._tag_embeddings(tags)
        entry_vectors = self._ensure_tagging_embeddings(entries, progress, should_cancel)
        hubness = self._hubness(tag_vectors) if use_hubness else None

        suggestions: list[AutoTagSuggestion] = []
        unsuggested: list[tuple[int, str]] = []
        verified = 0
        unverified = 0
        dropped = 0
        verify_error = ""
        verify_notes: list[TagVerificationNote] = []
        verify_model = verifier.config.model if verifier is not None else ""
        verifying = verifier is not None and verifier.config.is_configured
        if verifying and stage is not None:
            # Everything is embedded by now, so tell the UI why the next calls are slow.
            stage("verifying")

        for entry in entries:
            if should_cancel and should_cancel():
                break
            entry_vector = entry_vectors.get(entry.id)
            if entry_vector is None:
                unsuggested.append((entry.id, entry.path.name))
                continue

            raw = tag_vectors @ entry_vector
            if hubness is not None:
                # Rank by how much more similar this file is to a tag than the
                # library's files usually are. Hub tags keep their high raw score,
                # so they can still clear the threshold, but they no longer rank.
                scores = raw - hubness
                min_margin: float | None = 0.0
            else:
                scores = raw
                min_margin = None

            selected = select_tag_suggestions(
                scores,
                tags,
                threshold=threshold,
                max_per_entry=max_per_entry,
                existing_tag_ids=[tag.id for tag in entry.tags if tag.id is not None],
                facets=facets,
                one_per_facet=one_per_facet,
                raw_scores=raw,
                min_margin=min_margin,
            )
            if not selected:
                unsuggested.append((entry.id, entry.path.name))
                continue

            if verifying and verifier is not None:
                image_path = media_path_for_entry(
                    entry, self.library.library_dir, self._frame_cache_dir()
                )
                labels = tuple(tag.name for tag, _score in selected)
                if image_path is None:
                    unverified += 1
                    verify_notes.append(
                        TagVerificationNote(
                            entry_name=entry.path.name,
                            labels=labels,
                            error="No image could be read out of this file.",
                        )
                    )
                else:
                    outcome = verifier.verify(image_path, labels)
                    verify_notes.append(
                        TagVerificationNote(
                            entry_name=entry.path.name,
                            labels=labels,
                            reply=outcome.reply,
                            error=outcome.error,
                            kept=None if outcome.kept is None else len(outcome.kept),
                        )
                    )
                    if outcome.kept is None:
                        unverified += 1
                        verify_error = verify_error or outcome.error
                    else:
                        verified += 1
                        kept = [
                            pair for index, pair in enumerate(selected) if index in outcome.kept
                        ]
                        dropped += len(selected) - len(kept)
                        selected = kept

            if not selected:
                unsuggested.append((entry.id, entry.path.name))
                continue

            for tag, score in selected:
                suggestions.append(
                    AutoTagSuggestion(
                        entry_id=entry.id,
                        entry_name=entry.path.name,
                        tag_id=int(tag.id),
                        tag_name=tag.name,
                        score=score,
                    )
                )

        suggestions.sort(key=lambda item: (item.entry_name.lower(), -item.score))
        return AutoTagResult(
            suggestions=suggestions,
            unsuggested=unsuggested,
            verified=verified,
            unverified=unverified,
            dropped=dropped,
            verify_model=verify_model if verified or unverified else "",
            verify_error=verify_error,
            verify_notes=verify_notes,
        )

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
