# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only


"""Optional vision-language check of auto-tag suggestions.

Suggestions come from comparing a file's embedding with the embedding of a tag name.
That is cheap enough to run over a whole library, but it cannot separate labels that
sit close together: a cat and a dog are both "animal on grass", and a night-time
outdoor shot is as close to "Indoors" as it is to "Outdoors".

This module asks a vision-language model a much easier question — given the image,
which of these eight labels actually apply? — and keeps only the labels it accepts.
Nothing is downloaded or graded here: the model is reached over an OpenAI-compatible
HTTP API, so a local server (LM Studio, Ollama, ``llama-server``, vLLM) and a hosted
one are configured the same way.

Small vision models rarely answer in exactly the format they are asked for, and
reasoning models spend the start of their reply thinking. Both are handled rather
than treated as failures: numbers, tag names, JSON and ``none`` are all understood,
the token budget leaves room for a chain of thought, and whatever the model actually
said is kept in :attr:`VerificationOutcome.reply` so the user can see it.
"""

from __future__ import annotations

import base64
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

DEFAULT_VERIFY_TIMEOUT = 60.0

# Images are shrunk before sending: a 4K photo base64-encoded is several megabytes,
# which local servers reject or handle very slowly, and the answer only depends on
# what is in the frame.
VERIFY_IMAGE_LONGEST_SIDE = 768
VERIFY_IMAGE_QUALITY = 85

# Reasoning models think before answering, and the thinking is billed against
# max_tokens. A small budget cuts them off mid-thought, which leaves an empty answer
# that is indistinguishable from a broken one.
MAX_VERIFY_TOKENS = 4096
MAX_ERROR_LENGTH = 300
MAX_REPLY_LENGTH = 2000

_SYSTEM_PROMPT = (
    "You check whether tags from a photo library really apply to an image. "
    "Answer briefly, with the tag numbers only."
)

_BRACKETED_LIST = re.compile(r"\[[^\[\]]*\]")
_CODE_FENCE = re.compile(r"```[a-zA-Z]*")
# A reply made only of numbers and separators ("1, 3"): unambiguous, so it is read as
# a selection. Anything containing words goes through tag-name matching instead.
_NUMERIC_ONLY = re.compile(r"^[\s\d,;.|/\-():]*\d[\s\d,;.|/\-():]*$")
# Models like to open with a label: "Tags: 1, 3".
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:tags?|numbers?|selected|selection|keep|answer)\s*[:\-]\s*", re.IGNORECASE
)
_NONE_ANSWER = re.compile(r"\b(none|no tags?|nothing|n/?a)\b", re.IGNORECASE)
# Tag names are matched chunk by chunk, so "Cat" cannot be found inside "Cat ears"
# and a sentence that merely mentions a tag does not select it.
_CHUNK_SEPARATORS = re.compile(r"[,\n;:|•·\t]+")
_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*[.):\-]\s*")
_EDGE_PUNCTUATION = "\"'`*•-–—[](){}."


@dataclass(frozen=True)
class VerificationConfig:
    """Where to send verification requests, and whether to send them at all."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: float = DEFAULT_VERIFY_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """Whether both an endpoint and a model name have been given."""
        return bool(self.base_url.strip() and self.model.strip())


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of asking the model about one file.

    ``kept`` holds indexes into the candidate list that was sent. ``None`` means the
    file could not be checked at all — no image, an unreachable endpoint, an answer
    that could not be read — and the caller should keep every suggestion rather than
    treating a failure as a rejection.

    ``reply`` holds what the model actually said, trimmed for display, so a strange
    result can be diagnosed instead of guessed at.
    """

    kept: frozenset[int] | None = None
    error: str = ""
    reply: str = ""


def chat_completions_url(base_url: str) -> str:
    """Return the chat-completions endpoint for a base URL.

    Accepts what people actually paste: ``http://localhost:1234``,
    ``http://localhost:1234/v1``, or the full ``.../v1/chat/completions``.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if re.search(r"/v\d+$", base):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def encode_image_data_url(path: Path) -> str:
    """Return a ``data:`` URL holding the file as a downscaled JPEG."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((VERIFY_IMAGE_LONGEST_SIDE, VERIFY_IMAGE_LONGEST_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=VERIFY_IMAGE_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def verification_prompt(labels: Sequence[str]) -> str:
    """Build the question sent alongside the image."""
    numbered = "\n".join(f"{index}. {label}" for index, label in enumerate(labels, start=1))
    return (
        f"Candidate tags:\n{numbered}\n\n"
        "Look at the image and decide which candidate tags clearly apply to it. "
        "Leave out any tag that only loosely fits, and do not invent tags that are "
        "not on the list. Answer with the numbers only, separated by commas, for "
        "example: 1, 3. If none of the candidates apply, answer: none"
    )


def _normalise(value: str) -> str:
    """Fold a label or a reply fragment into a comparable form."""
    text = value.strip().strip(_EDGE_PUNCTUATION)
    text = _NUMBER_PREFIX.sub("", text)
    text = text.strip().strip(_EDGE_PUNCTUATION)
    return " ".join(text.casefold().split())


def _indexes_from_numbers(text: str, count: int) -> frozenset[int]:
    return frozenset(
        int(number) - 1 for number in re.findall(r"\d+", text) if 1 <= int(number) <= count
    )


def _match_names(text: str, labels: Sequence[str]) -> frozenset[int]:
    """Return the candidate indexes whose names appear as standalone fragments."""
    by_name: dict[str, int] = {}
    for index, label in enumerate(labels):
        key = _normalise(label)
        if key:
            by_name.setdefault(key, index)

    matched = set()
    for chunk in _CHUNK_SEPARATORS.split(text):
        index = by_name.get(_normalise(chunk))
        if index is not None:
            matched.add(index)
    return frozenset(matched)


def _parse_bracketed(text: str, labels: Sequence[str]) -> frozenset[int] | None:
    """Read a JSON array of numbers or of names, if there is one."""
    found = _BRACKETED_LIST.search(text)
    if found is None:
        return None
    try:
        parsed: Any = json.loads(found.group(0))
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    if not parsed:
        return frozenset()

    numbers = [item for item in parsed if isinstance(item, int) and not isinstance(item, bool)]
    names = [item for item in parsed if isinstance(item, str)]
    if not numbers and not names:
        return None

    kept = _indexes_from_numbers(" ".join(str(number) for number in numbers), len(labels))
    for name in names:
        kept |= _match_names(name, labels)
    # A list that matched nothing is a reply we did not understand, not a rejection.
    return kept or None


def parse_selection(text: str | None, labels: Sequence[str]) -> frozenset[int] | None:
    """Read the kept label indexes out of a model reply.

    Returns zero-based indexes, an empty set when the model rejected every candidate,
    or ``None`` when the reply could not be understood. Numbers outside the range that
    was sent are ignored, so a confused model cannot invent a tag.

    Small models answer in whatever shape they like, so four shapes are accepted:
    a JSON array of numbers or names, a bare list of numbers (``1, 3``), and the tag
    names themselves however they are separated (``Cat, Dog``). Only a reply that
    matches no candidate and does not say "none" is treated as unreadable.
    """
    if not text:
        return None

    selected = _parse_bracketed(text, labels)
    if selected is not None:
        return selected

    stripped = _ANSWER_PREFIX.sub("", _CODE_FENCE.sub(" ", text).strip())
    if _NUMERIC_ONLY.match(stripped):
        return _indexes_from_numbers(stripped, len(labels))

    matched = _match_names(stripped, labels)
    if matched:
        return matched
    if _NONE_ANSWER.search(stripped):
        return frozenset()
    return None


@dataclass(frozen=True)
class _Reply:
    """The assistant message, flattened for reading and for display."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""

    def preview(self) -> str:
        """Return what to show the user, trimmed to something a label can hold."""
        if self.content.strip():
            return self.content.strip()[:MAX_REPLY_LENGTH]
        if self.reasoning.strip():
            # An answer that never arrived: show the tail, which is where a truncated
            # chain of thought stops.
            tail = self.reasoning.strip()[-MAX_REPLY_LENGTH:]
            return f"[no answer, only reasoning]\n{tail}"
        return ""


def _read_reply(body: Any) -> _Reply | None:
    """Pull the assistant text and finish reason out of a chat-completions response."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = first.get("finish_reason")

    return _Reply(
        content=content if isinstance(content, str) else "",
        reasoning=reasoning if isinstance(reasoning, str) else "",
        finish_reason=finish_reason if isinstance(finish_reason, str) else "",
    )


def _unreadable_error(reply: _Reply) -> str:
    """Explain an unusable answer in terms of what the user can do about it."""
    if reply.finish_reason == "length":
        return (
            "The model ran out of tokens before it answered (finish_reason=length). "
            "A reasoning model needs a larger budget, or a model that does not think "
            "out loud."
        )
    if not reply.content.strip() and reply.reasoning.strip():
        return "The model produced reasoning but no answer, so there was nothing to read."
    if not reply.content.strip():
        return "The model returned an empty reply."
    return "The model answered with neither tag numbers nor tag names from the list."


class TagVerifier:
    """Asks a vision-language model which candidate tags apply to a file."""

    def __init__(
        self,
        config: VerificationConfig,
        *,
        post: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._post = post if post is not None else requests.post

    def verify(self, image_path: Path, labels: Sequence[str]) -> VerificationOutcome:
        """Return the labels the model accepted for this image."""
        if not labels:
            return VerificationOutcome(kept=frozenset())
        if not self.config.is_configured:
            return VerificationOutcome(kept=None, error="")

        try:
            data_url = encode_image_data_url(Path(image_path))
        except Exception as e:
            logger.warning("[TagVerifier] Could not read an image", path=image_path, error=e)
            return VerificationOutcome(kept=None, error=_describe(e))

        payload = {
            "model": self.config.model.strip(),
            "temperature": 0,
            "max_tokens": MAX_VERIFY_TOKENS,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": verification_prompt(labels)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }

        try:
            response = self._post(
                chat_completions_url(self.config.base_url),
                json=payload,
                headers=self._headers(),
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            logger.warning("[TagVerifier] Verification request failed", error=e)
            return VerificationOutcome(kept=None, error=_describe(e))

        reply = _read_reply(body)
        if reply is None:
            snippet = json.dumps(body)[:MAX_REPLY_LENGTH] if body is not None else ""
            return VerificationOutcome(
                kept=None,
                error="The server returned a response without an assistant message.",
                reply=snippet,
            )

        logger.debug(
            "[TagVerifier] Reply",
            model=self.config.model,
            finish_reason=reply.finish_reason,
            reply=reply.preview(),
        )
        kept = parse_selection(reply.content, labels)
        if kept is None:
            error = _unreadable_error(reply)
            logger.warning(
                "[TagVerifier] Could not read the model reply",
                model=self.config.model,
                finish_reason=reply.finish_reason,
                content=reply.content[:MAX_ERROR_LENGTH],
            )
            return VerificationOutcome(kept=None, error=error, reply=reply.preview())
        return VerificationOutcome(kept=kept, reply=reply.preview())

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.api_key.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers


def _describe(error: Exception) -> str:
    """Return a short, user-presentable description of a failure."""
    message = str(error).strip() or type(error).__name__
    return message[:MAX_ERROR_LENGTH]
