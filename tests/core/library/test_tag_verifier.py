# SPDX-FileCopyrightText: (c) TagStudio Contributors
# SPDX-License-Identifier: GPL-3.0-only

# pyright: reportPrivateUsage=false


from pathlib import Path

import pytest
from PIL import Image

from tagstudio.core.library.alchemy.tag_verifier import (
    MAX_VERIFY_TOKENS,
    TagVerifier,
    VerificationConfig,
    chat_completions_url,
    encode_image_data_url,
    parse_selection,
    verification_prompt,
)

LABELS = ["Cat", "Dog", "Chihuahua", "Outdoors"]


class FakeResponse:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = str(body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self) -> dict:
        return self._body


def _reply(content: str, *, reasoning: str = "", finish_reason: str = "stop") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "cat.png"
    Image.new("RGB", (1600, 900), (40, 90, 40)).save(path)
    return path


def test_chat_completions_url_accepts_what_people_paste():
    assert (
        chat_completions_url("http://localhost:1234") == "http://localhost:1234/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://localhost:1234/v1/")
        == "http://localhost:1234/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://localhost:8080/v1/chat/completions")
        == "http://localhost:8080/v1/chat/completions"
    )
    assert chat_completions_url("") == ""


@pytest.mark.parametrize(
    "reply",
    [
        '{"keep": [1, 3]}',
        "[1, 3]",
        '```json\n{"keep": [1, 3]}\n```',
        'Sure. {"keep": [1, 3]}',
        "1, 3",
        "1) 3",
        "Tags: 1, 3",
        "1. Cat\n3. Chihuahua",
    ],
)
def test_parse_selection_accepts_numbers(reply: str):
    assert parse_selection(reply, LABELS) == frozenset({0, 2})


@pytest.mark.parametrize(
    "reply",
    [
        "Cat, Chihuahua",
        "**Cat**, **Chihuahua**",
        "cat, CHIHUAHUA",
        '{"keep": ["Cat", "Chihuahua"]}',
        "Cat\nChihuahua",
    ],
)
def test_parse_selection_accepts_tag_names(reply: str):
    """Small vision models often answer with the names, not the numbers."""
    assert parse_selection(reply, LABELS) == frozenset({0, 2})


def test_parse_selection_ignores_names_that_were_not_offered():
    """A model that invents a tag must not have it counted, but real ones still are."""
    assert parse_selection("Animal Ears, Cat, Horse", LABELS) == frozenset({0})


def test_parse_selection_needs_whole_names():
    """\"Cat ears\" is a different tag, so it must not be read as \"Cat\"."""
    assert parse_selection("Cat ears", LABELS) is None


def test_parse_selection_reports_an_empty_selection():
    """Rejecting everything is an answer, not a failure."""
    assert parse_selection('{"keep": []}', LABELS) == frozenset()
    assert parse_selection("[]", LABELS) == frozenset()
    assert parse_selection("none", LABELS) == frozenset()
    assert parse_selection("None of the candidates apply.", LABELS) == frozenset()


def test_parse_selection_ignores_numbers_outside_the_sent_range():
    assert parse_selection('{"keep": [1, 9, 0]}', LABELS) == frozenset({0})


@pytest.mark.parametrize(
    "reply",
    [None, "", "I cannot see the image.", "The animal is sitting on grass.", '{"keep": 1}'],
)
def test_parse_selection_rejects_unreadable_replies(reply):
    """Anything unreadable must stay unverified so no suggestion is silently lost."""
    assert parse_selection(reply, LABELS) is None


def test_encode_image_data_url_downscales_before_sending(image_file: Path):
    data_url = encode_image_data_url(image_file)

    assert data_url.startswith("data:image/jpeg;base64,")
    # A 1600px image is sent small: large payloads are slow or rejected by servers.
    assert len(data_url) < 200_000


def test_verification_prompt_numbers_the_labels():
    prompt = verification_prompt(["Cat", "Dog", "Chihuahua"])

    assert "1. Cat" in prompt
    assert "3. Chihuahua" in prompt
    assert "do not invent tags" in prompt
    assert "1, 3" in prompt


def test_verify_returns_the_labels_the_model_kept(image_file: Path):
    requests_seen: list[tuple[str, dict, dict]] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        requests_seen.append((url, json, headers))
        return FakeResponse(_reply('{"keep": [2]}'))

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="lfm3-vl-2b"),
        post=fake_post,
    )
    outcome = verifier.verify(image_file, ["Cat", "Dog", "Chihuahua"])

    assert outcome.kept == frozenset({1})
    assert outcome.error == ""
    assert outcome.reply == '{"keep": [2]}'

    url, payload, headers = requests_seen[0]
    assert url == "http://localhost:1234/v1/chat/completions"
    assert payload["model"] == "lfm3-vl-2b"
    assert payload["temperature"] == 0
    # Reasoning models spend tokens thinking, so the budget has to fit both.
    assert payload["max_tokens"] == MAX_VERIFY_TOKENS
    assert payload["max_tokens"] >= 1024
    assert "Authorization" not in headers
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "1. Cat" in user_content[0]["text"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_verify_reads_a_reply_that_lists_tag_names(image_file: Path):
    """The format LM Studio users see: a comma-separated list of names."""
    body = _reply("Animal Ears, Blue Hair, Cat, Dog")

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="lfm3-vl-2b"),
        post=lambda *args, **kwargs: FakeResponse(body),
    )

    outcome = verifier.verify(image_file, ["Cat", "Dog", "Chihuahua"])

    assert outcome.kept == frozenset({0, 1})
    assert outcome.error == ""


def test_verify_reports_a_truncated_reply_along_with_its_text(image_file: Path):
    """A reasoning model can hit the token cap mid-thought; say so, and show it."""
    body = _reply("", reasoning="Let me look at the wooden deck and the", finish_reason="length")

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="lfm3-vl-2b"),
        post=lambda *args, **kwargs: FakeResponse(body),
    )

    outcome = verifier.verify(image_file, ["Cat", "Dog"])

    assert outcome.kept is None
    assert "finish_reason=length" in outcome.error
    assert "wooden deck" in outcome.reply


def test_verify_reports_an_answer_that_missing_after_reasoning(image_file: Path):
    body = _reply("", reasoning="The image shows two characters outdoors.")

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="m"),
        post=lambda *args, **kwargs: FakeResponse(body),
    )

    outcome = verifier.verify(image_file, ["Cat", "Dog"])

    assert outcome.kept is None
    assert "reasoning but no answer" in outcome.error
    assert "two characters outdoors" in outcome.reply


def test_verify_sends_the_api_key_when_given(image_file: Path):
    seen: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen.update(headers or {})
        return FakeResponse(_reply("1"))

    verifier = TagVerifier(
        VerificationConfig(base_url="http://host/v1", model="m", api_key="secret"),
        post=fake_post,
    )
    verifier.verify(image_file, ["Cat"])

    assert seen["Authorization"] == "Bearer secret"


def test_verify_leaves_suggestions_untouched_when_the_endpoint_fails(image_file: Path):
    def failing_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        raise ConnectionError("connection refused")

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="lfm3-vl-2b"),
        post=failing_post,
    )

    outcome = verifier.verify(image_file, ["Cat", "Dog"])

    assert outcome.kept is None
    assert "connection refused" in outcome.error


def test_verify_treats_an_http_error_as_unverified(image_file: Path):
    def error_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return FakeResponse({"error": "no such model"}, status_code=404)

    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="missing"),
        post=error_post,
    )

    assert verifier.verify(image_file, ["Cat"]).kept is None


def test_verify_keeps_the_body_when_there_is_no_message(image_file: Path):
    verifier = TagVerifier(
        VerificationConfig(base_url="http://localhost:1234/v1", model="m"),
        post=lambda *args, **kwargs: FakeResponse({"detail": "image too large"}),
    )

    outcome = verifier.verify(image_file, ["Cat"])

    assert outcome.kept is None
    assert "image too large" in outcome.reply


def test_verify_is_inert_without_an_endpoint(image_file: Path):
    def unexpected_post(*args, **kwargs):
        raise AssertionError("no request should be sent")

    verifier = TagVerifier(VerificationConfig(), post=unexpected_post)

    assert not verifier.config.is_configured
    assert verifier.verify(image_file, ["Cat"]).kept is None
