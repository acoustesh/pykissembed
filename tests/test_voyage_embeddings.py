"""Focused tests for the direct Voyage embeddings REST transport."""

from __future__ import annotations

from typing import cast

import pytest
import requests

from pykissembed.similarity import embeddings
from pykissembed.similarity.constants import VOYAGE_CODE_MODEL


class _Response:
    """Minimal requests response double used by the Voyage transport tests."""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        response = requests.Response()
        response.status_code = self.status_code
        raise requests.exceptions.HTTPError(response=response)

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _NoopEncoding:
    """Token encoder double that leaves every short test input unchanged."""

    @staticmethod
    def encode(_text: str) -> list[int]:
        return []

    @staticmethod
    def decode(_tokens: list[int]) -> str:
        return ""


@pytest.fixture(autouse=True)
def _voyage_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a non-secret Voyage key for request construction."""
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-test-key")
    monkeypatch.setattr(embeddings, "_get_tiktoken_encoding", _NoopEncoding)


def _success_payload(texts: list[str]) -> dict[str, object]:
    """Build a valid response in reverse order to exercise index restoration.

    Returns
    -------
    dict[str, object]
        Voyage-style JSON response payload.
    """
    return {
        "data": [
            {"index": index, "embedding": [index, float(len(text))]}
            for index, text in reversed(list(enumerate(texts)))
        ],
    }


def test_voyage_request_uses_official_rest_contract_and_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The REST request uses Voyage's endpoint and restores index order."""
    calls: list[dict[str, object]] = []

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response(_success_payload(["first", "second"]))

    monkeypatch.setattr(requests, "post", fake_post)
    make_request, _is_retryable = embeddings._build_provider_caller(  # ruff:ignore[private-member-access]
        "voyage",
        VOYAGE_CODE_MODEL,
        17.5,
    )

    result = make_request(["first", "second"])

    assert result == [[0.0, 5.0], [1.0, 6.0]]
    assert calls == [
        {
            "url": "https://api.voyageai.com/v1/embeddings",
            "headers": {
                "Authorization": "Bearer voyage-test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "input": ["first", "second"],
                "model": "voyage-code-3",
                "input_type": "document",
            },
            "timeout": 17.5,
        },
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "non-object response"),
        ({}, "does not contain a data array"),
        ({"data": "not-a-list"}, "does not contain a data array"),
        ({"data": []}, "0 embeddings for 1 inputs"),
        ({"data": ["not-an-object"]}, "non-object embedding item"),
        ({"data": [{"embedding": [1.0]}]}, "invalid embedding index"),
        ({"data": [{"index": True, "embedding": [1.0]}]}, "invalid embedding index"),
        ({"data": [{"index": 1, "embedding": [1.0]}]}, "invalid embedding index"),
        ({"data": [{"index": 0, "embedding": []}]}, "invalid embedding"),
        ({"data": [{"index": 0, "embedding": (1.0,)}]}, "invalid embedding"),
        ({"data": [{"index": 0, "embedding": [True]}]}, "non-numeric embedding"),
        ({"data": [{"index": 0, "embedding": ["1.0"]}]}, "non-numeric embedding"),
        ({"data": [{"index": 0, "embedding": [float("nan")]}]}, "non-numeric embedding"),
        ({"data": [{"index": 0, "embedding": [float("inf")]}]}, "non-numeric embedding"),
    ],
)
def test_voyage_response_rejects_malformed_payloads(payload: object, message: str) -> None:
    """Malformed envelopes, indices, and vectors fail before reaching the cache."""
    with pytest.raises(ValueError, match=message):
        embeddings._parse_voyage_response(payload, 1)  # ruff:ignore[private-member-access]


def test_voyage_response_rejects_duplicate_indices() -> None:
    """A complete-looking response cannot overwrite a duplicate index."""
    payload = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 0, "embedding": [2.0]},
        ],
    }

    with pytest.raises(ValueError, match="duplicate embedding indices"):
        embeddings._parse_voyage_response(payload, 2)  # ruff:ignore[private-member-access]


def test_voyage_response_rejects_inconsistent_vector_dimensions() -> None:
    """A batch cannot mix embedding dimensions before cache insertion."""
    payload = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0, 3.0]},
        ],
    }

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        embeddings._parse_voyage_response(payload, 2)  # ruff:ignore[private-member-access]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_voyage_retries_rate_limits_and_server_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    """HTTP 429 and 5xx failures use the existing exponential retry loop."""
    responses = iter([_Response({}, status_code), _Response(_success_payload(["text"]))])
    sleeps: list[float] = []

    def fake_post(*args: object, **kwargs: object) -> _Response:
        del args, kwargs
        return next(responses)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)

    result = embeddings.get_embeddings_batch(
        ["text"],
        provider="voyage",
        max_retries=2,
    )

    assert result == [[0.0, 4.0]]
    assert sleeps == [1.0]


def test_voyage_retries_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request timeout is retried without changing the result order."""
    calls = 0
    sleeps: list[float] = []

    def fake_post(*args: object, **kwargs: object) -> _Response:
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            msg = "timed out"
            raise requests.exceptions.Timeout(msg)
        return _Response(_success_payload(["text"]))

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)

    assert embeddings.get_embeddings_batch(
        ["text"],
        provider="voyage",
        max_retries=2,
    ) == [[0.0, 4.0]]
    assert calls == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "failure",
    [_Response({}, 401), _Response(ValueError("invalid JSON"))],
)
def test_voyage_does_not_retry_terminal_or_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
    failure: _Response,
) -> None:
    """Authentication and malformed-body failures remain terminal."""
    calls = 0

    def fake_post(*args: object, **kwargs: object) -> _Response:
        nonlocal calls
        del args, kwargs
        calls += 1
        return failure

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        embeddings.time,
        "sleep",
        lambda _delay: pytest.fail("terminal Voyage failure was retried"),
    )

    with pytest.raises((requests.exceptions.HTTPError, ValueError)):
        embeddings.get_embeddings_batch(["text"], provider="voyage", max_retries=3)
    assert calls == 1


def test_voyage_preserves_existing_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voyage requests retain the existing maximum batch size of 128 texts."""
    batch_sizes: list[int] = []

    def fake_post(
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _Response:
        del headers, timeout
        texts = json["input"]
        assert isinstance(texts, list)
        batch_sizes.append(len(texts))
        return _Response(_success_payload(cast("list[str]", texts)))

    monkeypatch.setattr(requests, "post", fake_post)
    texts = [f"text-{index}" for index in range(129)]

    result = embeddings.get_embeddings_batch(
        texts,
        provider="voyage",
        max_retries=1,
    )

    assert batch_sizes == [128, 1]
    assert len(result) == len(texts)
    assert all(isinstance(component, float) for vector in result for component in vector)
