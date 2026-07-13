"""Embedding API clients and utilities for OpenAI, Codestral, Voyage, Gemini, and Qwen.

Ported from ``mega-scrapper/tests/similarity/embeddings.py``. The cloud
provider clients (OpenAI, Codestral, Voyage, Gemini, Qwen) are imported lazily
so the module is importable without cloud dependencies installed. The
local provider (sentence-transformers) is handled by
``pykissembed-local``.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, TypeGuard, cast

import numpy as np

from pykissembed.config import get_config

if TYPE_CHECKING:
    from collections.abc import Callable

from pykissembed.similarity.constants import (
    CODESTRAL_EMBED_MODEL,
    GEMINI_EMBED_MODEL,
    OPENROUTER_API_URL,
    QWEN_EMBED_MODEL,
    VOYAGE_CODE_MODEL,
)

# Exponential backoff delays
_RETRY_DELAYS = [1.0, 2.0, 4.0]

# Maximum token limits per provider
_OPENAI_MAX_TOKENS = 8191  # text-embedding-3-large limit is 8192
_VOYAGE_MAX_TOKENS = 31000  # voyage-code-3 limit is 32000
_CODESTRAL_MAX_TOKENS = 8000  # Conservative estimate
_GEMINI_MAX_TOKENS = 2000  # gemini-embedding-001 limit is 2048
_QWEN_MAX_TOKENS = 32000  # qwen3-embedding-8b limit

# Maximum batch sizes per provider (number of texts per request)
_OPENAI_MAX_BATCH_SIZE = 2048  # OpenAI allows large batches
_VOYAGE_MAX_BATCH_SIZE = 128  # Voyage recommends smaller batches
_CODESTRAL_MAX_BATCH_SIZE = 128  # Conservative estimate
_GEMINI_MAX_BATCH_SIZE = 100  # Gemini API limit is 100 requests per batch
_QWEN_MAX_BATCH_SIZE = 32  # qwen3-embedding-8b OpenRouter batch limit


def _get_tiktoken_encoding() -> object:
    """Lazy-load tiktoken encoding.

    Returns
    -------
    object
        The ``cl100k_base`` tiktoken encoding.

    Raises
    ------
    RuntimeError
        If ``tiktoken`` is not installed.
    """
    try:
        import tiktoken  # noqa: PLC0415 — presence probe, kept lazy for a clearer error message
    except ImportError as exc:
        msg = "tiktoken is required for token-aware truncation"
        raise RuntimeError(msg) from exc
    return tiktoken.get_encoding("cl100k_base")


def _truncate_to_token_limit(text: str, max_tokens: int, encoder: object) -> str:
    """Truncate text to fit within a token limit using tiktoken.

    Parameters
    ----------
    text : str
        The text to potentially truncate.
    max_tokens : int
        Maximum number of tokens allowed.
    encoder : object
        Tiktoken encoder used for tokenisation and decoding.

    Returns
    -------
    str
        The original text if within the limit, otherwise the truncated text.
    """
    encode_fn = getattr(encoder, "encode", None)
    decode_fn = getattr(encoder, "decode", None)
    if not callable(encode_fn) or not callable(decode_fn):
        return text
    raw_tokens: object = encode_fn(text)
    tokens: list[object] = list(raw_tokens) if isinstance(raw_tokens, (list, tuple)) else []
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    raw_result: object = decode_fn(truncated_tokens)
    result: str = str(raw_result)
    return result


def _load_api_key_from_env(
    env_var: str,
    *,
    invalid_prefixes: tuple[str, ...] = (),
    min_length: int = 0,
) -> str | None:
    """Load an API key from an environment variable, falling back to a ``.env`` file.

    Searches ``os.environ`` first; if the variable is not set, reads the
    ``.env`` file located at the project root and looks for a matching
    ``<env_var>=<value>`` line.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to look up.
    invalid_prefixes : tuple[str, ...], optional
        If the key starts with any of these prefixes it is treated as invalid.
    min_length : int, optional
        Minimum acceptable key length; shorter keys are treated as invalid.

    Returns
    -------
    str | None
        The API key if found and valid, otherwise ``None``.
    """
    api_key = os.environ.get(env_var)
    if not api_key:
        env_file = get_config().root / ".env"
        if env_file.exists():
            prefix = f"{env_var}="
            with env_file.open(encoding="utf-8") as f:
                for raw_line in f:
                    stripped_line = raw_line.strip()
                    if stripped_line.startswith(prefix):
                        api_key = stripped_line.split("=", 1)[1].strip()
                        break

    if not api_key:
        return None
    if invalid_prefixes and api_key.startswith(invalid_prefixes):
        return None
    if min_length and len(api_key) < min_length:
        return None
    return api_key


load_api_key_from_env = _load_api_key_from_env


def _get_embeddings_with_retry(
    texts: list[str],
    *,
    max_tokens: int,
    max_retries: int,
    make_request: Callable[[list[str]], list[list[float]]],
    is_retryable: Callable[[Exception], bool],
    max_batch_size: int = 2048,
) -> list[list[float]]:
    """Truncate texts and fetch embeddings with exponential-backoff retry.

    Parameters
    ----------
    texts : list[str]
        Raw input texts.
    max_tokens : int
        Token budget per text (provider-specific).
    max_retries : int
        Maximum number of attempts per batch before re-raising the exception.
    make_request : Callable[[list[str]], list[list[float]]]
        ``(truncated_texts) -> embeddings``.  Called once per attempt.
    is_retryable : Callable[[Exception], bool]
        ``(exception) -> bool``.  Return ``True`` to retry the failed batch.
    max_batch_size : int, optional
        Maximum number of texts to send in a single request (default 2048).

    Returns
    -------
    list[list[float]]
        Embedding vectors, one per input text.
    """
    encoder = _get_tiktoken_encoding()
    truncated = [_truncate_to_token_limit(t, max_tokens, encoder) for t in texts]

    # Process in batches if needed
    all_embeddings: list[list[float]] = []
    for batch_start in range(0, len(truncated), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(truncated))
        batch = truncated[batch_start:batch_end]

        for attempt in range(max_retries):
            try:
                batch_embeddings = make_request(batch)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as exc:
                if is_retryable(exc) and attempt < max_retries - 1:
                    time.sleep(_RETRY_DELAYS[attempt])
                else:
                    raise

    return all_embeddings


def _require_api_key(env_var: str) -> str:
    """Load an API key or raise ``ValueError`` if missing.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to look up.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    ValueError
        If the key is not found in the environment or ``.env`` file.
    """
    api_key = _load_api_key_from_env(env_var)
    if not api_key:
        msg = f"{env_var} not found in environment or .env file"
        raise ValueError(msg)
    return api_key


def _build_provider_caller(
    provider: str,
    model: str,
    timeout: float,
) -> tuple[Callable[[list[str]], list[list[float]]], Callable[[Exception], bool]]:
    """Build request/retry callables for the specified embedding provider.

    Parameters
    ----------
    provider : str
        One of ``"openai"``, ``"codestral"``, ``"voyage"``, ``"gemini"``,
        ``"qwen"``.
    model : str
        Model identifier passed to the remote API.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    tuple[Callable[[list[str]], list[list[float]]], Callable[[Exception], bool]]
        ``(make_request, is_retryable)`` callables.

    Raises
    ------
    ValueError
        If *provider* is not a recognised embedding provider.
    """
    if provider == "gemini":
        # Optional cloud SDK: not a pykissembed core/cloud dependency, only
        # needed if the caller actually requests the gemini provider.
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        api_key = _load_api_key_from_env("GOOGLE_API_KEY")
        gemini_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

        def _gemini_request(truncated: list[str]) -> list[list[float]]:
            """Send an embedding request to the Gemini API.

            Parameters
            ----------
            truncated : list[str]
                Pre-truncated input texts.

            Returns
            -------
            list[list[float]]
                Embedding vectors returned by the Gemini API.

            Raises
            ------
            ValueError
                If the API response contains no embeddings.
            """
            contents: list[types.ContentUnion] = [*truncated]
            result = gemini_client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=3072,
                ),
            )
            if result.embeddings is None:
                msg = "Gemini API returned no embeddings"
                raise ValueError(msg)
            return [list(emb.values or []) for emb in result.embeddings]

        def _gemini_is_retryable(exc: Exception) -> bool:
            """Determine whether a Gemini API error is retryable.

            Parameters
            ----------
            exc : Exception
                The caught exception.

            Returns
            -------
            bool
                ``True`` if the error appears to be transient.
            """
            error_str = str(exc).lower()
            return "rate" in error_str or "quota" in error_str or "timeout" in error_str

        return (_gemini_request, _gemini_is_retryable)

    if provider == "openai":
        # Optional cloud SDK: not a pykissembed core dependency, only needed
        # if the caller actually requests the openai provider.
        import openai  # noqa: PLC0415

        openai_client = openai.OpenAI(
            api_key=_load_api_key_from_env("OPENAI_API_KEY"),
            timeout=timeout,
        )
        return (
            lambda t: [
                item.embedding
                for item in openai_client.embeddings.create(
                    input=t,
                    model=model,
                ).data
            ],
            lambda e: isinstance(
                e,
                (openai.RateLimitError, openai.APITimeoutError),
            ),
        )

    if provider in {"codestral", "qwen"}:
        # Optional cloud SDK: not a pykissembed core dependency, only needed
        # if the caller actually requests the codestral provider.
        import requests  # noqa: PLC0415

        api_key = _require_api_key("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        def _openrouter_request(
            truncated: list[str],
        ) -> list[list[float]]:
            """Send an embedding request to OpenRouter.

            Parameters
            ----------
            truncated : list[str]
                Pre-truncated input texts.

            Returns
            -------
            list[list[float]]
                Embedding vectors extracted from the ``data`` field of the
                JSON response.

            Raises
            ------
            ValueError
                If the JSON response does not contain a ``data`` field.
            """
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json={"model": model, "input": truncated},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if "data" not in data:
                msg = f"Unexpected API response: {data}"
                raise ValueError(msg)
            return [item["embedding"] for item in data["data"]]

        return (
            _openrouter_request,
            lambda e: isinstance(
                e,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                ),
            ),
        )

    if provider == "voyage":
        # Optional cloud SDK: not a pykissembed core dependency, only needed
        # if the caller actually requests the voyage provider.
        from voyageai.client import Client as VoyageClient  # noqa: PLC0415

        voyage_client = VoyageClient(
            api_key=_require_api_key("VOYAGE_API_KEY"),
            timeout=timeout,
        )

        def _voyage_is_retryable(exc: Exception) -> bool:
            """Determine whether a Voyage API error is retryable.

            Parameters
            ----------
            exc : Exception
                The caught exception.

            Returns
            -------
            bool
                ``True`` if the error appears to be a rate-limit error.
            """
            error_str = str(exc).lower()
            return "rate" in error_str and "limit" in error_str

        return (
            lambda t: [
                list(map(float, emb))
                for emb in voyage_client.embed(
                    t,
                    model=model,
                    input_type="document",
                ).embeddings
            ],
            _voyage_is_retryable,
        )

    msg = f"Unknown embedding provider: {provider!r}"
    raise ValueError(msg)


# Per-provider defaults: (max_tokens, default_model, default_timeout, max_batch_size)
_PROVIDER_DEFAULTS: dict[str, tuple[int, str, float, int]] = {
    "openai": (_OPENAI_MAX_TOKENS, "text-embedding-3-large", 30.0, _OPENAI_MAX_BATCH_SIZE),
    "codestral": (_CODESTRAL_MAX_TOKENS, CODESTRAL_EMBED_MODEL, 120.0, _CODESTRAL_MAX_BATCH_SIZE),
    "voyage": (_VOYAGE_MAX_TOKENS, VOYAGE_CODE_MODEL, 120.0, _VOYAGE_MAX_BATCH_SIZE),
    "gemini": (_GEMINI_MAX_TOKENS, GEMINI_EMBED_MODEL, 120.0, _GEMINI_MAX_BATCH_SIZE),
    "qwen": (_QWEN_MAX_TOKENS, QWEN_EMBED_MODEL, 120.0, _QWEN_MAX_BATCH_SIZE),
}


def get_embeddings_batch(
    texts: list[str],
    *,
    provider: str = "openai",
    model: str | None = None,
    max_retries: int = 3,
    timeout: float | None = None,
) -> list[list[float]]:
    """Get embeddings for a list of texts from a supported provider.

    Parameters
    ----------
    texts : list[str]
        Input texts to embed.
    provider : str, optional
        One of ``"openai"``, ``"codestral"``, ``"voyage"``, ``"gemini"``,
        ``"qwen"``
        (default ``"openai"``).
    model : str | None, optional
        Model override.  ``None`` uses the provider default.
    max_retries : int, optional
        Maximum number of retry attempts per batch (default 3).
    timeout : float | None, optional
        Request timeout override in seconds.  ``None`` uses the provider
        default.

    Returns
    -------
    list[list[float]]
        Embedding vectors, one per input text.
    """
    max_tokens, default_model, default_timeout, max_batch_size = _PROVIDER_DEFAULTS[provider]
    make_request, is_retryable = _build_provider_caller(
        provider,
        model or default_model,
        timeout if timeout is not None else default_timeout,
    )
    return _get_embeddings_with_retry(
        texts,
        max_tokens=max_tokens,
        max_retries=max_retries,
        make_request=make_request,
        is_retryable=is_retryable,
        max_batch_size=max_batch_size,
    )


def compute_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Parameters
    ----------
    a : list[float]
        First vector.
    b : list[float]
        Second vector.

    Returns
    -------
    float
        Cosine similarity in the range ``[-1, 1]``, or ``0.0`` if either
        vector has zero norm.
    """
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def compute_combined_embedding(
    openai_text_emb: list[float],
    openai_ast_emb: list[float],
    codestral_text_emb: list[float],
    codestral_ast_emb: list[float],
    voyage_text_emb: list[float],
    voyage_ast_emb: list[float],
    gemini_text_emb: list[float],
    gemini_ast_emb: list[float],
    qwen_text_emb: list[float],
    qwen_ast_emb: list[float],
) -> list[float]:
    """Compute a combined embedding by concatenating all ten base embeddings and L2-normalising.

    The ten input embeddings are concatenated in order (OpenAI text,
    OpenAI AST, Codestral text, Codestral AST, Voyage text, Voyage AST,
    Gemini text, Gemini AST, Qwen text, Qwen AST) using Python list addition,
    and the resulting vector is L2-normalised. If the combined vector has zero norm, it is
    returned unnormalised.

    Parameters
    ----------
    openai_text_emb : list[float]
        OpenAI embedding of the raw text.
    openai_ast_emb : list[float]
        OpenAI embedding of the AST representation.
    codestral_text_emb : list[float]
        Codestral embedding of the raw text.
    codestral_ast_emb : list[float]
        Codestral embedding of the AST representation.
    voyage_text_emb : list[float]
        Voyage embedding of the raw text.
    voyage_ast_emb : list[float]
        Voyage embedding of the AST representation.
    gemini_text_emb : list[float]
        Gemini embedding of the raw text.
    gemini_ast_emb : list[float]
        Gemini embedding of the AST representation.
    qwen_text_emb : list[float]
        Qwen embedding of the raw text.
    qwen_ast_emb : list[float]
        Qwen embedding of the AST representation.

    Returns
    -------
    list[float]
        L2-normalised concatenation of the ten input embeddings.
    """
    combined = (
        openai_text_emb
        + openai_ast_emb
        + codestral_text_emb
        + codestral_ast_emb
        + voyage_text_emb
        + voyage_ast_emb
        + gemini_text_emb
        + gemini_ast_emb
        + qwen_text_emb
        + qwen_ast_emb
    )

    # L2 normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = (np.array(combined) / norm).tolist()

    return combined


def _is_float_embedding(value: object) -> TypeGuard[list[float]]:
    """Check whether *value* is a list whose elements are all floats.

    Parameters
    ----------
    value : object
        The value to inspect.

    Returns
    -------
    bool
        ``True`` if *value* is a ``list`` and every element is a ``float``.
    """
    if not isinstance(value, list):
        return False
    return all(isinstance(component, float) for component in cast("list[object]", value))


is_float_embedding = _is_float_embedding


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Check whether *value* is a dict with all string keys.

    Parameters
    ----------
    value : object
        The value to inspect.

    Returns
    -------
    bool
        ``True`` if *value* is a ``dict`` and every key is a ``str``.
    """
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


is_str_object_dict = _is_str_object_dict


def _is_embedding_cache(value: object) -> TypeGuard[dict[str, list[float]]]:
    """Check whether *value* is a ``dict[str, list[float]]`` embedding cache.

    Parameters
    ----------
    value : object
        The value to inspect.

    Returns
    -------
    bool
        ``True`` if *value* is a ``dict`` with all string keys mapping to
        ``list[float]`` embedding vectors.
    """
    return is_str_object_dict(value) and all(
        _is_float_embedding(embedding) for embedding in value.values()
    )


is_embedding_cache = _is_embedding_cache


def get_cached_embedding(
    baselines: dict[str, object],
    content_hash: str,
    cache_key: str,
) -> list[float] | None:
    """Look up a cached embedding in a nested dict structure.

    Expects *baselines* to contain a sub-dict under *cache_key*, which in
    turn maps *content_hash* to an embedding vector (``list[float]``).
    Returns ``None`` if the sub-dict is missing, is not a valid
    ``dict[str, object]``, or the entry for *content_hash* is not a
    ``list[float]``.

    Parameters
    ----------
    baselines : dict[str, object]
        Top-level cache mapping provider/cache keys to per-hash dicts.
    content_hash : str
        Hash identifying the content whose embedding is requested.
    cache_key : str
        Key selecting the provider-specific sub-dict within *baselines*.

    Returns
    -------
    list[float] | None
        The cached embedding, or ``None`` if not found or invalid.
    """
    provider_cache = baselines.get(cache_key)
    if not is_str_object_dict(provider_cache):
        return None

    cached_embedding = provider_cache.get(content_hash)
    if not _is_float_embedding(cached_embedding):
        return None
    return cached_embedding
