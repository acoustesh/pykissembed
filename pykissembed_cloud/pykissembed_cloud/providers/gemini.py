"""Google gemini-embedding-001 via OpenRouter."""

from __future__ import annotations

from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


class GeminiProvider(OpenAICompatProvider):
    """``google/gemini-embedding-001`` routed through OpenRouter."""

    name = "gemini"
    model_id = "google/gemini-embedding-001"
    schema_version = "1"
    max_tokens = 2048
    batch_size = 100


__all__ = ["GeminiProvider"]
