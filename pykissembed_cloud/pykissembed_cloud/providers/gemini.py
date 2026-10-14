"""Google gemini-embedding-2 via OpenRouter."""

from __future__ import annotations

from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


class GeminiProvider(OpenAICompatProvider):
    """``google/gemini-embedding-2`` routed through OpenRouter."""

    name = "gemini"
    model_id = "google/gemini-embedding-2"
    schema_version = "1"
    max_tokens = 8000  # model limit is 8192; prefix adds a few tokens
    batch_size = 100
    # gemini-embedding-2 selects the task by text prefix, not a request field.
    input_prefix = "task: sentence similarity | query: "


__all__ = ["GeminiProvider"]
