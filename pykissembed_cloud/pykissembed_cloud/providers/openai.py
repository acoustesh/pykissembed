"""OpenAI text-embedding-3-large via OpenRouter."""

from __future__ import annotations

from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    """``openai/text-embedding-3-large`` routed through OpenRouter."""

    name = "openai"
    model_id = "openai/text-embedding-3-large"
    schema_version = "1"
    max_tokens = 8191
    batch_size = 100


__all__ = ["OpenAIProvider"]
