"""Qwen Qwen3-Embedding-8B via OpenRouter."""

from __future__ import annotations

from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


class QwenProvider(OpenAICompatProvider):
    """``qwen/qwen3-embedding-8b`` routed through OpenRouter."""

    name = "qwen"
    model_id = "qwen/qwen3-embedding-8b"
    schema_version = "1"
    max_tokens = 32000
    batch_size = 32


__all__ = ["QwenProvider"]
