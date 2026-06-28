"""OpenRouter-routed cloud embedding providers.

Each provider is a thin ``OpenAICompatProvider`` subclass that only sets
its identity attributes (``name``, ``model_id``, ``max_tokens``,
``batch_size``, ``schema_version``). The base class handles everything
else — client construction, batching, response parsing.
"""

from __future__ import annotations

from pykissembed_cloud.providers.gemini import GeminiProvider
from pykissembed_cloud.providers.openai import OpenAIProvider
from pykissembed_cloud.providers.qwen import QwenProvider

__all__ = ["GeminiProvider", "OpenAIProvider", "QwenProvider"]
