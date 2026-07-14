"""OpenAI-compatible cloud embedding providers.

``openai``, ``gemini``, and ``qwen`` are thin ``OpenAICompatProvider`` subclasses
routed through OpenRouter (single ``OPENROUTER_API_KEY``); ``jina`` targets the
native Jina API with its own ``JINA_API_KEY``. Each subclass only sets its
identity attributes (and, for Jina, its endpoint/key/extra_body overrides) — the
base class handles client construction, batching, and response parsing.
"""

from __future__ import annotations

from pykissembed_cloud.providers.gemini import GeminiProvider
from pykissembed_cloud.providers.jina import JinaProvider
from pykissembed_cloud.providers.openai import OpenAIProvider
from pykissembed_cloud.providers.qwen import QwenProvider

__all__ = ["GeminiProvider", "JinaProvider", "OpenAIProvider", "QwenProvider"]
