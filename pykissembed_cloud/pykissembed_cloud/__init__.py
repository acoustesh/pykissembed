"""pykissembed-cloud — OpenRouter-routed cloud embedding providers for pykissembed.

Three providers are bundled (``openai``, ``gemini``, ``qwen``), all
routed through the OpenAI-compatible OpenRouter API. A single
``OPENROUTER_API_KEY`` enables any of them.

The package also ships a small ``dotenv`` loader that walks up from
``Path.cwd()`` looking for a ``.env`` file, so users can drop their
key into a project-root ``.env`` instead of exporting it.
"""

from __future__ import annotations

__version__ = "0.1.4"

__all__ = ["__version__"]
