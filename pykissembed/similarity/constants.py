"""Constants for similarity detection.

Unlike the mega-scrapper original which hardcoded ``MEGA_SCRAPPER_DIR``,
this module resolves paths dynamically from ``[tool.pykissembed]`` config
so the same code works in any consumer project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pykissembed.config import get_config

if TYPE_CHECKING:
    from pathlib import Path

# Refactor index defaults (can be overridden in config)
DEFAULT_REFACTOR_INDEX_THRESHOLD = 15.0
DEFAULT_REFACTOR_INDEX_TOP_N = 5

# OpenRouter config
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
CODESTRAL_EMBED_MODEL = "mistralai/codestral-embed-2505"

# Voyage AI config
VOYAGE_CODE_MODEL = "voyage-code-3"

# Gemini config
GEMINI_EMBED_MODEL = "gemini-embedding-001"


def baselines_dir() -> Path:
    """Return the configured baselines directory.

    Returns
    -------
    Path
        Absolute path to ``<root>/<baseline_dir>``.
    """
    return get_config().baseline_path


def baselines_file() -> Path:
    """Return the path to ``similarity.json``.

    Returns
    -------
    Path
        Absolute path to the similarity baselines JSON file.
    """
    return baselines_dir() / "similarity.json"


def function_hashes_file() -> Path:
    """Return the path to ``function_hashes.json``.

    Returns
    -------
    Path
        Absolute path to the function hashes JSON file.
    """
    return baselines_dir() / "function_hashes.json"


def _embedding_cache_file(provider_name: str) -> Path:
    """Return the path to a provider's compressed embedding cache file.

    Parameters
    ----------
    provider_name : str
        Provider identifier (e.g. ``"openai_text"``).

    Returns
    -------
    Path
        Absolute path to ``<baselines_dir>/<provider_name>_embeddings.json.zlib``.
    """
    return baselines_dir() / f"{provider_name}_embeddings.json.zlib"


# Embedding cache file paths (8 base + 1 combined)
# Text variants (signature + docstring + comments, keyed by text_hash)
OPENAI_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("openai_text")
CODESTRAL_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("codestral_text")
VOYAGE_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("voyage_text")
GEMINI_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("gemini_text")

# AST variants (ast.unparse() output, keyed by hash)
OPENAI_AST_EMBEDDINGS_FILE = _embedding_cache_file("openai_ast")
CODESTRAL_AST_EMBEDDINGS_FILE = _embedding_cache_file("codestral_ast")
VOYAGE_AST_EMBEDDINGS_FILE = _embedding_cache_file("voyage_ast")
GEMINI_AST_EMBEDDINGS_FILE = _embedding_cache_file("gemini_ast")

# Combined (8-way concatenation, keyed by text_hash)
COMBINED_EMBEDDINGS_FILE = _embedding_cache_file("combined")
