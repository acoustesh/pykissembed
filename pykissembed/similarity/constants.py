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
QWEN_EMBED_MODEL = "qwen/qwen3-embedding-8b"

# Voyage AI config
VOYAGE_CODE_MODEL = "voyage-code-3"

# Gemini config
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# Jina config — native (non-OpenRouter) API with asymmetric query/passage tasks.
# Each function is embedded twice (once as query, once as passage); similarity is
# the symmetrized cross-score, so Jina uses a dedicated path, not single-vector
# cosine. See pykissembed.similarity.jina_similarity.
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_EMBED_MODEL = "jina-code-embeddings-1.5b"
# code2code: code-vs-code equivalence. nl2code: documented intent vs implementation.
JINA_CODE2CODE_QUERY = "code2code.query"
JINA_CODE2CODE_PASSAGE = "code2code.passage"
JINA_NL2CODE_QUERY = "nl2code.query"
JINA_NL2CODE_PASSAGE = "nl2code.passage"


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


# Embedding cache file paths (10 base + 1 combined)
# Text variants (signature + docstring + comments, keyed by text_hash)
OPENAI_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("openai_text")
CODESTRAL_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("codestral_text")
VOYAGE_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("voyage_text")
GEMINI_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("gemini_text")
QWEN_TEXT_EMBEDDINGS_FILE = _embedding_cache_file("qwen_text")

# AST variants (ast.unparse() output, keyed by hash)
OPENAI_AST_EMBEDDINGS_FILE = _embedding_cache_file("openai_ast")
CODESTRAL_AST_EMBEDDINGS_FILE = _embedding_cache_file("codestral_ast")
VOYAGE_AST_EMBEDDINGS_FILE = _embedding_cache_file("voyage_ast")
GEMINI_AST_EMBEDDINGS_FILE = _embedding_cache_file("gemini_ast")
QWEN_AST_EMBEDDINGS_FILE = _embedding_cache_file("qwen_ast")

# Jina raw caches — query + passage per variant (asymmetric retrieval).
# Text variant (nl2code): query = docstring, passage = code, keyed by text_hash.
# AST variant (code2code): query = code, passage = code, keyed by AST hash.
JINA_TEXT_QUERY_EMBEDDINGS_FILE = _embedding_cache_file("jina_text_query")
JINA_TEXT_PASSAGE_EMBEDDINGS_FILE = _embedding_cache_file("jina_text_passage")
JINA_AST_QUERY_EMBEDDINGS_FILE = _embedding_cache_file("jina_ast_query")
JINA_AST_PASSAGE_EMBEDDINGS_FILE = _embedding_cache_file("jina_ast_passage")

# Combined (14-way concatenation, keyed by text_hash): the 10 cosine members
# plus 4 Jina members (normalized concat(Q,P) and concat(P,Q) for each variant),
# derived on the fly from the raw caches above.
COMBINED_EMBEDDINGS_FILE = _embedding_cache_file("combined")
