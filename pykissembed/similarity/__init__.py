"""Code Similarity Detection package for pykissembed.

This package detects near-duplicate functions using embedding providers:
- Pairwise similarity detection (finds copy-paste code)
- Neighbor clustering (finds functions with multiple similar counterparts)
- Refactor index computation (combines complexity + similarity for priority)

Providers:
- OpenAI-Text, OpenAI-AST (text-embedding-3-large)
- Codestral-Text, Codestral-AST (codestral-embed via OpenRouter)
- Voyage-Text, Voyage-AST (voyage-code-3)
- Gemini-Text, Gemini-AST (gemini-embedding-001)
- Combined (8-way concatenation)

Modules
-------
    types: FunctionInfo dataclass
    storage: Embedding compression/decompression and cache management
    ast_helpers: Function extraction from Python AST
    pca: PCA dimensionality reduction and clustering
    embeddings: Embedding API clients (OpenAI, Codestral, Voyage, Gemini)
    refactor_index: Refactor index computation
    complexity: Complexity map loaders (CC, COG)
    checks: Unified similarity check workflow
    populate_embeddings: CLI to fetch missing embeddings
"""

from __future__ import annotations

from pykissembed.similarity.ast_helpers import (
    extract_all_function_infos,
    extract_function_infos,
    extract_function_infos_from_file,
)
from pykissembed.similarity.checks import (
    CODESTRAL_AST_PROVIDER,
    CODESTRAL_TEXT_PROVIDER,
    COMBINED_PROVIDER,
    GEMINI_AST_PROVIDER,
    GEMINI_TEXT_PROVIDER,
    OPENAI_AST_PROVIDER,
    OPENAI_TEXT_PROVIDER,
    VOYAGE_AST_PROVIDER,
    VOYAGE_TEXT_PROVIDER,
    run_provider_similarity_checks,
)
from pykissembed.similarity.complexity import (
    load_all_complexity_maps,
    load_complexity_maps,
)
from pykissembed.similarity.embeddings import (
    compute_combined_embedding,
    compute_cosine_similarity,
    get_cached_embedding,
    get_embeddings_batch,
)
from pykissembed.similarity.pca import (
    cluster_functions_kmeans_with_pca,
    fit_pca,
    transform_embeddings_with_pca,
)
from pykissembed.similarity.refactor_index import (
    compute_max_similarities,
    compute_refactor_indices,
    compute_similarity_indices,
    compute_similarity_matrix,
    get_refactor_priority_message,
    get_refactor_priority_message_for_complexity,
)
from pykissembed.similarity.storage import (
    REGISTRY,
    EmbeddingRegistry,
    HashType,
    ProviderEntry,
    get_valid_hashes,
    load_baselines,
    load_minimal_baselines,
    load_provider_embeddings,
    save_baselines,
)
from pykissembed.similarity.types import FunctionInfo, PCAModel

__all__ = [
    "CODESTRAL_AST_PROVIDER",
    "CODESTRAL_TEXT_PROVIDER",
    "COMBINED_PROVIDER",
    "GEMINI_AST_PROVIDER",
    "GEMINI_TEXT_PROVIDER",
    "OPENAI_AST_PROVIDER",
    "OPENAI_TEXT_PROVIDER",
    "REGISTRY",
    "VOYAGE_AST_PROVIDER",
    "VOYAGE_TEXT_PROVIDER",
    "EmbeddingRegistry",
    "FunctionInfo",
    "HashType",
    "PCAModel",
    "ProviderEntry",
    "cluster_functions_kmeans_with_pca",
    "compute_combined_embedding",
    "compute_cosine_similarity",
    "compute_max_similarities",
    "compute_refactor_indices",
    "compute_similarity_indices",
    "compute_similarity_matrix",
    "extract_all_function_infos",
    "extract_function_infos",
    "extract_function_infos_from_file",
    "fit_pca",
    "get_cached_embedding",
    "get_embeddings_batch",
    "get_refactor_priority_message",
    "get_refactor_priority_message_for_complexity",
    "get_valid_hashes",
    "load_all_complexity_maps",
    "load_baselines",
    "load_complexity_maps",
    "load_minimal_baselines",
    "load_provider_embeddings",
    "run_provider_similarity_checks",
    "save_baselines",
    "transform_embeddings_with_pca",
]
