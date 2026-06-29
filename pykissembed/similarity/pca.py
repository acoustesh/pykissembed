"""PCA dimensionality reduction and clustering helpers.

Ported from ``mega-scrapper/tests/similarity/pca.py``. Supports GPU (cuML)
with CPU (sklearn) fallback.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pykissembed.similarity.types import FunctionInfo, PCAModel


class _CupyArray(Protocol):
    """Minimal CuPy array protocol used at dynamic boundaries."""

    def get(self) -> npt.NDArray[np.floating[Any]]:
        """Return host (NumPy) array."""
        ...

    def __getitem__(self, key: object) -> _CupyArray:
        """Slice/index array."""
        ...


class _CupyModule(Protocol):
    """Minimal CuPy module protocol for array conversion operations."""

    float32: type[np.float32]

    def asarray(self, a: npt.ArrayLike, dtype: object | None = None) -> object:
        """Convert array-like input to CuPy array."""
        ...

    def cumsum(self, a: object) -> _CupyArray:
        """Compute cumulative sum."""
        ...


class _GpuPCAModel(Protocol):
    """PCA protocol for GPU boundary where input/output are dynamic."""

    explained_variance_ratio_: object

    def fit(self, x: object) -> object:
        """Fit PCA model."""
        ...

    def transform(self, x: object) -> object:
        """Transform embeddings."""
        ...


def _load_cupy_module() -> _CupyModule:
    """Load CuPy with runtime validation and typed protocol return.

    Returns
    -------
        The validated CuPy module typed as ``_CupyModule``.

    Raises
    ------
    TypeError
        If required CuPy attributes are missing.
    """
    cp_module = import_module("cupy")
    if not hasattr(cp_module, "asarray") or not callable(cp_module.asarray):
        msg = "cupy.asarray not available"
        raise TypeError(msg)
    if not hasattr(cp_module, "cumsum") or not callable(cp_module.cumsum):
        msg = "cupy.cumsum not available"
        raise TypeError(msg)
    if not hasattr(cp_module, "float32"):
        msg = "cupy.float32 not available"
        raise TypeError(msg)
    return cast("_CupyModule", cp_module)


def _to_numpy_float_array(value: object, *, name: str) -> npt.NDArray[np.floating[Any]]:
    """Validate and convert dynamic arrays to NumPy floating arrays.

    Returns
    -------
        The input array validated as a floating NumPy ndarray.

    Raises
    ------
    TypeError
        If *value* is not a NumPy ndarray with a floating dtype.
    """
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a NumPy ndarray, got {type(value)!r}"
        raise TypeError(msg)
    value_array = cast("npt.NDArray[np.generic]", value)
    if not np.issubdtype(value_array.dtype, np.floating):
        msg = f"{name} must have a floating dtype, got {value_array.dtype!r}"
        raise TypeError(msg)
    return cast("npt.NDArray[np.floating[Any]]", value_array)


def _to_numpy_from_cupy(value: object, *, name: str) -> npt.NDArray[np.floating[Any]]:
    """Convert CuPy-like arrays to validated NumPy floating arrays.

    Returns
    -------
        The converted and validated floating NumPy ndarray.

    Raises
    ------
    TypeError
        If *value* does not expose a callable ``.get()`` method.
    """
    get_fn = getattr(value, "get", None)
    if get_fn is None or not callable(get_fn):
        msg = f"{name} must expose a callable .get() method"
        raise TypeError(msg)
    return _to_numpy_float_array(get_fn(), name=name)


def _load_sklearn_pca_class() -> type:
    """Load sklearn PCA class dynamically to avoid stub dependency.

    Returns
    -------
        The ``sklearn.decomposition.PCA`` class.

    Raises
    ------
    TypeError
        If ``sklearn.decomposition.PCA`` is unavailable.
    """
    decomp_module = import_module("sklearn.decomposition")
    pca_cls = getattr(decomp_module, "PCA", None)
    if not isinstance(pca_cls, type):
        msg = "sklearn.decomposition.PCA is unavailable"
        raise TypeError(msg)
    return pca_cls


def _load_sklearn_kmeans_class() -> type:
    """Load sklearn KMeans class dynamically to avoid stub dependency.

    Returns
    -------
        The ``sklearn.cluster.KMeans`` class.

    Raises
    ------
    TypeError
        If ``sklearn.cluster.KMeans`` is unavailable.
    """
    cluster_module = import_module("sklearn.cluster")
    kmeans_cls = getattr(cluster_module, "KMeans", None)
    if not isinstance(kmeans_cls, type):
        msg = "sklearn.cluster.KMeans is unavailable"
        raise TypeError(msg)
    return kmeans_cls


def _get_pca_class() -> tuple[type, bool]:
    """Get the best available PCA implementation (GPU or CPU).

    Returns
    -------
        Tuple of (PCA_class, is_gpu). Falls back to sklearn if cuML unavailable.

    """
    try:
        cuml_decomp = import_module("cuml.decomposition")
        pca_gpu_cls: type = cuml_decomp.PCA

        # Test that cuML actually works (catches version mismatches)
        _ = pca_gpu_cls(n_components=2)
        return pca_gpu_cls, True
    except (ImportError, AttributeError):
        # AttributeError catches cuML version incompatibilities
        return _load_sklearn_pca_class(), False


def fit_pca(
    embeddings_cache: dict[str, list[float]],
    variance_threshold: float,
    *,
    pca_cache: dict[str, tuple[PCAModel | None, int, bool]] | None = None,
    cache_key: str = "",
) -> tuple[PCAModel | None, int, bool]:
    """Fit PCA on embeddings cache for dimensionality reduction.

    When *pca_cache* and *cache_key* are supplied the result is stored and
    returned from cache on subsequent calls with the same key.

    Returns
    -------
        Tuple of (fitted PCA model, number of components to use, is_gpu)
        Returns (None, 0, False) if not enough embeddings (<10)

    """
    if pca_cache is not None:
        full_key = f"{cache_key}_{variance_threshold}"
        if full_key in pca_cache:
            return pca_cache[full_key]

    if len(embeddings_cache) < 10:
        return None, 0, False

    pca_class, is_gpu = _get_pca_class()
    all_embeddings = np.array(list(embeddings_cache.values()), dtype=np.float32)
    n_samples, n_features = all_embeddings.shape
    max_components = min(n_samples, n_features)

    if is_gpu:
        cp = _load_cupy_module()
        all_embeddings_gpu = cp.asarray(all_embeddings)
        pca = pca_class(n_components=max_components)
        pca_gpu = cast("_GpuPCAModel", pca)
        pca_gpu.fit(all_embeddings_gpu)
        cumulative_variance = _to_numpy_from_cupy(
            cp.cumsum(pca_gpu.explained_variance_ratio_),
            name="cumsum",
        )
    else:
        pca = pca_class(n_components=max_components, random_state=42)
        pca_cpu = cast("_GpuPCAModel", pca)
        pca_cpu.fit(all_embeddings)
        cumulative_variance = np.cumsum(
            _to_numpy_float_array(
                pca_cpu.explained_variance_ratio_,
                name="explained_variance_ratio_",
            ),
        )

    n_components = int(np.searchsorted(cumulative_variance, variance_threshold).item()) + 1
    n_components = min(n_components, max_components)

    result = (pca, n_components, is_gpu)
    if pca_cache is not None:
        pca_cache[f"{cache_key}_{variance_threshold}"] = result
    return result


def transform_embeddings_with_pca(
    functions: list[FunctionInfo],
    pca_model: PCAModel,
    n_components: int,
    is_gpu: bool = True,
) -> None:
    """Transform function embeddings using pre-fitted PCA model in-place."""
    if not is_gpu:
        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "⚠️  WARNING: Using CPU for PCA transform!\n"
            "    GPU (cuML) is strongly recommended for performance.\n"
            "    Install cuML: conda install -c rapidsai cuml\n" + "=" * 60,
            stacklevel=2,
        )

    # Collect embeddings and indices
    embeddings_to_transform: list[list[float]] = []
    indices: list[int] = []
    for i, func in enumerate(functions):
        if func.embedding is not None:
            embeddings_to_transform.append(func.embedding)
            indices.append(i)

    if not embeddings_to_transform:
        return

    # Backend-specific array creation and transform
    if is_gpu:
        cp = _load_cupy_module()
        emb_array = cp.asarray(embeddings_to_transform, dtype=cp.float32)
        pca_gpu_model = cast("_GpuPCAModel", pca_model)
        transformed = pca_gpu_model.transform(emb_array)
        transformed_array = cast("_CupyArray", transformed)
        transformed_slice = transformed_array[:, :n_components]
        reduced = _to_numpy_from_cupy(transformed_slice, name="transformed")
    else:
        emb_array = np.array(embeddings_to_transform, dtype=np.float32)
        reduced = _to_numpy_float_array(
            pca_model.transform(emb_array)[:, :n_components],
            name="transformed",
        )

    # Assign back
    for idx, emb_idx in enumerate(indices):
        functions[emb_idx].embedding = reduced[idx].tolist()


def _make_kmeans(
    cls: type,
    *,
    n_clusters: int,
    random_state: int,
    n_init: object,
    max_iter: int,
) -> Any:
    """Construct a KMeans instance with validated int ``n_init``.

    Returns
    -------
        A configured KMeans instance.

    Raises
    ------
    TypeError
        If *n_init* is not a positive integer.
    """
    if not isinstance(n_init, int) or n_init < 1:
        msg = f"n_init must be a positive int, got {n_init!r}"
        raise TypeError(msg)
    n_init_int = n_init
    return cls(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init_int,
        max_iter=max_iter,
    )


def cluster_functions_kmeans_with_pca(
    functions: list[FunctionInfo],
    pca_model: PCAModel,
    n_components: int,
    n_clusters: int = 2,
) -> tuple[list[list[FunctionInfo]], list[str]]:
    """Cluster functions using k-means on PCA-reduced embeddings.

    Returns
    -------
        A tuple of (clustered function lists, cluster name strings).
    """
    kmeans_cls = _load_sklearn_kmeans_class()

    embeddings: list[list[float]] = []
    valid_functions: list[FunctionInfo] = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
            valid_functions.append(func)

    if len(valid_functions) < n_clusters:
        return [valid_functions], ["all_functions"]

    emb_matrix = np.array(embeddings, dtype=np.float32)
    emb_reduced = _to_numpy_float_array(
        pca_model.transform(emb_matrix)[:, :n_components],
        name="emb_reduced",
    )

    kmeans = _make_kmeans(
        kmeans_cls,
        n_clusters=n_clusters,
        random_state=42,
        n_init=30,
        max_iter=3000,
    )
    labels_obj = kmeans.fit_predict(emb_reduced)
    labels = np.asarray(labels_obj, dtype=np.int64)

    clusters: list[list[FunctionInfo]] = [[] for _ in range(n_clusters)]
    for func, label in zip(valid_functions, labels, strict=True):
        clusters[int(label)].append(func)

    for cluster in clusters:
        cluster.sort(key=lambda f: f.start_line)

    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
    return clusters, cluster_names
