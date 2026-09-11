"""PCA dimensionality reduction and clustering helpers.

Ported from ``mega-scrapper/tests/similarity/pca.py``. Supports GPU (cuML)
with CPU (sklearn) fallback.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, runtime_checkable

import numpy as np
import numpy.typing as npt

from pykissembed.similarity.types import PCAModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykissembed.similarity.types import FunctionInfo

# What fit_pca stores per cache key: the fitted model, the component count
# chosen for it, and whether it came from the GPU backend.
type PCACacheEntry = tuple[PCAModel | None, int, bool]

_MIN_EMBEDDINGS_FOR_PCA = 10


@runtime_checkable
class _CupyArray(Protocol):
    """Minimal CuPy array protocol used at dynamic boundaries."""

    def get(self) -> npt.NDArray[np.floating[Any]]:
        """Return host (NumPy) array."""
        ...

    def __getitem__(self, key: object) -> _CupyArray:
        """Slice/index array."""
        ...


@runtime_checkable
class _CupyModule(Protocol):
    """Minimal CuPy module protocol for array conversion operations."""

    float32: type[np.float32]

    def asarray(self, a: npt.ArrayLike, dtype: object | None = None) -> object:
        """Convert array-like input to CuPy array."""
        ...

    def cumsum(self, a: object) -> _CupyArray:
        """Compute cumulative sum."""
        ...


@runtime_checkable
class _PCAEstimator(Protocol):
    """Fit/transform surface shared by cuML and scikit-learn PCA.

    Deliberately methods-only: ``explained_variance_ratio_`` is not set until
    ``fit()`` has run, so including it would make an unfitted estimator fail
    the isinstance check. Read it through :func:`_explained_variance_ratio`
    after fitting instead.
    """

    def fit(self, x: object) -> object:
        """Fit PCA model."""
        ...

    def transform(self, x: object) -> object:
        """Transform embeddings."""
        ...


def _as_pca_estimator(value: object, *, name: str) -> _PCAEstimator:
    """Check that *value* exposes the fit/transform surface.

    Parameters
    ----------
    value : object
        Candidate estimator from cuML or scikit-learn.
    name : str
        Label used in error messages.

    Returns
    -------
    _PCAEstimator
        The same object, narrowed to the estimator protocol.

    Raises
    ------
    TypeError
        If *value* lacks a callable ``fit`` or ``transform``.
    """
    # isinstance against a runtime_checkable Protocol verifies that the methods
    # are present, not that their signatures match — the same guarantee the
    # hasattr checks it replaced provided, but visible to the type checker.
    if not isinstance(value, _PCAEstimator):
        msg = f"{name} must expose callable fit() and transform() methods"
        raise TypeError(msg)
    return value


def _explained_variance_ratio(model: object, *, name: str) -> object:
    """Read ``explained_variance_ratio_`` from a fitted PCA model.

    Parameters
    ----------
    model : object
        A PCA estimator on which ``fit()`` has already been called.
    name : str
        Label used in error messages.

    Returns
    -------
    object
        The raw attribute, left untyped for the array validators downstream.

    Raises
    ------
    TypeError
        If the attribute is absent, which means the model was never fitted.
    """
    ratio = getattr(model, "explained_variance_ratio_", None)
    if ratio is None:
        msg = f"{name} has no explained_variance_ratio_; was fit() called?"
        raise TypeError(msg)
    return ratio


def _load_cupy_module() -> _CupyModule:
    """Load CuPy with runtime validation and typed protocol return.

    Returns
    -------
    _CupyModule
        The validated CuPy module typed as ``_CupyModule``.

    Raises
    ------
    TypeError
        If required CuPy attributes are missing.
    """
    cp_module = import_module("cupy")
    if not isinstance(cp_module, _CupyModule):
        msg = "cupy is missing asarray, cumsum or float32"
        raise TypeError(msg)
    return cp_module


def _is_floating_array(value: object) -> TypeGuard[npt.NDArray[np.floating[Any]]]:
    """Return whether *value* is a NumPy array with a floating dtype.

    Parameters
    ----------
    value : object
        Candidate array from a dynamic (GPU or third-party) boundary.

    Returns
    -------
    bool
        ``True`` when *value* is an ``ndarray`` whose dtype is a subtype of
        ``np.floating``. As a ``TypeGuard`` this makes the dtype check itself
        the proof the checker follows, instead of an unchecked assertion.
    """
    return isinstance(value, np.ndarray) and bool(np.issubdtype(value.dtype, np.floating))


def _to_numpy_float_array(value: object, *, name: str) -> npt.NDArray[np.floating[Any]]:
    """Validate and convert dynamic arrays to NumPy floating arrays.

    Parameters
    ----------
    value : object
        Candidate array from a dynamic (GPU or third-party) boundary.
    name : str
        Label used in error messages to identify the offending value.

    Returns
    -------
    npt.NDArray[np.floating[Any]]
        The input array validated as a floating NumPy ndarray.

    Raises
    ------
    TypeError
        If *value* is not a NumPy ndarray with a floating dtype.
    """
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a NumPy ndarray, got {type(value)!r}"
        raise TypeError(msg)
    if not _is_floating_array(value):
        msg = f"{name} must have a floating dtype, got {value.dtype!r}"
        raise TypeError(msg)
    return value


def _to_numpy_from_cupy(value: object, *, name: str) -> npt.NDArray[np.floating[Any]]:
    """Convert CuPy-like arrays to validated NumPy floating arrays.

    Parameters
    ----------
    value : object
        CuPy-like array exposing a callable ``.get()`` method.
    name : str
        Label used in error messages to identify the offending value.

    Returns
    -------
    npt.NDArray[np.floating[Any]]
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


def _load_sklearn_pca_class() -> Callable[..., object]:
    """Load sklearn PCA class dynamically to avoid stub dependency.

    Returns
    -------
    Callable[..., object]
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


def _load_sklearn_kmeans_class() -> Callable[..., object]:
    """Load sklearn KMeans class dynamically to avoid stub dependency.

    Returns
    -------
    Callable[..., object]
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


def _get_pca_class() -> tuple[Callable[..., object], bool]:
    """Get the best available PCA implementation (GPU or CPU).

    Returns
    -------
    tuple[Callable[..., object], bool]
        ``(PCA_class, is_gpu)``; falls back to sklearn when cuML is
        unavailable or incompatible.
    """
    try:
        cuml_decomp = import_module("cuml.decomposition")
        pca_gpu_cls: Callable[..., object] = cuml_decomp.PCA

        # Test that cuML actually works (catches version mismatches)
        _ = pca_gpu_cls(n_components=2)
    # Unparenthesized multi-exception `except` (PEP 758, Python 3.14+) —
    # equivalent to `except (ImportError, AttributeError):`. Valid syntax
    # on this project's py314+ baseline even though it reads like the
    # old (and in Python 3, invalid) Python 2 two-argument form.
    except ImportError, AttributeError:
        # AttributeError catches cuML version incompatibilities
        return _load_sklearn_pca_class(), False
    else:
        return pca_gpu_cls, True


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

    Parameters
    ----------
    embeddings_cache : dict[str, list[float]]
        Hash-keyed embedding vectors to fit on.
    variance_threshold : float
        Cumulative explained variance target used to pick the component
        count; also part of the cache key.
    pca_cache : dict[str, tuple[PCAModel | None, int, bool]] | None
        Optional session-scoped fitted-model cache keyed by
        ``f"{cache_key}_{variance_threshold}"``.
    cache_key : str
        Provider identifier combined with *variance_threshold* for the
        cache entry.

    Returns
    -------
    tuple[PCAModel | None, int, bool]
        ``(fitted PCA model, number of components to use, is_gpu)``;
        returns ``(None, 0, False)`` when there are fewer than 10
        embeddings.

    Raises
    ------
    TypeError
        If the selected backend's PCA class does not expose the expected
        fit/transform surface, or was not fitted.
    """
    # variance_threshold is part of the cache key, not just cache_key: the
    # same embedding set fitted for two different variance targets needs a
    # different component count, so keying on cache_key alone would return
    # a stale n_components for whichever threshold wasn't fitted first.
    if pca_cache is not None:
        full_key = f"{cache_key}_{variance_threshold}"
        if full_key in pca_cache:
            return pca_cache[full_key]

    if len(embeddings_cache) < _MIN_EMBEDDINGS_FOR_PCA:
        return None, 0, False

    pca_class, is_gpu = _get_pca_class()
    all_embeddings = np.array(list(embeddings_cache.values()), dtype=np.float32)
    n_samples, n_features = all_embeddings.shape
    max_components = min(n_samples, n_features)

    if is_gpu:
        cp = _load_cupy_module()
        all_embeddings_gpu = cp.asarray(all_embeddings)
        raw_model = pca_class(n_components=max_components)
        estimator = _as_pca_estimator(raw_model, name="cuML PCA")
        _ = estimator.fit(all_embeddings_gpu)
        cumulative_variance = _to_numpy_from_cupy(
            cp.cumsum(_explained_variance_ratio(estimator, name="cuML PCA")),
            name="cumsum",
        )
    else:
        raw_model = pca_class(n_components=max_components, random_state=42)
        estimator = _as_pca_estimator(raw_model, name="sklearn PCA")
        _ = estimator.fit(all_embeddings)
        cumulative_variance = np.cumsum(
            _to_numpy_float_array(
                _explained_variance_ratio(estimator, name="sklearn PCA"),
                name="explained_variance_ratio_",
            ),
        )

    # searchsorted returns the 0-based index of the first component whose
    # *cumulative* explained variance reaches variance_threshold; +1
    # converts that index into a component *count* (index 0 passing the
    # threshold means "1 component is enough").
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold).item()) + 1
    n_components = min(n_components, max_components)

    # Checked against the public protocol from the raw object, not narrowed
    # from _PCAEstimator: the two describe transform() differently (dynamic GPU
    # arrays vs NumPy), so narrowing between them would be an unsafe overlap.
    if not isinstance(raw_model, PCAModel):
        msg = "fitted PCA model does not expose transform()"
        raise TypeError(msg)
    result = (raw_model, n_components, is_gpu)
    if pca_cache is not None:
        pca_cache[f"{cache_key}_{variance_threshold}"] = result
    return result


def transform_embeddings_with_pca(
    functions: list[FunctionInfo],
    pca_model: PCAModel,
    n_components: int,
    *,
    is_gpu: bool = True,
) -> None:
    """Transform function embeddings using a pre-fitted PCA model in-place.

    Functions without an embedding are skipped; the rest get their
    ``embedding`` replaced by the first *n_components* reduced dimensions.

    Parameters
    ----------
    functions : list[FunctionInfo]
        Functions whose ``embedding`` values are transformed in place.
    pca_model : PCAModel
        Pre-fitted PCA model (GPU or CPU).
    n_components : int
        Number of leading components to keep.
    is_gpu : bool
        Whether *pca_model* is a GPU (cuML) model; a CPU model triggers a
        performance warning.

    Raises
    ------
    TypeError
        If the GPU backend's ``transform()`` does not return a sliceable
        CuPy array.
    """
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
        pca_gpu_model = _as_pca_estimator(pca_model, name="PCA model")
        transformed = pca_gpu_model.transform(emb_array)
        if not isinstance(transformed, _CupyArray):
            msg = "cuML transform() must return a sliceable CuPy array"
            raise TypeError(msg)
        transformed_slice = transformed[:, :n_components]
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


@runtime_checkable
class _KMeansModel(Protocol):
    """The single k-means method this module calls."""

    def fit_predict(self, x: object) -> object:
        """Fit the model and return a cluster label per sample."""
        ...


def _make_kmeans(
    cls: Callable[..., object],
    *,
    n_clusters: int,
    random_state: int,
    n_init: object,
    max_iter: int,
) -> _KMeansModel:
    """Construct a KMeans instance with validated int ``n_init``.

    Parameters
    ----------
    cls : Callable[..., object]
        Dynamically loaded KMeans class.
    n_clusters : int
        Number of clusters.
    random_state : int
        Seed for reproducibility.
    n_init : object
        Number of initialisations; must be a positive integer.
    max_iter : int
        Maximum iterations per run.

    Returns
    -------
    _KMeansModel
        A configured KMeans instance, checked to expose ``fit_predict``.

    Raises
    ------
    TypeError
        If *n_init* is not a positive integer, or the constructed object
        does not expose ``fit_predict``.
    """
    # sklearn's KMeans also accepts n_init="auto" (its own default since
    # 1.4), but that heuristic's chosen value has changed between sklearn
    # releases — pinning to an explicit positive int here keeps clustering
    # results reproducible across environments/versions.
    if not isinstance(n_init, int) or n_init < 1:
        msg = f"n_init must be a positive int, got {n_init!r}"
        raise TypeError(msg)
    n_init_int = n_init
    model = cls(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init_int,
        max_iter=max_iter,
    )
    # Presence check only: it proves fit_predict exists, not its signature.
    # That is what lets the return type be a protocol the checker can enforce
    # rather than an untyped instance callers may use however they like.
    if not isinstance(model, _KMeansModel):
        msg = "KMeans class did not produce an object with fit_predict()"
        raise TypeError(msg)
    return model


def cluster_functions_kmeans_with_pca(
    functions: list[FunctionInfo],
    pca_model: PCAModel,
    n_components: int,
    n_clusters: int = 2,
) -> tuple[list[list[FunctionInfo]], list[str]]:
    """Cluster functions using k-means on PCA-reduced embeddings.

    Functions without an embedding are excluded. When fewer valid
    functions remain than requested clusters, everything is returned in a
    single ``all_functions`` bucket instead of fitting k-means.

    Parameters
    ----------
    functions : list[FunctionInfo]
        Candidate functions to cluster.
    pca_model : PCAModel
        Pre-fitted PCA model applied before clustering.
    n_components : int
        Number of leading components kept from the transform.
    n_clusters : int, optional
        Number of clusters to form (default 2).

    Returns
    -------
    tuple[list[list[FunctionInfo]], list[str]]
        A tuple of (clustered function lists sorted by start line, cluster
        name strings).
    """
    kmeans_cls = _load_sklearn_kmeans_class()

    embeddings: list[list[float]] = []
    valid_functions: list[FunctionInfo] = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
            valid_functions.append(func)

    # KMeans requires at least as many samples as clusters; when there
    # aren't enough embedded functions to fill every cluster, fall back to
    # one bucket containing everything rather than letting sklearn raise.
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
