"""PCA solver selection and numerical regressions for wide GPU inputs."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.decomposition import PCA, IncrementalPCA

from pykissembed.similarity import pca
from pykissembed.similarity.types import FunctionInfo

if TYPE_CHECKING:
    import numpy.typing as npt


class _GPUArray(np.ndarray):
    """NumPy array with CuPy's host-transfer interface for CPU-only tests."""

    def get(self) -> np.ndarray:
        """Return the underlying NumPy array.

        Returns
        -------
        np.ndarray
            Host array without the GPU test-double subclass.
        """
        return np.asarray(self)


@pytest.fixture
def gpu_backend(monkeypatch: pytest.MonkeyPatch):
    """Emulate GPU array operations and record which PCA solver was invoked.

    Returns
    -------
    dict
        Recorded covariance-model construction and reduced-SVD input shapes.
    """
    calls: dict[str, list[tuple[int, ...]]] = {"covariance": [], "svd": []}

    def array(value: npt.ArrayLike, dtype: npt.DTypeLike | None = None):
        return np.asarray(value, dtype=dtype).view(_GPUArray)

    def incremental_model(*, n_components: int, batch_size: int):
        estimator = IncrementalPCA(n_components=n_components, batch_size=batch_size)

        def fit(value: np.ndarray) -> None:
            assert batch_size == value.shape[0]
            assert n_components == min(value.shape)
            calls["svd"].append(value.shape)
            _ = estimator.fit(value)
            ratio = estimator.explained_variance_ratio_
            assert ratio is not None
            model.explained_variance_ratio_ = array(ratio)

        model = SimpleNamespace(fit=fit, transform=lambda value: array(estimator.transform(value)))
        return model

    def covariance_model(*, n_components: int):
        calls["covariance"].append((n_components,))
        return PCA(n_components=n_components, svd_solver="full")

    cp = SimpleNamespace(
        asarray=array,
        cumsum=lambda value: array(np.cumsum(value)),
        float32=np.float32,
    )
    real_import = pca.import_module

    def import_backend(name: str):
        if name == "cuml.decomposition":
            return SimpleNamespace(IncrementalPCA=incremental_model)
        return real_import(name)

    monkeypatch.setattr(pca, "import_module", import_backend)
    monkeypatch.setattr(pca, "_load_cupy_module", lambda: cp)
    monkeypatch.setattr(pca, "_get_pca_class", lambda: (covariance_model, True))
    return calls


@pytest.mark.parametrize("shape", [(20, 96), (96, 20), (20, 20)])
def test_gpu_solver_selection_and_variance_match_sklearn(
    gpu_backend: dict[str, list[tuple[int, ...]]], shape: tuple[int, int]
) -> None:
    """Only wide GPU matrices bypass covariance PCA, preserving the variance target."""
    values = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    values += np.arange(shape[1], dtype=np.float32)
    cache = {str(i): row.tolist() for i, row in enumerate(values)}
    model, count, is_gpu = pca.fit_pca(cache, 0.8)
    reference = PCA(n_components=min(shape), svd_solver="full").fit(values)
    expected_count = int(np.searchsorted(np.cumsum(reference.explained_variance_ratio_), 0.8)) + 1

    assert model is not None
    assert is_gpu is True
    assert count == expected_count
    assert len(gpu_backend["covariance"]) == int(shape[0] >= shape[1])
    assert gpu_backend["svd"] == ([shape] if shape[0] < shape[1] else [])
    ratio = np.asarray(getattr(model, "explained_variance_ratio_", None))
    np.testing.assert_allclose(ratio, reference.explained_variance_ratio_, atol=1e-6)


@pytest.mark.parametrize("rank", [3, 20])
def test_wide_gpu_projection_and_cache_match_sklearn(
    gpu_backend: dict[str, list[tuple[int, ...]]], rank: int
) -> None:
    """Cached PCA preserves centered geometry for training and unseen embeddings."""
    rng = np.random.default_rng(42)
    values = (rng.normal(size=(20, rank)) @ rng.normal(size=(rank, 96))).astype(np.float32)
    values += 5
    cache = {str(i): row.tolist() for i, row in enumerate(values)}
    fitted_cache = {}
    result = pca.fit_pca(cache, 0.9, pca_cache=fitted_cache, cache_key="combined")
    assert pca.fit_pca(cache, 0.9, pca_cache=fitted_cache, cache_key="combined") is result
    model, count, is_gpu = result
    assert model is not None
    assert gpu_backend["svd"] == [(20, 96)]

    unseen = rng.normal(size=(4, 96)).astype(np.float32) + 5
    probes = np.concatenate((values[:4], unseen))
    functions = [
        FunctionInfo(
            name=str(i), file="sample.py", start_line=1, end_line=2,
            loc=2, hash=str(i), text="", embedding=row.tolist(),
        )
        for i, row in enumerate(probes)
    ]
    functions[0].embedding = None
    pca.transform_embeddings_with_pca(functions, model, count, is_gpu=is_gpu)
    assert functions[0].embedding is None
    actual = np.array([function.embedding for function in functions[1:]])
    reference = PCA(n_components=20, svd_solver="full").fit(values)
    expected = reference.transform(probes[1:])[:, :count]
    # Singular vectors may differ in sign; inner products test the complete
    # projected geometry that downstream cosine similarity consumes.
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, rtol=2e-5, atol=1e-3)

    _ = pca.fit_pca(cache, 0.5, pca_cache=fitted_cache, cache_key="combined")
    assert gpu_backend["svd"] == [(20, 96), (20, 96)]


def test_wide_constant_embeddings_remain_finite(
    gpu_backend: dict[str, list[tuple[int, ...]]]
) -> None:
    """Zero-variance inputs retain finite zero projections despite undefined variance."""
    values = np.ones((10, 32), dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="invalid value"):
        model, count, is_gpu = pca.fit_pca({str(i): row.tolist() for i, row in enumerate(values)}, 1.0)
    assert model is not None
    assert is_gpu is True
    assert 1 <= count <= len(values)
    np.testing.assert_array_equal(model.transform(values.view(_GPUArray)), 0)
    assert gpu_backend["covariance"] == []


def test_wide_cpu_inputs_keep_sklearn_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Machines without cuML continue to use sklearn PCA for wide inputs."""
    monkeypatch.setattr(pca, "_get_pca_class", lambda: (PCA, False))
    values = np.random.default_rng(42).normal(size=(20, 96)).astype(np.float32)
    model, count, is_gpu = pca.fit_pca({str(i): row.tolist() for i, row in enumerate(values)}, 0.9)
    assert isinstance(model, PCA)
    assert is_gpu is False
    assert 1 <= count <= len(values)


@pytest.mark.skipif(find_spec("cupy") is None, reason="CuPy is unavailable")
def test_real_gpu_wide_pca_avoids_feature_squared_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit a matrix whose fp32 feature covariance alone would exceed 12 GiB."""
    cp = import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device is unavailable")

    def reject_covariance(*, n_components: int):
        pytest.fail(f"wide GPU PCA called the covariance solver: {n_components}")

    monkeypatch.setattr(pca, "_get_pca_class", lambda: (reject_covariance, True))
    values = np.random.default_rng(42).normal(size=(32, 65536)).astype(np.float32)
    cache = {str(i): row.tolist() for i, row in enumerate(values)}
    model, count, is_gpu = pca.fit_pca(cache, 0.9)
    assert model is not None
    assert is_gpu is True
    actual = cp.asnumpy(model.transform(cp.asarray(values)))[:, :count]
    reference = PCA(n_components=32, svd_solver="full").fit(values)
    expected = reference.transform(values)[:, :count]
    expected_count = int(np.searchsorted(np.cumsum(reference.explained_variance_ratio_), 0.9)) + 1
    assert count == expected_count
    np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, rtol=5e-3, atol=25)
