"""Parity tests: cosine-similarity matmul CPU (NumPy) vs GPU (``cuml.accel``).

These tests verify that pykissembed's similarity gate produces equivalent
results whether the N×N cosine-similarity matmul runs on CPU (vanilla
NumPy, fp32/fp64) or GPU (``cuml.accel``-accelerated, fp32 BLAS). They do
**not** test the similarity test surface that ``pykissembed`` auto-
collects (that lives in ``pykissembed/checks/code_similarity.py`` and is
exercised by ``pytest -m similarity``). They test the underlying numerical
kernel that pykissembed's similarity checks depend on
(``refactor_index.compute_similarity_matrix``).

Environment requirements:

- **CPU tests** run in any environment that has ``numpy`` (the
  pykissembed venv has it).
- **GPU parity test** additionally requires ``cuml`` (only available in
  the pixi RAPIDS env at ``~/rapids-playground/``). It calls
  ``pytest.skip`` cleanly when cuml is not importable, so the default
  ``pytest`` run is unaffected.

Cuml is never imported at module level. See ``_similarity_backend.py`` for
the rationale (module-level ``pytest.importorskip("cuml")`` would skip the
entire file, including the CPU tests).
"""

from __future__ import annotations

import numpy as np
import pytest

# Sibling helper module; pytest's default import mode prepends the test
# file's directory to ``sys.path``, so the bare name resolves correctly.
# A previous version used ``from tests._similarity_backend import …``,
# which fails because ``pykissembed_local/tests/__init__.py`` and
# ``pykissembed_cloud/tests/__init__.py`` shadow the root ``tests/``
# namespace package on ``sys.path``. See commit history for context.
from _similarity_backend import (
    assert_matrix_parity,
    compute_similarity_matrix_accel,
    compute_similarity_matrix_cpu,
    has_cuml,
)
from pykissembed.similarity.types import FunctionInfo


# Small enough to keep the CPU test fast; large enough that the GPU path
# actually does meaningful work in the pixi env.
_DEFAULT_N_SAMPLES = 40
_DEFAULT_N_FEATURES = 64


def _make_function_infos(
    n_samples: int = _DEFAULT_N_SAMPLES,
    n_features: int = _DEFAULT_N_FEATURES,
    *,
    seed: int = 0,
) -> list[FunctionInfo]:
    """Build deterministic L2-normalised random ``FunctionInfo`` fixtures.

    The ``embedding`` attribute is the only field exercised by
    ``compute_similarity_matrix``. Other fields are filled with stable
    sentinels so the dataclass doesn't reject the input.
    """
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal(size=(n_samples, n_features))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalised = (raw / norms).astype(np.float32)

    # The hash/loc/text fields are unused by compute_similarity_matrix
    # but the dataclass requires them; supply stable sentinels.
    return [
        FunctionInfo(
            name=f"fixture_fn_{i}",
            file=f"fixture_{i}.py",
            start_line=1,
            end_line=2,
            loc=2,
            hash=f"hash_{i}",
            text=f"def fixture_fn_{i}(): pass",
            text_for_embedding=f"def fixture_fn_{i}(): pass",
            text_hash=f"text_hash_{i}",
            ast_text=f"def fixture_fn_{i}(): pass",
            embedding=normalised[i].tolist(),
        )
        for i in range(n_samples)
    ]


def test_cpu_similarity_matrix_has_zero_diagonal() -> None:
    """The diagonal of the cosine-similarity matrix is zeroed (self-similarity masked)."""
    funcs = _make_function_infos()
    matrix = compute_similarity_matrix_cpu(funcs)
    assert matrix.shape == (_DEFAULT_N_SAMPLES, _DEFAULT_N_SAMPLES)
    np.testing.assert_array_equal(np.diag(matrix), 0.0)


def test_cpu_similarity_matrix_is_symmetric() -> None:
    """Cosine similarity is symmetric, so the matrix should be too."""
    funcs = _make_function_infos()
    matrix = compute_similarity_matrix_cpu(funcs)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)


def test_cpu_similarity_matrix_entries_in_unit_interval() -> None:
    """Cosine similarity on L2-normalised rows is in [-1, 1]."""
    funcs = _make_function_infos()
    matrix = compute_similarity_matrix_cpu(funcs)
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    assert float(off_diag.min()) >= -1.0 - 1e-5
    assert float(off_diag.max()) <= 1.0 + 1e-5


@pytest.mark.skipif(not has_cuml(), reason="cuml not available in this environment")
def test_cuml_accel_matches_cpu_similarity_matrix() -> None:
    """CPU and GPU paths must agree on the full N×N similarity matrix.

    This is the regression-guard test: if ``cuml.accel`` changes anything
    in pykissembed's hot loop, this fails.
    """
    funcs = _make_function_infos(n_samples=80, n_features=64)
    cpu = compute_similarity_matrix_cpu(funcs)
    gpu = compute_similarity_matrix_accel(funcs)
    assert_matrix_parity(cpu, gpu, atol=1e-4)


def test_assert_matrix_parity_detects_shape_mismatch() -> None:
    """The parity helper must raise when shapes differ."""
    cpu = np.zeros((4, 4), dtype=np.float32)
    gpu = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_matrix_parity(cpu, gpu)


def test_assert_matrix_parity_detects_drift() -> None:
    """The parity helper must raise when off-diagonal entries drift beyond atol."""
    cpu = np.eye(4, dtype=np.float32)
    gpu = np.eye(4, dtype=np.float32)
    gpu[0, 1] = 1.0  # off-diagonal drift well beyond 1e-4
    cpu[0, 1] = 0.0
    with pytest.raises(AssertionError, match="drift exceeds atol"):
        assert_matrix_parity(cpu, gpu, atol=1e-4)


def test_assert_matrix_parity_detects_asymmetry() -> None:
    """The parity helper must raise when either matrix is asymmetric.

    The matrix must be internally asymmetric *and* identical between the
    two sides — otherwise the off-diagonal drift check fires first.
    """
    # Construct a strictly-upper-triangular matrix (asymmetric) with
    # zeroed diagonal. Both cpu and gpu receive the same matrix, so the
    # drift check passes; the symmetry check then catches the bug.
    cpu = np.array(
        [
            [0.0, 0.5, 0.3],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    gpu = cpu.copy()
    with pytest.raises(AssertionError, match="not symmetric"):
        assert_matrix_parity(cpu, gpu, atol=1e-4)


def test_assert_matrix_parity_accepts_small_fp32_drift() -> None:
    """Reasonable fp32 round-trip drift (well under atol) must pass."""
    rng = np.random.default_rng(seed=42)
    base = rng.standard_normal(size=(6, 16), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    cpu = base @ base.T
    np.fill_diagonal(cpu, 0.0)
    # Synthesise a "GPU" copy with tiny drift on a few entries.
    gpu = cpu.copy().astype(np.float32)
    gpu[0, 1] += 5e-5  # within atol=1e-4
    gpu[2, 3] -= 5e-5
    assert_matrix_parity(cpu.astype(np.float32), gpu, atol=1e-4)
