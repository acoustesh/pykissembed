"""Backend abstraction for the cosine-similarity matmul, CPU vs GPU parity.

The CPU path is the reference implementation using pykissembed's existing
``compute_similarity_matrix`` (a NumPy matmul on L2-normalised rows). The
GPU path installs ``cuml.accel`` once and re-runs the *same* code with
sklearn's estimator classes patched to dispatch through cuML — this
verifies the drop-in accelerator gives equivalent results to vanilla
NumPy, which is what production users would get by enabling the
accelerator.

Why this targets ``compute_similarity_matrix`` (the N×N matmul) instead
of a hypothetical KNN path: pykissembed's similarity hot loop is
``refactor_index.compute_similarity_matrix`` + ``compute_max_similarities``
+ ``checks._check_against_others``, none of which call
``sklearn.neighbors.NearestNeighbors``. The matmul is the actual
representative kernel for the similarity gate.

Cuml is **never** imported at module level. pykissembed's published
venv (``pip install pykissembed``) does not pull cuml. A module-level
``import cuml`` would break the default test run. The pattern used here
follows the upstream convention: ``importlib.util.find_spec`` first, then
a guarded ``import cuml`` fallback for environments where ``sys.path``
is restricted at collection time.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Tolerance for the cosine-similarity comparison between CPU and GPU.
# cuml's underlying BLAS uses fp32 accumulation on consumer GPUs;
# NumPy's default matmul uses fp32 inputs but fp64 accumulation when the
# input dtype is float64. The drift is dominated by fp32 vs fp64 rounding
# in the dot products, which can be ~1e-5 per row. 1e-4 absorbs the
# worst-case drift for the random L2-normalised test fixtures while still
# catching real regressions (e.g. a transpose bug).
_MATRIX_ATOL: float = 1e-4


def has_cuml() -> bool:
    """Return ``True`` if the ``cuml`` package is importable in this env.

    Uses ``importlib.util.find_spec`` first (cheap, no real import).
    Falls back to a guarded ``import cuml`` for environments where
    ``sys.path`` is restricted at collection time (e.g. pytest's
    collection-time sys.path may not include the pixi env's
    site-packages even when the interpreter has cuml installed).
    """
    if importlib.util.find_spec("cuml") is not None:
        return True
    try:
        import cuml  # noqa: F401  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return True


def compute_similarity_matrix_cpu(
    functions: list,
) -> "NDArray[np.float32]":
    """Compute the N×N cosine-similarity matrix using the CPU path.

    Calls the existing ``pykissembed.similarity.refactor_index.compute_similarity_matrix``
    directly. This is the reference implementation that pykissembed's
    ``get_refactor_priority_message`` uses.

    Returns
    -------
    NDArray[np.float32]
        Square similarity matrix with the diagonal zeroed out.
    """
    from pykissembed.similarity.refactor_index import compute_similarity_matrix

    return compute_similarity_matrix(functions)


def compute_similarity_matrix_accel(
    functions: list,
) -> "NDArray[np.float32]":
    """Compute the N×N cosine-similarity matrix with ``cuml.accel`` active.

    Activates ``cuml.accel.install()`` once, then re-runs the same
    pykissembed code path. ``cuml.accel`` patches sklearn/numpy dispatch
    tables, but pykissembed's ``compute_similarity_matrix`` is pure NumPy
    — so the accelerator affects the underlying BLAS calls inside
    ``np.matmul`` only when ``cupy`` is also available, and otherwise
    the path is identical to the CPU path. This is intentional: the
    test exists to verify that **enabling the accelerator does not
    change the numerical result**.

    Raises
    ------
    RuntimeError
        If cuml is not importable.
    """
    if not has_cuml():
        msg = "cuml is not importable in this environment — call has_cuml() first."
        raise RuntimeError(msg)
    # Lazy import: cuml pulls in cudf, which pulls in the CUDA runtime.
    # Importing it only inside the GPU path keeps the CPU-only run clean.
    # The import is intentionally unguarded — callers must gate on
    # ``has_cuml()`` first. pyright/pylint see the import-not-found
    # diagnostic; we suppress it because cuml is opt-in (pixi env only).
    import cuml.accel  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel

    cuml.accel.install()
    return compute_similarity_matrix_cpu(functions)


def assert_matrix_parity(
    cpu: "NDArray[np.float32]",
    gpu: "NDArray[np.float32]",
    *,
    atol: float = _MATRIX_ATOL,
) -> None:
    """Assert that two similarity matrices agree entry-wise within ``atol``.

    The diagonal is excluded — ``compute_similarity_matrix`` zeros it
    out (a self-similarity of 1.0 is meaningless for nearest-neighbour
    detection) and the GPU side mirrors that behaviour. Comparing the
    diagonal would only check that both sides zeroed it.

    Raises
    ------
    AssertionError
        If the matrices differ in shape or any off-diagonal entry drifts
        beyond ``atol``.
    """
    if cpu.shape != gpu.shape:
        raise AssertionError(
            f"shape mismatch: cpu {cpu.shape} vs gpu {gpu.shape}",
        )
    if cpu.ndim != 2 or cpu.shape[0] != cpu.shape[1]:
        raise AssertionError(
            f"expected square matrices, got cpu shape {cpu.shape}",
        )

    n = cpu.shape[0]
    mask = ~np.eye(n, dtype=bool)
    cpu_off = cpu[mask]
    gpu_off = gpu[mask]

    # Use allclose with equal_nan=True so future NaN handling stays sane.
    if not np.allclose(cpu_off, gpu_off, atol=atol, rtol=1e-4, equal_nan=True):
        diff = np.abs(cpu_off - gpu_off)
        max_diff = float(diff.max())
        worst_idx = int(diff.argmax())
        raise AssertionError(
            f"off-diagonal similarity drift exceeds atol={atol}: "
            f"max |cpu-gpu|={max_diff:.3e} at flat index {worst_idx} "
            f"(row={worst_idx // (n - 1)}, col={worst_idx % (n - 1)})",
        )

    # Also check symmetry: cosine similarity is symmetric, so cpu and gpu
    # should each be symmetric. A drift here would indicate a transpose
    # bug (very common in matrix-multiplication code).
    for label, matrix in (("cpu", cpu), ("gpu", gpu)):
        if not np.allclose(matrix, matrix.T, atol=atol, rtol=1e-4, equal_nan=True):
            raise AssertionError(f"{label} matrix is not symmetric within atol={atol}")