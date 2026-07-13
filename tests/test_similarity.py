"""Tests for the similarity sub-package.

These tests verify the core functionality of the ported similarity
modules: AST helpers, storage, embeddings, refactor index, and the
check workflow.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from pykissembed.plugin import (
    _CHECK_STEMS,
    _checks_dir,
)
from pykissembed.similarity import checks as similarity_checks
from pykissembed.similarity.ast_helpers import (
    compute_content_hash,
    extract_function_infos_from_file,
)
from pykissembed.similarity.checks import OPENAI_TEXT_PROVIDER
from pykissembed.similarity.embeddings import (
    compute_combined_embedding,
    compute_cosine_similarity,
    is_float_embedding,
    is_str_object_dict,
    load_api_key_from_env,
)
from pykissembed.similarity.refactor_index import (
    compute_max_similarities,
    compute_refactor_indices,
    compute_similarity_indices,
    compute_similarity_matrix,
)
from pykissembed.similarity.storage import (
    HashType,
    ProviderEntry,
    get_valid_hashes,
)
from pykissembed.similarity.types import FunctionInfo


class TestComputeContentHash:
    """Tests for compute_content_hash."""

    @staticmethod
    def test_deterministic() -> None:
        """Same input produces same hash."""
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        assert h1 == h2

    @staticmethod
    def test_different_input() -> None:
        """Different inputs produce different hashes."""
        assert compute_content_hash("hello") != compute_content_hash("world")

    @staticmethod
    def test_returns_hex_string() -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        h = compute_content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestSharedTypeGuards:
    """Tests for shared runtime type guards."""

    @staticmethod
    def test_is_str_object_dict() -> None:
        """Only dictionaries with string keys satisfy the shared guard."""
        assert is_str_object_dict({"key": object()})
        assert is_str_object_dict({})
        assert not is_str_object_dict({1: "value"})
        assert not is_str_object_dict([("key", "value")])

    @staticmethod
    def test_is_float_embedding() -> None:
        """Only lists containing floats satisfy the shared embedding guard."""
        assert is_float_embedding([0.1, 2.0])
        assert is_float_embedding([])
        assert not is_float_embedding([0.1, 2])
        assert not is_float_embedding((0.1, 2.0))


class TestSimilarityProximityExclusions:
    """Tests for structurally related class/function exclusions."""

    @staticmethod
    def test_class_function_proximity_ignores_blank_and_comment_lines(tmp_path: Path) -> None:
        """A nearby class/function pair is excluded using executable-line distance."""
        source_file = tmp_path / "nearby.py"
        source_file.write_text(
            "class TestCase:\n    pass\n\n# separating comment\ndef test_case():\n    pass\n",
            encoding="utf-8",
        )
        test_class = FunctionInfo(
            name="TestCase",
            file=str(source_file),
            start_line=1,
            end_line=2,
            loc=2,
            hash="class-hash",
            text="class TestCase:\n    pass",
        )
        test_function = FunctionInfo(
            name="test_case",
            file=str(source_file),
            start_line=5,
            end_line=6,
            loc=2,
            hash="function-hash",
            text="def test_case():\n    pass",
        )

        assert similarity_checks._is_excluded_pair(  # noqa: SLF001
            test_class,
            test_function,
            [],
            [],
            class_function_proximity=0,
        )

    @staticmethod
    def test_nested_class_method_pair_is_excluded() -> None:
        """A method inside its class has zero source distance and is excluded."""
        test_class = FunctionInfo(
            name="TestCase",
            file="module.py",
            start_line=1,
            end_line=10,
            loc=4,
            hash="class-hash",
            text="class TestCase:\n    def test_case(self): ...",
        )
        test_method = FunctionInfo(
            name="test_case",
            file="module.py",
            start_line=2,
            end_line=3,
            loc=2,
            hash="method-hash",
            text="def test_case(self):\n    pass",
        )

        assert similarity_checks._is_excluded_pair(  # noqa: SLF001
            test_class,
            test_method,
            [],
            [],
            class_function_proximity=1,
        )
        test_class.embedding = [1.0, 0.0]
        test_method.embedding = [1.0, 0.0]
        assert similarity_checks._find_violations(  # noqa: SLF001
            [test_class, test_method],
            threshold_pair=0.9,
            threshold_neighbor=0.8,
            class_function_proximity=1,
        ) == ([], [])


class TestApiKeyLoading:
    """Tests for shared API-key validation."""

    @staticmethod
    def test_api_key_validation_options(monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid prefixes and minimum length apply to environment keys."""
        env_var = "PYKISSEMBED_TEST_API_KEY"
        monkeypatch.setenv(env_var, "valid-api-key")
        assert load_api_key_from_env(env_var, min_length=10) == "valid-api-key"
        assert load_api_key_from_env(env_var, invalid_prefixes=("valid-",)) is None
        assert load_api_key_from_env(env_var, min_length=20) is None


class TestExtractFunctionInfosFromFile:
    """Tests for extract_function_infos_from_file."""

    @staticmethod
    def test_extracts_functions(tmp_path: Path) -> None:
        """Functions are extracted from a Python file."""
        f = tmp_path / "example.py"
        f.write_text(
            dedent(
                '''
                def foo(x):
                    """Docstring."""
                    return x + 1

                def bar(y):
                    """Another docstring."""
                    return y * 2
                ''',
            ),
            encoding="utf-8",
        )
        funcs = extract_function_infos_from_file(f, min_loc=1)
        names = [fn.name for fn in funcs]
        assert "foo" in names
        assert "bar" in names

    @staticmethod
    def test_skips_init(tmp_path: Path) -> None:
        """__init__ methods are skipped."""
        f = tmp_path / "example.py"
        f.write_text(
            dedent(
                '''
                class MyClass:
                    def __init__(self):
                        pass

                    def real_method(self):
                        """Doc."""
                        return 42
                ''',
            ),
            encoding="utf-8",
        )
        funcs = extract_function_infos_from_file(f, min_loc=1)
        names = [fn.name for fn in funcs]
        assert "__init__" not in names
        assert "real_method" in names

    @staticmethod
    def test_min_loc_filter(tmp_path: Path) -> None:
        """Functions below min_loc are excluded."""
        f = tmp_path / "example.py"
        f.write_text(
            "def tiny():\n"
            '    """Tiny."""\n'
            "    return 1\n"
            "\n"
            "def big():\n"
            '    """Big function.\n'
            "\n"
            "    Parameters\n"
            "    ----------\n"
            "    None.\n"
            "\n"
            "    Returns\n"
            "    -------\n"
            "    int\n"
            "        A number.\n"
            "\n"
            '    """\n'
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    return x + y + z\n",
            encoding="utf-8",
        )
        funcs = extract_function_infos_from_file(f, min_loc=5)
        names = [fn.name for fn in funcs]
        assert "tiny" not in names
        assert "big" in names

    @staticmethod
    def test_function_info_fields(tmp_path: Path) -> None:
        """FunctionInfo has all expected fields populated."""
        f = tmp_path / "example.py"
        f.write_text(
            dedent(
                '''
                def my_func(x):
                    """Docstring.

                    Parameters
                    ----------
                    x : int
                        Input.

                    Returns
                    -------
                    int
                        Output.

                    """
                    return x + 1
                ''',
            ),
            encoding="utf-8",
        )
        funcs = extract_function_infos_from_file(f, min_loc=1)
        assert len(funcs) == 1
        func = funcs[0]
        assert func.name == "my_func"
        assert func.start_line > 0
        assert func.end_line >= func.start_line
        assert func.loc > 0
        assert len(func.hash) == 64
        assert len(func.text_hash) == 64
        assert "def my_func" in func.text
        assert func.text_for_embedding  # non-empty
        assert func.ast_text  # non-empty
        assert func.embedding is None


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity."""

    @staticmethod
    def test_identical_vectors() -> None:
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert compute_cosine_similarity(v, v) == pytest.approx(1.0)

    @staticmethod
    def test_orthogonal_vectors() -> None:
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert compute_cosine_similarity(a, b) == pytest.approx(0.0)

    @staticmethod
    def test_zero_vector() -> None:
        """Zero vector returns 0.0."""
        assert compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)

    @staticmethod
    def test_opposite_vectors() -> None:
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert compute_cosine_similarity(a, b) == pytest.approx(-1.0)


class TestComputeCombinedEmbedding:
    """Tests for compute_combined_embedding."""

    @staticmethod
    def test_length_is_sum_of_inputs() -> None:
        """Combined embedding length equals sum of input lengths."""
        inputs = [[1.0] for _ in range(8)]
        combined = compute_combined_embedding(*inputs)
        assert len(combined) == 8

    @staticmethod
    def test_is_normalized() -> None:
        """Combined embedding is L2-normalized."""
        inputs = [
            [3.0, 0.0],
            [4.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        combined = compute_combined_embedding(*inputs)
        norm = sum(c**2 for c in combined) ** 0.5
        assert norm == pytest.approx(1.0)


class TestGetValidHashes:
    """Tests for get_valid_hashes."""

    @staticmethod
    def test_empty_baselines() -> None:
        """Empty baselines produce empty sets."""
        text, ast, mapping = get_valid_hashes({})
        assert text == set()
        assert ast == set()
        assert mapping == {}

    @staticmethod
    def test_dict_entries() -> None:
        """Dict entries with hash and text_hash are extracted."""
        baselines: dict[str, object] = {
            "function_hashes": {
                "func1": {"hash": "ast123", "text_hash": "text456"},
            },
        }
        text, ast, mapping = get_valid_hashes(baselines)
        assert "text456" in text
        assert "ast123" in ast
        assert mapping["text456"] == "ast123"

    @staticmethod
    def test_legacy_string_entries() -> None:
        """Legacy string entries are treated as AST hashes."""
        baselines: dict[str, object] = {
            "function_hashes": {
                "func1": "legacy_hash",
            },
        }
        _, ast, _ = get_valid_hashes(baselines)
        assert "legacy_hash" in ast

    @staticmethod
    def test_invalid_entry_raises() -> None:
        """Non-str, non-dict entries raise TypeError."""
        baselines: dict[str, object] = {
            "function_hashes": {
                "func1": 42,
            },
        }
        with pytest.raises(TypeError):
            get_valid_hashes(baselines)


class TestProviderEntry:
    """Tests for ProviderEntry properties."""

    @staticmethod
    def test_hash_field_text() -> None:
        """Text provider uses text_hash field."""
        entry = ProviderEntry(
            name="test_text",
            label="Test-Text",
            cache_key="test_text_embeddings",
            file_path=Path("test.json.zlib"),  # never read; dataclass field placeholder
            hash_type=HashType.TEXT,
        )
        assert entry.hash_field == "text_hash"

    @staticmethod
    def test_hash_field_ast() -> None:
        """AST provider uses hash field."""
        entry = ProviderEntry(
            name="test_ast",
            label="Test-AST",
            cache_key="test_ast_embeddings",
            file_path=Path("test.json.zlib"),  # never read; dataclass field placeholder
            hash_type=HashType.AST,
        )
        assert entry.hash_field == "hash"

    @staticmethod
    def test_threshold_keys() -> None:
        """Threshold keys are derived from name."""
        entry = ProviderEntry(
            name="openai_text",
            label="OpenAI-Text",
            cache_key="openai_text_embeddings",
            file_path=Path("test.json.zlib"),  # never read; dataclass field placeholder
            hash_type=HashType.TEXT,
        )
        assert entry.threshold_pair_key == "openai_text_similarity_threshold_pair"
        assert entry.threshold_neighbor_key == "openai_text_similarity_threshold_neighbor"
        assert entry.pca_variance_key == "openai_text_pca_variance_threshold"


class TestRefactorIndex:
    """Tests for refactor index computation."""

    @staticmethod
    def test_similarity_matrix_shape() -> None:
        """Similarity matrix is square with zero diagonal."""
        funcs = [
            FunctionInfo(
                name=f"f{i}",
                file=f"file{i}.py",
                start_line=1,
                end_line=10,
                loc=5,
                hash=f"hash{i}",
                text="text",
                embedding=[1.0, 0.0, 0.0],
            )
            for i in range(3)
        ]
        matrix = compute_similarity_matrix(funcs)
        assert matrix.shape == (3, 3)
        # Diagonal should be zero
        for i in range(3):
            assert matrix[i, i] == pytest.approx(0.0)

    @staticmethod
    def test_max_similarities() -> None:
        """Max similarities returns per-row max."""
        matrix = np.array([[0.0, 0.5, 0.8], [0.5, 0.0, 0.3], [0.8, 0.3, 0.0]], dtype=np.float32)
        max_sims = compute_max_similarities(matrix)
        assert max_sims[0] == pytest.approx(0.8)
        assert max_sims[1] == pytest.approx(0.5)
        assert max_sims[2] == pytest.approx(0.8)

    @staticmethod
    def test_similarity_indices_formula() -> None:
        """Similarity index = 25.403 * max_sim^5."""
        max_sims = np.array([1.0, 0.5], dtype=np.float64)
        indices = compute_similarity_indices(max_sims)
        assert indices[0] == pytest.approx(25.403)
        assert indices[1] == pytest.approx(25.403 * 0.5**5)

    @staticmethod
    def test_refactor_indices_formula() -> None:
        """Refactor index = 0.25*CC + 0.15*COG + 0.6*similarity_index."""
        cc = np.array([10.0], dtype=np.float32)
        cog = np.array([5.0], dtype=np.float32)
        sim_idx = np.array([20.0], dtype=np.float64)
        result = compute_refactor_indices(cc, cog, sim_idx)
        expected = 0.25 * 10.0 + 0.15 * 5.0 + 0.6 * 20.0
        assert result[0] == pytest.approx(expected)


class TestSimilarityConfiguration:
    """Tests for plumbing ``similarity.json`` configuration through checks."""

    @staticmethod
    def test_similarity_violations_suppress_the_test_traceback(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Similarity failures report violations without displaying check implementation code."""
        captured: dict[str, object] = {}

        def fake_fail(message: str, *, pytrace: bool = True) -> None:
            captured["message"] = message
            captured["pytrace"] = pytrace

        monkeypatch.setattr(similarity_checks.pytest, "fail", fake_fail)

        similarity_checks._report_violations(  # noqa: SLF001
            ["module.py:1 duplicated() vs other.py:1 copy() - similarity: 100.0%"],
            [],
            threshold_pair=0.98,
            threshold_neighbor=0.8,
            functions=[],
            load_complexity_maps_fn=lambda: ({}, {}),
            refactor_index_threshold=10.0,
            refactor_index_top_n=10,
        )

        assert captured["pytrace"] is False
        assert "High similarity pairs" in str(captured["message"])

    @staticmethod
    def test_provider_checks_use_generic_pca_and_refactor_settings(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Generic settings affect the parallel-provider failure workflow."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(similarity_checks, "load_provider_embeddings", lambda *_: None)

        def fake_fit_pca(
            embeddings_cache: dict[str, list[float]],
            pca_variance: float,
            **_: object,
        ) -> tuple[None, int, bool]:
            captured["pca_variance"] = pca_variance
            assert embeddings_cache
            return None, 0, False

        def fake_report_violations(
            *_: object,
            refactor_index_threshold: float,
            refactor_index_top_n: int,
        ) -> None:
            captured["refactor_index_threshold"] = refactor_index_threshold
            captured["refactor_index_top_n"] = refactor_index_top_n

        monkeypatch.setattr(similarity_checks, "fit_pca", fake_fit_pca)
        monkeypatch.setattr(similarity_checks, "_find_violations", lambda *_: (["pair"], []))
        monkeypatch.setattr(similarity_checks, "_report_violations", fake_report_violations)

        functions = [
            FunctionInfo(
                name=f"function_{index}",
                file="module.py",
                start_line=index + 1,
                end_line=index + 1,
                loc=1,
                hash=f"ast-{index}",
                text="pass",
                text_hash=f"text-{index}",
            )
            for index in range(2)
        ]
        baselines: dict[str, object] = {
            "config": {
                "pca_variance_threshold": 0.75,
                "refactor_index_threshold": 12.0,
                "refactor_index_top_n": 3,
            },
            OPENAI_TEXT_PROVIDER.cache_key: {
                "text-0": [1.0, 0.0],
                "text-1": [0.0, 1.0],
            },
        }

        similarity_checks.run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=False,
            cached_only=True,
            provider=OPENAI_TEXT_PROVIDER,
            threshold_pair=0.98,
            threshold_neighbor=0.8,
            load_complexity_maps_fn=lambda: ({}, {}),
        )

        assert captured == {
            "pca_variance": 0.75,
            "refactor_index_threshold": 12.0,
            "refactor_index_top_n": 3,
        }

    @staticmethod
    def test_provider_specific_pca_setting_overrides_generic_setting() -> None:
        """Provider-specific PCA configuration has precedence over the generic key."""
        config: dict[str, object] = {
            "pca_variance_threshold": 0.75,
            OPENAI_TEXT_PROVIDER.pca_variance_key: 0.9,
        }

        assert (
            similarity_checks._extract_pca_variance(  # noqa: SLF001
                config,
                OPENAI_TEXT_PROVIDER,
            )
            == 0.9
        )


class TestPluginCollection:
    """Tests that the pytest plugin collects check modules."""

    @staticmethod
    def test_check_modules_collected() -> None:
        """The plugin's pytest_collect_file hook collects check modules.

        We verify this by checking that the check module stems are in
        the _CHECK_STEMS set, and that the _checks_dir() function returns
        a valid directory.
        """
        assert "code_complexity" in _CHECK_STEMS
        assert "code_similarity" in _CHECK_STEMS
        assert "comment_density" in _CHECK_STEMS
        assert "docstring_format" in _CHECK_STEMS
        assert "lint_typecheck" in _CHECK_STEMS

        checks = _checks_dir()
        assert checks is not None
        assert checks.is_dir()
        # The directory should contain the check modules
        for stem in _CHECK_STEMS:
            assert (checks / f"{stem}.py").exists()
