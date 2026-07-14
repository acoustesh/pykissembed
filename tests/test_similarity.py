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
    jina_combined_members,
    load_api_key_from_env,
)
from pykissembed.similarity.exclusions import is_excluded_pair
from pykissembed.similarity.jina_similarity import build_symmetrized_matrix
from pykissembed.similarity.populate_embeddings import (
    _JINA_TEXT_CFG,
    _jina_texts,
    _populate_combined,
    cli_provider_name,
)
from pykissembed.similarity.refactor_index import (
    compute_max_similarities,
    compute_refactor_indices,
    compute_similarity_indices,
    compute_similarity_matrix,
    get_refactor_priority_message,
)
from pykissembed.similarity.storage import (
    REGISTRY,
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

        assert is_excluded_pair(
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

        assert is_excluded_pair(
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

    @staticmethod
    def test_nested_pair_excluded_from_refactor_priority() -> None:
        """A nested class/method pair no longer drives the refactor recommendation."""
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
        test_class.embedding = [1.0, 0.0]
        test_method.embedding = [1.0, 0.0]
        functions = [test_class, test_method]

        # Negative proximity disables the containment exclusion, so the
        # identical embeddings still drive a recommendation.
        assert (
            get_refactor_priority_message(
                functions,
                {},
                {},
                threshold=1.0,
                class_function_proximity=-1,
            )
            is not None
        )
        # With the exclusion active the nested pair is ignored and nothing is
        # recommended from its structural similarity alone.
        assert (
            get_refactor_priority_message(
                functions,
                {},
                {},
                threshold=1.0,
                class_function_proximity=1,
            )
            is None
        )


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
        """Combined embedding length equals sum of all 14 member lengths."""
        inputs = [[1.0] for _ in range(14)]
        combined = compute_combined_embedding(*inputs)
        assert len(combined) == 14

    @staticmethod
    def test_is_normalized() -> None:
        """Combined embedding is L2-normalized."""
        inputs = [[3.0, 0.0], [4.0, 0.0]] + [[0.0, 0.0] for _ in range(12)]
        combined = compute_combined_embedding(*inputs)
        norm = sum(c**2 for c in combined) ** 0.5
        assert norm == pytest.approx(1.0)

    @staticmethod
    def test_jina_members_occupy_final_four_segments() -> None:
        """The four Jina members occupy the final four segments of the concat."""
        inputs = [[0.0] for _ in range(10)] + [[3.0], [4.0], [6.0], [8.0]]
        combined = compute_combined_embedding(*inputs)
        # Whole vector is L2-normalized: only the last four entries are non-zero,
        # with magnitude sqrt(9+16+36+64) = sqrt(125).
        norm = 125.0**0.5
        expected = [0.0] * 10 + [3.0 / norm, 4.0 / norm, 6.0 / norm, 8.0 / norm]
        assert combined == pytest.approx(expected)


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


class TestEmbeddingRegistry:
    """Tests for the built-in similarity provider registry."""

    @staticmethod
    def test_qwen_variants_participate_in_combined_embeddings() -> None:
        """Both Qwen cache variants are base-provider dependencies."""
        assert REGISTRY.by_cache_key("qwen_text_embeddings").hash_type is HashType.TEXT
        assert REGISTRY.by_cache_key("qwen_ast_embeddings").hash_type is HashType.AST
        assert REGISTRY.combined_dependencies[-2:] == [
            "qwen_text_embeddings",
            "qwen_ast_embeddings",
        ]
        assert len(REGISTRY.base_providers) == 10

    @staticmethod
    def test_jina_raw_caches_are_standalone_not_cosine_base() -> None:
        """The four Jina query/passage caches are standalone, excluded from cosine base."""
        expected = [
            "jina_text_query_embeddings",
            "jina_text_passage_embeddings",
            "jina_ast_query_embeddings",
            "jina_ast_passage_embeddings",
        ]
        assert REGISTRY.standalone_dependencies == expected
        for key in expected:
            entry = REGISTRY.by_cache_key(key)
            assert entry.standalone is True
            assert key in REGISTRY.files  # still persisted round-trip
        # Standalone caches do not inflate the cosine base / combined concat.
        assert len(REGISTRY.base_providers) == 10
        assert all(key not in REGISTRY.combined_dependencies for key in expected)
        assert REGISTRY.by_cache_key("jina_text_query_embeddings").hash_type is HashType.TEXT
        assert REGISTRY.by_cache_key("jina_ast_query_embeddings").hash_type is HashType.AST


def _jina_func(content_hash: str) -> FunctionInfo:
    """Build a minimal FunctionInfo keyed by *content_hash* for Jina matrix tests.

    Returns
    -------
    FunctionInfo
        A minimal function whose ``hash`` field equals *content_hash*.
    """
    return FunctionInfo(
        name=content_hash,
        file="f.py",
        start_line=1,
        end_line=2,
        loc=1,
        hash=content_hash,
        text="x",
    )


class TestJinaCombinedMembers:
    """Tests for jina_combined_members (Combined representation of a Jina variant)."""

    @staticmethod
    def test_returns_two_unit_norm_orderings() -> None:
        """Both members are unit-norm and are the (Q,P)/(P,Q) orderings of each other."""
        qp, pq = jina_combined_members([1.0, 0.0], [0.0, 1.0])
        assert len(qp) == len(pq) == 4
        assert sum(c**2 for c in qp) ** 0.5 == pytest.approx(1.0)
        assert sum(c**2 for c in pq) ** 0.5 == pytest.approx(1.0)
        inv = 1.0 / 2.0**0.5
        assert qp == pytest.approx([inv, 0.0, 0.0, inv])
        assert pq == pytest.approx([0.0, inv, inv, 0.0])

    @staticmethod
    def test_zero_vectors_returned_unnormalized() -> None:
        """A zero concat is returned unchanged (no divide-by-zero)."""
        qp, pq = jina_combined_members([0.0, 0.0], [0.0, 0.0])
        assert qp == [0.0, 0.0, 0.0, 0.0]
        assert pq == [0.0, 0.0, 0.0, 0.0]


class TestBuildSymmetrizedMatrix:
    """Tests for the Jina symmetrized query/passage similarity matrix."""

    @staticmethod
    def test_symmetric_zero_diagonal_and_cross_score() -> None:
        """S is symmetric, zero-diagonal, and equals (cos(Qi,Pj)+cos(Qj,Pi))/2."""
        functions = [_jina_func("a"), _jina_func("b")]
        query_cache = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
        passage_cache = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        matrix = build_symmetrized_matrix(functions, query_cache, passage_cache, "hash")
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == pytest.approx(0.0)
        assert matrix[1, 1] == pytest.approx(0.0)
        # cos(Qa,Pb)=0, cos(Qb,Pa)=1 -> 0.5
        assert matrix[0, 1] == pytest.approx(0.5)
        assert matrix[1, 0] == pytest.approx(matrix[0, 1])

    @staticmethod
    def test_missing_vectors_contribute_zero() -> None:
        """A function absent from either cache contributes zero similarity."""
        functions = [_jina_func("a"), _jina_func("b")]
        matrix = build_symmetrized_matrix(functions, {"a": [1.0, 0.0]}, {"a": [1.0, 0.0]}, "hash")
        assert matrix[0, 1] == pytest.approx(0.0)


class TestJinaTexts:
    """Tests for the Jina nl2code query/passage input-text selection."""

    @staticmethod
    def test_nl2code_uses_docstring_with_signature_fallback() -> None:
        """nl2code query is the docstring, falling back to the signature text."""
        documented = FunctionInfo(
            name="a",
            file="f.py",
            start_line=1,
            end_line=2,
            loc=1,
            hash="a",
            text="x",
            text_for_embedding="def a(): ...",
            ast_text="def a():\n    return 1",
            docstring="Do the thing.",
        )
        undocumented = FunctionInfo(
            name="b",
            file="f.py",
            start_line=1,
            end_line=2,
            loc=1,
            hash="b",
            text="x",
            text_for_embedding="def b(): ...",
            ast_text="def b():\n    return 2",
            docstring="",
        )
        query_texts, passage_texts = _jina_texts([documented, undocumented], _JINA_TEXT_CFG)
        assert query_texts == ["Do the thing.", "def b(): ..."]
        assert passage_texts == [documented.ast_text, undocumented.ast_text]


def _combined_member_func(text_hash: str, ast_hash: str) -> FunctionInfo:
    """Build a minimal FunctionInfo carrying both combined-member hash keys.

    Returns
    -------
    FunctionInfo
        A minimal function keyed by *text_hash* and *ast_hash*.
    """
    return FunctionInfo(
        name="f",
        file="f.py",
        start_line=1,
        end_line=2,
        loc=1,
        hash=ast_hash,
        text="x",
        text_hash=text_hash,
    )


def _complete_member_baselines(text_hash: str, ast_hash: str) -> dict[str, object]:
    """Build baselines where every combined member cache covers one function.

    Returns
    -------
    dict[str, object]
        Baselines with empty ``function_hashes`` and all 14 member caches
        holding a vector under the hash matching each member's hash type.
    """
    baselines: dict[str, object] = {"function_hashes": {}}
    for cache_key in REGISTRY.combined_dependencies + REGISTRY.standalone_dependencies:
        entry = REGISTRY.by_cache_key(cache_key)
        content_hash = text_hash if entry.hash_type is HashType.TEXT else ast_hash
        baselines[cache_key] = {content_hash: [1.0, 0.0]}
    return baselines


class TestPopulateCombined:
    """Tests for the combined populate step (also used by test auto-population)."""

    @staticmethod
    def test_builds_from_live_functions_without_prior_function_hashes() -> None:
        """Combined embeddings are built for the passed functions even when function_hashes starts empty."""
        baselines = _complete_member_baselines("text1", "ast1")
        built = _populate_combined(baselines, [_combined_member_func("text1", "ast1")])
        assert built == 1
        combined = baselines[REGISTRY.combined.cache_key]
        assert isinstance(combined, dict)
        assert "text1" in combined

    @staticmethod
    def test_records_function_hashes_for_live_functions() -> None:
        """The live functions' hash pairs are recorded so save_baselines persists them."""
        baselines = _complete_member_baselines("text1", "ast1")
        _populate_combined(baselines, [_combined_member_func("text1", "ast1")])
        assert baselines["function_hashes"] == {
            "f.py:f:1": {"hash": "ast1", "text_hash": "text1"},
        }


class TestCliProviderName:
    """Tests for the cache-key to CLI provider-name mapping."""

    @staticmethod
    def test_maps_cosine_and_raw_jina_cache_keys() -> None:
        """Cosine keys drop the suffix; raw Jina keys fold onto their variant."""
        assert cli_provider_name("qwen_text_embeddings") == "qwen-text"
        assert cli_provider_name("jina_text_query_embeddings") == "jina-text"
        assert cli_provider_name("jina_ast_passage_embeddings") == "jina-ast"
        assert cli_provider_name("combined_embeddings") == "combined"


class TestSkipMissingEmbeddings:
    """Tests for the missing-embeddings skip diagnosis."""

    @staticmethod
    def _skip_message(
        baselines: dict[str, object],
        uncached: list[FunctionInfo],
        provider: ProviderEntry,
        total: int,
    ) -> str:
        """Run the skip helper and capture its message.

        Returns
        -------
        str
            The pytest.skip message raised by the helper.
        """
        with pytest.raises(pytest.skip.Exception) as excinfo:
            similarity_checks._skip_missing_embeddings(  # noqa: SLF001
                baselines,
                uncached,
                provider,
                total=total,
            )
        return str(excinfo.value)

    def test_all_functions_missing_omits_function_list(self) -> None:
        """When every function is uncached, say so instead of listing names."""
        uncached = [_combined_member_func(f"t{i}", f"a{i}") for i in range(2)]
        message = self._skip_message({}, uncached, OPENAI_TEXT_PROVIDER, total=2)
        assert "All 2 functions lack cached OpenAI-Text embeddings" in message
        assert "f.py" not in message
        assert "populate-embeddings --provider openai-text" in message

    def test_partial_missing_lists_functions(self) -> None:
        """A partial gap names the affected functions."""
        uncached = [_combined_member_func("t1", "a1")]
        message = self._skip_message({}, uncached, OPENAI_TEXT_PROVIDER, total=5)
        assert "1 of 5 functions lack cached OpenAI-Text embeddings" in message
        assert "f.py:f" in message

    def test_member_gaps_name_the_providers_to_populate(self) -> None:
        """Combined skips point at the member providers that are missing."""
        baselines = _complete_member_baselines("t1", "a1")
        baselines["qwen_text_embeddings"] = {}
        baselines["jina_ast_query_embeddings"] = {}
        uncached = [_combined_member_func("t1", "a1")]
        message = self._skip_message(
            baselines,
            uncached,
            similarity_checks.COMBINED_PROVIDER,
            total=5,
        )
        assert "qwen-text: 1" in message
        assert "jina-ast: 1" in message
        assert "populate-embeddings --provider <name>" in message

    def test_complete_members_suggest_rerun_without_cached_only(self) -> None:
        """With all members cached, the skip explains combined rebuilds locally."""
        baselines = _complete_member_baselines("t1", "a1")
        uncached = [_combined_member_func("t1", "a1")]
        message = self._skip_message(
            baselines,
            uncached,
            similarity_checks.COMBINED_PROVIDER,
            total=5,
        )
        assert "--cached-only" in message


class TestAutoPopulateMissingEmbeddings:
    """Tests for automatic population when cache-only mode is disabled."""

    @staticmethod
    def test_provider_workflow_populates_missing_embeddings(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-Jina provider populates missing cache entries by default."""
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
        baselines: dict[str, object] = {OPENAI_TEXT_PROVIDER.cache_key: {}}
        requested_providers: list[str] = []
        populated_counts: list[int] = []
        saved: list[dict[str, object]] = []

        def fake_populate(
            target_baselines: dict[str, object],
            target_functions: list[FunctionInfo],
        ) -> int:
            populated_counts.append(len(target_functions))
            target_baselines[OPENAI_TEXT_PROVIDER.cache_key] = {
                function.text_hash: [float(index), float(1 - index)]
                for index, function in enumerate(target_functions)
            }
            return len(target_functions)

        def fake_get_provider_populator(name: str):
            requested_providers.append(name)
            return fake_populate

        monkeypatch.setattr(similarity_checks, "load_provider_embeddings", lambda *_: None)
        monkeypatch.setattr(
            similarity_checks,
            "get_provider_populator",
            fake_get_provider_populator,
        )
        monkeypatch.setattr(similarity_checks, "save_baselines", saved.append)
        monkeypatch.setattr(
            similarity_checks,
            "fit_pca",
            lambda *_args, **_kwargs: (None, 0, False),
        )

        similarity_checks.run_provider_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=False,
            cached_only=False,
            provider=OPENAI_TEXT_PROVIDER,
            threshold_pair=1.1,
            threshold_neighbor=1.1,
            load_complexity_maps_fn=lambda: ({}, {}),
        )

        assert requested_providers == ["openai-text"]
        assert populated_counts == [2]
        assert saved == [baselines]
        assert all(function.embedding is not None for function in functions)

    @staticmethod
    def test_jina_workflow_populates_missing_embeddings(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Jina provider populates both missing raw caches by default."""
        provider = similarity_checks.JINA_TEXT_PROVIDER
        query_key = f"{provider.name}_query_embeddings"
        passage_key = f"{provider.name}_passage_embeddings"
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
        baselines: dict[str, object] = {query_key: {}, passage_key: {}}
        requested_providers: list[str] = []
        populated_counts: list[int] = []
        saved: list[dict[str, object]] = []

        def fake_populate(
            target_baselines: dict[str, object],
            target_functions: list[FunctionInfo],
        ) -> int:
            populated_counts.append(len(target_functions))
            target_baselines[query_key] = {
                function.text_hash: [float(index), float(1 - index)]
                for index, function in enumerate(target_functions)
            }
            target_baselines[passage_key] = {
                function.text_hash: [float(index), float(1 - index)]
                for index, function in enumerate(target_functions)
            }
            return len(target_functions)

        def fake_get_provider_populator(name: str):
            requested_providers.append(name)
            return fake_populate

        monkeypatch.setattr(similarity_checks, "load_provider_embeddings", lambda *_: None)
        monkeypatch.setattr(
            similarity_checks,
            "get_provider_populator",
            fake_get_provider_populator,
        )
        monkeypatch.setattr(similarity_checks, "save_baselines", saved.append)

        similarity_checks.run_jina_similarity_checks(
            baselines=baselines,
            functions=functions,
            update_baselines=False,
            cached_only=False,
            provider=provider,
            threshold_pair=1.1,
            threshold_neighbor=1.1,
            load_complexity_maps_fn=lambda: ({}, {}),
        )

        assert requested_providers == ["jina-text"]
        assert populated_counts == [2]
        assert saved == [baselines]


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

        assert similarity_checks._extract_pca_variance(  # noqa: SLF001
            config,
            OPENAI_TEXT_PROVIDER,
        ) == pytest.approx(0.9)


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
