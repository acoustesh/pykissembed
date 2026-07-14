"""Structural exclusions for similarity comparisons.

Shared between the pair/neighbor violation check
(``pykissembed.similarity.checks``) and the refactor-priority recommendation
(``pykissembed.similarity.refactor_index``). Both consult the same rule so a
method is never treated as a genuine near-duplicate of the class that
encloses it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pykissembed.config import get_config

if TYPE_CHECKING:
    from pykissembed.similarity.types import FunctionInfo

_EXCLUSION_PAIR_SIZE = 2


def is_excluded_pair(
    func_a: FunctionInfo,
    func_b: FunctionInfo,
    excluded_file_pairs: list[list[str]],
    excluded_function_pairs: list[list[str]],
    class_function_proximity: int = 0,
) -> bool:
    """Check whether a pair is configured or structurally excluded.

    Returns
    -------
    bool
        True if the pair is excluded, False otherwise.
    """
    # Check file-level exclusions
    for pair in excluded_file_pairs:
        if len(pair) != _EXCLUSION_PAIR_SIZE:
            continue
        pattern_a, pattern_b = pair
        if (pattern_a in func_a.file and pattern_b in func_b.file) or (
            pattern_b in func_a.file and pattern_a in func_b.file
        ):
            return True

    # Check function-level exclusions
    func_a_key = f"{func_a.file}:{func_a.name}"
    func_b_key = f"{func_b.file}:{func_b.name}"
    for pair in excluded_function_pairs:
        if len(pair) != _EXCLUSION_PAIR_SIZE:
            continue
        pattern_a, pattern_b = pair
        if (pattern_a in func_a_key and pattern_b in func_b_key) or (
            pattern_b in func_a_key and pattern_a in func_b_key
        ):
            return True

    # A class naturally contains its methods' source. Comparing the class
    # block with one of those methods produces a structural false positive;
    # nearby class/function pairs have the same property in small test files.
    is_class_a = func_a.text.lstrip().startswith(("class ", "class\t"))
    is_class_b = func_b.text.lstrip().startswith(("class ", "class\t"))
    if func_a.file != func_b.file or is_class_a == is_class_b:
        return False

    first, second = sorted((func_a, func_b), key=lambda func: func.start_line)
    if first.end_line >= second.start_line:
        return class_function_proximity >= 0

    source_file = Path(first.file)
    if not source_file.is_absolute():
        source_file = get_config().root / source_file
    try:
        source_lines = source_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False

    code_lines_between = sum(
        bool(stripped := line.strip()) and not stripped.startswith("#")
        for line in source_lines[first.end_line : second.start_line - 1]
    )
    return code_lines_between <= class_function_proximity
