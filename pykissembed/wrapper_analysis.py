"""Static analysis for exact pass-through wrapper proliferation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from pykissembed.paths import iter_py_files, warn_non_utf8

if TYPE_CHECKING:
    from pathlib import Path

_AUTO_EXEMPT_DECORATOR_TAILS = frozenset(
    {
        "abstractmethod",
        "cached_property",
        "overload",
        "override",
        "property",
    },
)
_AUTO_EXEMPT_DECORATOR_NAMES = frozenset(
    {
        "app.callback",
        "app.command",
        "pytest.fixture",
        "pytest.hookimpl",
        "typer.callback",
        "typer.command",
    },
)


@dataclass(frozen=True, slots=True)
class WrapperCandidate:
    """An exact pass-through wrapper with its project-wide call-site count."""

    identifier: str
    name: str
    line: int
    call_count: int


def parse_source_files(paths: list[Path]) -> list[tuple[Path, ast.Module]]:
    """Parse configured Python files, skipping unreadable or invalid source.

    Returns
    -------
    list[tuple[Path, ast.Module]]
        Parsed files paired with their AST modules, sorted by path.
    """
    # Resolve first so overlapping configured directories cannot parse the
    # same source file twice; sorting keeps later diagnostics deterministic.
    files = sorted(
        {
            file_path.resolve()
            for base_dir in paths
            for file_path in iter_py_files(base_dir)
        },
    )
    modules: list[tuple[Path, ast.Module]] = []
    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            warn_non_utf8(file_path, exc)
            continue
        try:
            modules.append((file_path, ast.parse(source, filename=str(file_path))))
        except SyntaxError:
            continue
    return modules


def find_wrapper_candidates(
    modules: list[tuple[Path, ast.Module]],
    *,
    root: Path,
    wrapper_exclude: list[str],
    wrapper_exempt_decorators: list[str],
) -> list[WrapperCandidate]:
    """Return non-exempt exact forwarding wrappers and their call counts.

    Returns
    -------
    list[WrapperCandidate]
        Candidates sorted by their stable path-and-qualified-name identifier.
    """
    call_counts = _call_counts(modules)
    candidates: list[WrapperCandidate] = []
    for file_path, tree in modules:
        # Qualified identifiers include enclosing functions and classes, so
        # index each module's upward AST links before visiting its functions.
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            candidate = _wrapper_candidate(
                node,
                file_path=file_path,
                root=root,
                parents=parents,
                call_counts=call_counts,
                wrapper_exclude=wrapper_exclude,
                wrapper_exempt_decorators=wrapper_exempt_decorators,
            )
            if candidate is not None:
                candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: candidate.identifier)


def _wrapper_candidate(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    file_path: Path,
    root: Path,
    parents: dict[ast.AST, ast.AST],
    call_counts: dict[str, int],
    wrapper_exclude: list[str],
    wrapper_exempt_decorators: list[str],
) -> WrapperCandidate | None:
    """Create a candidate when *node* is a non-exempt exact forwarder.

    Returns
    -------
    WrapperCandidate | None
        The candidate with its call count, or ``None`` when it is not subject
        to the wrapper rule.
    """
    call = _forwarding_call(node)
    if call is None:
        return None
    identifier = _wrapper_identifier(node, file_path=file_path, root=root, parents=parents)
    if _is_exempt_wrapper(
        node,
        identifier=identifier,
        wrapper_exclude=wrapper_exclude,
        wrapper_exempt_decorators=wrapper_exempt_decorators,
    ):
        return None
    return WrapperCandidate(
        identifier=identifier,
        name=node.name,
        line=node.lineno,
        call_count=call_counts.get(node.name, 0),
    )


def _forwarding_call(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Call | None:
    """Return the exact forwarding call in *node*, if its body has one.

    Returns
    -------
    ast.Call | None
        The forwarded call, or ``None`` when *node* has another body shape.
    """
    body = _executable_body(node)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    return_value = body[0].value
    if isinstance(return_value, ast.Await):
        # Awaiting remains a direct forward only for an async wrapper; a
        # synchronous function cannot validly contain this AST shape.
        if not isinstance(node, ast.AsyncFunctionDef):
            return None
        return_value = return_value.value
    if not isinstance(return_value, ast.Call) or _call_terminal_name(return_value.func) is None:
        return None
    return return_value if _is_unchanged_forwarding(node, return_value) else None


def _executable_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return *node*'s body after removing its leading docstring statement.

    Returns
    -------
    list[ast.stmt]
        The executable statements in their original order.
    """
    return node.body[1:] if ast.get_docstring(node) is not None else node.body


def _is_unchanged_forwarding(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
) -> bool:
    """Return whether *call* forwards every parameter of *node* unchanged.

    Returns
    -------
    bool
        ``True`` when positional, variadic, keyword-only, and keyword
        parameters are forwarded without transformation or reordering.
    """
    positional_names = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
    receiver_name = _receiver_name(call.func)
    if positional_names and positional_names[0] == receiver_name:
        # ``self.method(...)`` and ``cls.method(...)`` consume the receiver
        # through the call target instead of forwarding it as an argument.
        positional_names = positional_names[1:]
    return _matches_positional_arguments(call.args, positional_names, node.args.vararg) and (
        _matches_keyword_arguments(call.keywords, node.args.kwonlyargs, node.args.kwarg)
    )


def _receiver_name(call_target: ast.expr) -> str | None:
    """Return the receiver name for an attribute call target.

    Returns
    -------
    str | None
        The receiver variable name, or ``None`` for non-name receivers.
    """
    if isinstance(call_target, ast.Attribute) and isinstance(call_target.value, ast.Name):
        return call_target.value.id
    return None


def _matches_positional_arguments(
    values: list[ast.expr],
    parameter_names: list[str],
    vararg: ast.arg | None,
) -> bool:
    """Return whether positional call arguments forward the declared parameters.

    Returns
    -------
    bool
        ``True`` when positional parameters and an optional ``*args`` match
        their declared names exactly and in order.
    """
    expected_count = len(parameter_names) + int(vararg is not None)
    if len(values) != expected_count:
        return False
    fixed_values = values[: len(parameter_names)]
    # Pairing with ``strict=True`` rejects an arity mismatch as well as a
    # reordered or renamed positional argument.
    if not all(map(_is_parameter_name, fixed_values, parameter_names, strict=True)):
        return False
    if vararg is None:
        return True
    variadic_value = values[-1]
    return isinstance(variadic_value, ast.Starred) and _is_parameter_name(
        variadic_value.value,
        vararg.arg,
    )


def _matches_keyword_arguments(
    keywords: list[ast.keyword],
    keyword_only: list[ast.arg],
    kwarg: ast.arg | None,
) -> bool:
    """Return whether keyword call arguments forward declared parameters.

    Returns
    -------
    bool
        ``True`` when keyword-only parameters and an optional ``**kwargs``
        are forwarded exactly once and in declaration order.
    """
    expected_count = len(keyword_only) + int(kwarg is not None)
    if len(keywords) != expected_count:
        return False
    fixed_keywords = keywords[: len(keyword_only)]
    if not all(
        keyword.arg == argument.arg and _is_parameter_name(keyword.value, argument.arg)
        for keyword, argument in zip(fixed_keywords, keyword_only, strict=True)
    ):
        return False
    if kwarg is None:
        return True
    variadic_keyword = keywords[-1]
    # ``arg is None`` is the AST representation of ``**kwargs``; named
    # keyword nodes would transform the wrapper's forwarding convention.
    return variadic_keyword.arg is None and _is_parameter_name(variadic_keyword.value, kwarg.arg)


def _is_parameter_name(value: ast.expr, parameter_name: str) -> bool:
    """Return whether *value* is a direct read of *parameter_name*.

    Returns
    -------
    bool
        ``True`` when *value* is a name expression with the expected name.
    """
    return isinstance(value, ast.Name) and value.id == parameter_name


def _call_counts(modules: list[tuple[Path, ast.Module]]) -> dict[str, int]:
    """Count direct and attribute calls by their terminal static name.

    Returns
    -------
    dict[str, int]
        Project-wide call counts keyed by the terminal callee name.
    """
    counts: dict[str, int] = {}
    for _, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Count by terminal syntax because this static pass does not
                # resolve imports, attributes, or lexical scopes.
                name = _call_terminal_name(node.func)
                if name is not None:
                    counts[name] = counts.get(name, 0) + 1
    return counts


def _call_terminal_name(call_target: ast.expr) -> str | None:
    """Return the terminal name of a direct or attribute call target.

    Returns
    -------
    str | None
        The callee name for ``name()`` and ``object.name()``, or ``None``.
    """
    if isinstance(call_target, ast.Name):
        return call_target.id
    if isinstance(call_target, ast.Attribute):
        return call_target.attr
    return None


def _parent_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Build an AST child-to-parent lookup table.

    Returns
    -------
    dict[ast.AST, ast.AST]
        Every AST child mapped to its direct parent node.
    """
    # Python AST nodes have no parent pointers, but qualified names must walk
    # outward through every containing class and function scope.
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _wrapper_identifier(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    file_path: Path,
    root: Path,
    parents: dict[ast.AST, ast.AST],
) -> str:
    """Return a stable relative-path and qualified-name wrapper identifier.

    Returns
    -------
    str
        An identifier in the form ``relative/path.py:QualifiedName``.
    """
    try:
        relative_path = file_path.relative_to(root).as_posix()
    except ValueError:
        relative_path = str(file_path)
    return f"{relative_path}:{_qualified_name(node, parents)}"


def _qualified_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parents: dict[ast.AST, ast.AST],
) -> str:
    """Return *node*'s class-and-function-qualified source name.

    Returns
    -------
    str
        The nested class/function name joined with dots.
    """
    names = [node.name]
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(parent.name)
        parent = parents.get(parent)
    return ".".join(reversed(names))


def _is_exempt_wrapper(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    identifier: str,
    wrapper_exclude: list[str],
    wrapper_exempt_decorators: list[str],
) -> bool:
    """Return whether a forwarding wrapper is intentionally exempt.

    Returns
    -------
    bool
        ``True`` for dunder methods, configured identifiers, or exempt
        decorators.
    """
    return (
        (node.name.startswith("__") and node.name.endswith("__"))
        or _matches_any(identifier, wrapper_exclude)
        or _has_exempt_decorator(node, wrapper_exempt_decorators)
    )


def _has_exempt_decorator(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    wrapper_exempt_decorators: list[str],
) -> bool:
    """Return whether *node* has an automatic or configured exempt decorator.

    Returns
    -------
    bool
        ``True`` when a decorator's full or terminal name is exempt.
    """
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name is None:
            continue
        terminal_name = name.rsplit(".", maxsplit=1)[-1]
        if terminal_name in _AUTO_EXEMPT_DECORATOR_TAILS or name in _AUTO_EXEMPT_DECORATOR_NAMES:
            return True
        if _matches_any(name, wrapper_exempt_decorators) or _matches_any(
            terminal_name,
            wrapper_exempt_decorators,
        ):
            return True
    return False


def _decorator_name(decorator: ast.expr) -> str | None:
    """Return the syntactic dotted name for a decorator expression.

    Returns
    -------
    str | None
        The decorator's dotted name, or ``None`` when it is dynamic.
    """
    # Parameterized decorators are calls, so recurse into their callee to
    # compare the same static dotted name used for unparameterized forms.
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    if isinstance(decorator, ast.Name):
        return decorator.id
    if not isinstance(decorator, ast.Attribute):
        return None
    prefix = _decorator_name(decorator.value)
    return f"{prefix}.{decorator.attr}" if prefix is not None else decorator.attr


def _matches_any(value: str, patterns: list[str]) -> bool:
    """Return whether *value* matches any configured glob pattern.

    Returns
    -------
    bool
        ``True`` when at least one pattern matches *value*.
    """
    return any(fnmatchcase(value, pattern) for pattern in patterns)
