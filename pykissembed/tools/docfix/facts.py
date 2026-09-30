"""Facts about a definition that its source code proves.

Everything here is derived from the AST alone: parameter names and
annotations, whether the body returns a value or yields, and which explicit
``raise`` statements can escape to the caller. Only a definition's own scope
counts; nested functions, classes and lambdas are separate symbols.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from itertools import starmap

from pykissembed.wrapper_analysis import decorator_name

type DefNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

# Return annotations meaning "no value to document".
_NO_VALUE_ANNOTATIONS = frozenset({"None", "NoReturn", "Never", "typing.NoReturn", "typing.Never"})
# Handlers that swallow (practically) everything raised inside the try.
_CATCH_ALL = frozenset({"Exception", "BaseException"})


@dataclass(frozen=True, slots=True)
class Param:
    """One caller-supplied parameter, stars kept on variadics."""

    name: str
    annotation: str


@dataclass(slots=True)
class Facts:
    """What the code proves about one function or class.

    Attributes
    ----------
    name : str
        Definition name.
    kind : str
        ``"function"`` or ``"class"``.
    params : list[Param]
        Caller-supplied parameters in signature order; for a class, those
        of its ``__init__``.
    implicit : str
        Name of the implicitly bound first parameter (``self``/``cls``), or
        ``""`` for plain functions and staticmethods.
    returns : str
        Return annotation text, or ``""`` when unannotated.
    returns_value : bool
        Whether the own scope has a ``return`` with a non-``None`` value.
    is_generator : bool
        Whether the own scope contains ``yield`` or ``yield from``.
    is_property : bool
        Whether the function is a (cached) property.
    is_overload : bool
        Whether the function is an ``@overload`` stub.
    raises : list[str]
        Exception names explicitly raised and not caught locally.
    attributes : list[Param]
        Public instance attributes (classes only).
    signature_known : bool
        ``False`` for a class without an explicit ``__init__``.
    """

    name: str
    kind: str
    params: list[Param] = field(default_factory=list)
    implicit: str = ""
    returns: str = ""
    returns_value: bool = False
    is_generator: bool = False
    is_property: bool = False
    is_overload: bool = False
    raises: list[str] = field(default_factory=list)
    attributes: list[Param] = field(default_factory=list)
    signature_known: bool = True

    @property
    def documents_return(self) -> bool:
        """Whether a ``Returns`` section is required.

        Returns
        -------
        bool
            ``True`` for a non-generator that returns a value, per its
            annotation or, without one, its ``return`` statements.
        """
        # Generators document Yields; constructors never return a value.
        if self.kind != "function" or self.is_generator or self.name == "__init__":
            return False
        # An explicit annotation is the contract, even for a stub body.
        if self.returns:
            return self.returns not in _NO_VALUE_ANNOTATIONS
        return self.returns_value


def _unparse(node: ast.expr | None) -> str:
    """Return source text for an optional annotation.

    Parameters
    ----------
    node : ast.expr | None
        Annotation expression.

    Returns
    -------
    str
        The unparsed text, or ``""`` when there is no annotation.
    """
    return "" if node is None else ast.unparse(node)


def own_scope(node: ast.AST) -> list[ast.AST]:
    """Return every node in *node*'s body without entering nested scopes.

    Parameters
    ----------
    node : ast.AST
        A function or class definition.

    Returns
    -------
    list[ast.AST]
        Descendants that execute in *node*'s own frame.
    """
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(getattr(node, "body", []))
    # Nested definitions are collected but not entered: their returns and
    # yields belong to them, not to the enclosing definition.
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    while stack:
        current = stack.pop()
        found.append(current)
        if not isinstance(current, scopes):
            stack.extend(ast.iter_child_nodes(current))
    return found


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, implicit: str) -> list[Param]:
    """Return the caller-supplied parameters in signature order.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function.
    implicit : str
        ``"self"``/``"cls"`` when the first positional parameter is bound
        implicitly, else ``""``.

    Returns
    -------
    list[Param]
        Parameters with ``*``/``**`` prefixes on variadics.
    """
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    # self/cls are bound implicitly and are never documented (numpydoc).
    if implicit and positional:
        positional = positional[1:]
    params = [Param(a.arg, _unparse(a.annotation)) for a in positional]
    if args.vararg is not None:
        params.append(Param("*" + args.vararg.arg, _unparse(args.vararg.annotation)))
    params.extend(Param(a.arg, _unparse(a.annotation)) for a in args.kwonlyargs)
    if args.kwarg is not None:
        params.append(Param("**" + args.kwarg.arg, _unparse(args.kwarg.annotation)))
    return params


def _implicit_param(node: ast.FunctionDef | ast.AsyncFunctionDef, *, in_class: bool) -> str:
    """Name the implicitly bound first parameter of a method, if any.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function.
    in_class : bool
        Whether the function is defined directly in a class body.

    Returns
    -------
    str
        The first positional parameter's name for instance and class
        methods, or ``""`` for staticmethods and plain functions.
    """
    positional = [*node.args.posonlyargs, *node.args.args]
    names = {decorator_name(d) for d in node.decorator_list}
    # The name is taken from the code rather than assumed to be "self".
    if not in_class or not positional or "staticmethod" in names:
        return ""
    return positional[0].arg


def _exception_name(node: ast.expr | None) -> str:
    """Return the class name an exception expression raises.

    Parameters
    ----------
    node : ast.expr | None
        The ``raise`` operand.

    Returns
    -------
    str
        ``"ValueError"`` for ``ValueError(...)`` or ``ValueError``; dotted
        names are kept. ``""`` for re-raised variables and dynamic operands.
    """
    target = node.func if isinstance(node, ast.Call) else node
    # "raise err" re-raises a variable whose class is unknown; exception
    # classes are capitalised by convention, variables are not.
    if isinstance(target, ast.Name):
        return target.id if target.id[:1].isupper() else ""
    if isinstance(target, ast.Attribute):
        return ast.unparse(target)
    return ""


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    """Return the exception names one ``except`` clause catches.

    Parameters
    ----------
    handler : ast.ExceptHandler
        The clause.

    Returns
    -------
    set[str]
        Caught names; a bare ``except:`` catches ``BaseException``.
    """
    # A bare "except:" catches everything, like "except BaseException:".
    if handler.type is None:
        return {"BaseException"}
    items = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {ast.unparse(item) for item in items}


def is_caught(name: str, caught: frozenset[str]) -> bool:
    """Return whether an exception *name* is swallowed by a *caught* set.

    Parameters
    ----------
    name : str
        Raised exception name, possibly dotted.
    caught : frozenset[str]
        Names caught by the enclosing handlers.

    Returns
    -------
    bool
        ``True`` on a name match or a catch-all, or when both are builtin
        exceptions related by inheritance.
    """
    short = name.rsplit(".", 1)[-1]
    if caught & _CATCH_ALL or short in {c.rsplit(".", 1)[-1] for c in caught}:
        return True
    # Inheritance is only known for builtins (KeyError is a LookupError);
    # a project exception caught by its base class is assumed to escape.
    raised = getattr(builtins, short, None)
    if not isinstance(raised, type):
        return False
    for other in caught:
        base = getattr(builtins, other, None)
        if isinstance(base, type) and issubclass(raised, base):
            return True
    return False


def _nested_statements(statement: ast.stmt) -> list[ast.stmt]:
    """Return the statements of every block nested directly in *statement*.

    Parameters
    ----------
    statement : ast.stmt
        A compound statement such as ``if``, ``for``, ``with`` or ``match``.

    Returns
    -------
    list[ast.stmt]
        Statements from its ``body``, ``orelse``, ``finalbody`` and ``case``
        blocks; empty for a simple statement.
    """
    blocks: list[object] = [
        getattr(statement, name, None) for name in ("body", "orelse", "finalbody")
    ]
    blocks.extend(case.body for case in getattr(statement, "cases", []))
    # An ast.IfExp also has a "body", but it is an expression, not statements.
    return [
        item
        for block in blocks
        if isinstance(block, list)
        for item in block
        if isinstance(item, ast.stmt)
    ]


def _rethrown(names: set[str]) -> str:
    """Return the exception a bare ``raise`` re-raises inside a handler.

    Parameters
    ----------
    names : set[str]
        Names the ``except`` clause catches.

    Returns
    -------
    str
        The single caught name, or ``""`` for a tuple or a catch-all, where
        the escaping class cannot be named.
    """
    return next(iter(names)) if len(names) == 1 and not names & _CATCH_ALL else ""


def _try_blocks(
    statement: ast.Try | ast.TryStar, caught: frozenset[str], active: str
) -> list[tuple[list[ast.stmt], frozenset[str], str]]:
    """Split a ``try`` statement into blocks with the context each runs in.

    Parameters
    ----------
    statement : ast.Try | ast.TryStar
        The ``try`` statement.
    caught : frozenset[str]
        Names caught by enclosing ``try`` blocks.
    active : str
        The exception a bare ``raise`` re-raises outside the handlers.

    Returns
    -------
    list[tuple[list[ast.stmt], frozenset[str], str]]
        ``(statements, caught, active)`` for the body (which the handlers
        guard), each handler (where a bare ``raise`` re-raises what it
        caught), and the ``else``/``finally`` blocks (unguarded).
    """
    handled = [(handler.body, _handler_names(handler)) for handler in statement.handlers]
    guarded = caught | frozenset(name for _, names in handled for name in names)
    blocks = [(statement.body, guarded, active)]
    blocks.extend((body, caught, _rethrown(names)) for body, names in handled)
    blocks.append(([*statement.orelse, *statement.finalbody], caught, active))
    return blocks


def _escaping(statements: list[ast.stmt], caught: frozenset[str], active: str) -> list[str]:
    """Collect exception names that escape a statement list.

    A ``try`` statement is split by :func:`_try_blocks`, so only its body is
    walked with the handlers' names added to *caught*.

    Parameters
    ----------
    statements : list[ast.stmt]
        Statements of the definition's own scope.
    caught : frozenset[str]
        Names caught by enclosing ``try`` blocks.
    active : str
        The exception a bare ``raise`` would re-raise here, or ``""``.

    Returns
    -------
    list[str]
        Escaping names, possibly with duplicates. Nested functions and
        classes are skipped because their raises happen only when called.
    """
    found: list[str] = []
    for statement in statements:
        if isinstance(statement, ast.Raise):
            name = _exception_name(statement.exc) if statement.exc is not None else active
            found.extend([name] if name and not is_caught(name, caught) else [])
        elif isinstance(statement, ast.Try | ast.TryStar):
            for block, block_caught, block_active in _try_blocks(statement, caught, active):
                found.extend(_escaping(block, block_caught, block_active))
        elif not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.extend(_escaping(_nested_statements(statement), caught, active))
    return found


def _function_facts(node: ast.FunctionDef | ast.AsyncFunctionDef, *, in_class: bool) -> Facts:
    """Collect the facts of one function or method.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        The function.
    in_class : bool
        Whether it is defined directly in a class body.

    Returns
    -------
    Facts
        Signature, return, generator and raise facts.
    """
    implicit = _implicit_param(node, in_class=in_class)
    decorators = {decorator_name(d) for d in node.decorator_list}
    scope = own_scope(node)
    # dict.fromkeys de-duplicates while keeping first-raised order.
    raises = list(dict.fromkeys(_escaping(node.body, frozenset(), "")))
    return Facts(
        name=node.name,
        kind="function",
        params=_signature(node, implicit),
        implicit=implicit,
        returns=_unparse(node.returns),
        returns_value=any(
            isinstance(n, ast.Return)
            and n.value is not None
            # "return None" is a procedure's early exit, not a value.
            and not (isinstance(n.value, ast.Constant) and n.value.value is None)
            for n in scope
        ),
        is_generator=any(isinstance(n, ast.Yield | ast.YieldFrom) for n in scope),
        is_property=bool(decorators & {"property", "functools.cached_property", "cached_property"}),
        is_overload=any(name and name.rsplit(".", 1)[-1] == "overload" for name in decorators),
        raises=raises,
    )


def _self_attributes(init: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Param]:
    """Return public ``self.x`` attributes an ``__init__`` assigns.

    Parameters
    ----------
    init : ast.FunctionDef | ast.AsyncFunctionDef
        The constructor.

    Returns
    -------
    list[Param]
        Attribute names with annotations when written as ``self.x: T = ...``.
    """
    found: dict[str, str] = {}
    for node in own_scope(init):
        targets: list[ast.expr] = []
        annotation = ""
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            annotation = ast.unparse(node.annotation)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and not target.attr.startswith("_")
            ):
                # The first assignment wins, as it usually carries the annotation.
                _ = found.setdefault(target.attr, annotation)
    return list(starmap(Param, found.items()))


def _class_facts(node: ast.ClassDef) -> Facts:
    """Collect the facts of one class.

    Parameters
    ----------
    node : ast.ClassDef
        The class.

    Returns
    -------
    Facts
        Constructor parameters and public instance attributes. Without an
        explicit ``__init__`` the signature is unknown (dataclasses, plain
        inheritance), so parameter checks are skipped.
    """
    facts = Facts(name=node.name, kind="class", signature_known=False)
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == "__init__":
            facts.params = _signature(child, _implicit_param(child, in_class=True))
            facts.signature_known = True
            facts.attributes = _self_attributes(child)
        elif (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and not child.target.id.startswith("_")
        ):
            facts.attributes.append(Param(child.target.id, ast.unparse(child.annotation)))
    return facts


def collect(node: DefNode, *, in_class: bool) -> Facts:
    """Collect the facts of any function or class definition.

    Parameters
    ----------
    node : DefNode
        The definition.
    in_class : bool
        Whether *node* sits directly in a class body.

    Returns
    -------
    Facts
        The definition's proven facts.
    """
    if isinstance(node, ast.ClassDef):
        return _class_facts(node)
    return _function_facts(node, in_class=in_class)
