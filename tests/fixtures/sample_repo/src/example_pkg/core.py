"""Core helpers — deliberately includes one missing docstring."""

from __future__ import annotations


def add(x: int, y: int) -> int:
    """Add two integers.

    Parameters
    ----------
    x : int
        First addend.
    y : int
        Second addend.

    Returns
    -------
    int
        ``x + y``.

    """
    return x + y


def greet(name: str) -> str:
    """Return a greeting for *name*.

    Parameters
    ----------
    name : str
        Name to greet.

    Returns
    -------
    str
        Greeting message.

    """
    return f"Hello, {name}"


def no_docstring_func(x: int) -> int:
    return x * 2


def too_complex(n: int) -> str:
    """Deliberately over-complex function (many branches).

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    str
        Classification label.

    """
    if n < 0:
        return "neg"
    if n == 0:
        return "zero"
    if n < 10:
        return "small"
    if n < 100:
        return "medium"
    if n < 1000:
        return "large"
    if n < 10000:
        return "xlarge"
    return "huge"
