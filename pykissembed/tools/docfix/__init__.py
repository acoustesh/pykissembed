"""``docfix``: bring docstrings into NumPy-convention conformance, safely.

The tool only ever edits docstring text. Deterministic fixes re-format or
re-label what is already written, or add facts the code proves (names,
annotations). Anything that needs new wording is listed as a ``prose``
finding and is filled only by the opt-in drafting pass. Every changed file
is verified by the guards in :mod:`pykissembed.tools.docfix.engine` before
it is written. The default run changes nothing.
"""

from pykissembed.tools.docfix.common import CODES
from pykissembed.tools.docfix.draft import DEFAULT_MODEL
from pykissembed.tools.docfix.engine import FileResult, Options
from pykissembed.tools.docfix.runner import Request, main, run, selection

__all__ = ["CODES", "DEFAULT_MODEL", "FileResult", "Options", "Request", "main", "run", "selection"]
