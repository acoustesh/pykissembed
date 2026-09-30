"""Network-backed helpers: LLM prose drafts, Jev grading and gate targeting.

Nothing here runs unless asked for. Drafts come from an OpenRouter chat
model, are validated against the requested slots, and are cached in the
same SQLite file as Jev responses, so a re-run is free. Grading reuses the
docstring gate's own rubric, states and cache, which means a grade taken
while fixing is already cached for the next gate run.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pykissembed._http_retry import is_retryable_http_error
from pykissembed.baselines_engine import load_envelope
from pykissembed.checks.jev_docstring_audit import BASELINE_FILENAME, min_score_for, symbol_level
from pykissembed.jev import (
    MAX_STATE_CHARS,
    RETRY_DELAYS,
    canonical_json,
    extract_symbol_states,
    state_for,
)
from pykissembed.similarity.ast_helpers import compute_content_hash
from pykissembed.similarity.embeddings import requests_api

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping
    from pathlib import Path

    from pykissembed.config import PyqtestConfig
    from pykissembed.tools.docfix.facts import DefNode

# An OpenRouter preset: the model and its routing are configured on OpenRouter.
DEFAULT_MODEL = "@preset/deepseek"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 90.0
# Bump when the prompt or validation changes, so cached drafts are not reused.
PROMPT_VERSION = 1
MAX_DRAFT_CHARS = 400
MAX_SUMMARY_CHARS = 80

_SYSTEM_PROMPT = """\
You write fragments of NumPy-style Python docstrings. The code you are given
is the only source of truth: describe what it actually does, and never
speculate about intent, callers, or behaviour that is not visible in it.

Reply with one JSON object whose keys are exactly the requested slot ids and
whose values are plain-text strings without line breaks:
- "summary": one sentence of at most 80 characters, ending with a period.
- "param:NAME": what the parameter means and how the code uses it.
- "returns": what the returned value is.
- "yields": what each yielded value is.
- "raises:EXC": the condition under which EXC is raised, starting with "If".
- "attr:NAME": what the attribute holds.
Do not restate types. Use ``double backticks`` for code; no other markup."""

# Some models wrap JSON in a ```json fence despite response_format.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def validate_drafts(payload: object, slots: list[str]) -> dict[str, str] | None:
    """Accept a model reply only if it fills exactly the requested slots.

    Parameters
    ----------
    payload : object
        Decoded JSON reply.
    slots : list[str]
        Requested slot ids.

    Returns
    -------
    dict[str, str] | None
        Whitespace-normalised drafts, or ``None`` when any key is missing or
        extra, any value is empty, too long or contains a triple quote, or the
        summary is not one period-terminated line.
    """
    # Extra keys would be prose nobody asked for; missing ones leave gaps.
    if not isinstance(payload, dict) or set(payload) != set(slots):
        return None
    drafts: dict[str, str] = {}
    for slot in slots:
        value = payload[slot]
        if not isinstance(value, str):
            return None
        # Newlines are dropped: the caller re-wraps text to the file's width.
        text = " ".join(value.split())
        # A triple quote would terminate the docstring token early.
        if not text or len(text) > MAX_DRAFT_CHARS or '"""' in text or "'''" in text:
            return None
        if slot == "summary" and (len(text) > MAX_SUMMARY_CHARS or not text.endswith(".")):
            return None
        drafts[slot] = text
    return drafts


def _content(response: object) -> object:
    """Extract the JSON object from a chat-completions response body.

    Parameters
    ----------
    response : object
        Decoded response body.

    Returns
    -------
    object
        The decoded message content, or ``None`` when absent or not JSON.
    """
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return None
    try:
        return json.loads(_FENCE.sub("", content.strip()))
    except json.JSONDecodeError:
        return None


@dataclass(slots=True)
class OpenRouterDrafter:
    """Drafts prose slots with an OpenRouter chat model, within a call budget.

    Replies are cached in a ``docfix_drafts`` table of the shared Jev cache
    database, keyed by the full request, so repeated runs cost nothing.

    Attributes
    ----------
    api_key : str
        OpenRouter API key.
    conn : sqlite3.Connection
        Connection to the shared cache database.
    model : str
        OpenRouter model id.
    max_calls : int
        Maximum number of live (uncached) requests.
    calls : int
        Live requests made so far.
    """

    api_key: str
    conn: sqlite3.Connection
    model: str = DEFAULT_MODEL
    max_calls: int = 50
    calls: int = 0

    def __post_init__(self) -> None:
        """Create the draft cache table next to the Jev responses."""
        _ = self.conn.execute(
            "CREATE TABLE IF NOT EXISTS docfix_drafts ("
            "key TEXT PRIMARY KEY, model TEXT NOT NULL, response TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        self.conn.commit()

    def draft(self, source: str, slots: list[str]) -> dict[str, str] | None:
        """Return validated drafts for *slots*, from cache or one request.

        Parameters
        ----------
        source : str
            The symbol's source segment.
        slots : list[str]
            Slot ids to fill.

        Returns
        -------
        dict[str, str] | None
            Drafts by slot id, or ``None`` when the reply is invalid, the
            request fails, or the call budget is spent.
        """
        body = self._body(source, slots)
        key = compute_content_hash(canonical_json({"v": PROMPT_VERSION, "body": body}))
        row = self.conn.execute(
            "SELECT response FROM docfix_drafts WHERE key = ?", (key,)
        ).fetchone()
        # Cached replies are re-validated, so tightening the rules applies
        # retroactively without paying for new requests.
        if row is not None:
            return validate_drafts(json.loads(row[0]), slots)
        # Cache hits are free; only live requests count against the budget.
        if self.calls >= self.max_calls:
            return None
        self.calls += 1
        payload = _content(self._post(body))
        drafts = validate_drafts(payload, slots)
        # Invalid replies are not cached, so a later run can try again.
        if drafts is not None:
            _ = self.conn.execute(
                "INSERT OR REPLACE INTO docfix_drafts VALUES (?, ?, ?, ?)",
                (key, self.model, canonical_json(payload), datetime.now(UTC).isoformat()),
            )
            self.conn.commit()
        return drafts

    def _body(self, source: str, slots: list[str]) -> dict[str, object]:
        """Build the chat-completions request body.

        Parameters
        ----------
        source : str
            The symbol's source segment.
        slots : list[str]
            Slot ids to fill.

        Returns
        -------
        dict[str, object]
            Deterministic request body (temperature zero, JSON output).
        """
        # Same truncation as the Jev states, so huge classes cost the same.
        user = json.dumps({"slots": slots, "source": source[:MAX_STATE_CHARS]})
        return {
            "model": self.model,
            # Deterministic replies make the cache key a faithful identity.
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }

    def _post(self, body: dict[str, object]) -> object:
        """Send *body*, retrying transient failures like the Jev client.

        Parameters
        ----------
        body : dict[str, object]
            Request body.

        Returns
        -------
        object
            The decoded response, or ``None`` on a permanent failure.
        """
        # requests is an optional dependency; without it drafting is a no-op.
        try:
            post, timeout_error, _ = requests_api()
        except ImportError, TypeError:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = post(CHAT_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
                raise_for_status = getattr(response, "raise_for_status", None)
                if callable(raise_for_status):
                    _ = raise_for_status()
                decode = getattr(response, "json", None)
                return decode() if callable(decode) else None
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # Only timeouts, 429 and 5xx are retried; anything else is final.
                if attempt < len(RETRY_DELAYS) and is_retryable_http_error(exc, timeout_error):
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                return None
        return None


@dataclass(slots=True)
class JevGrader:
    """Judges rewrites with the docstring gate's rubric, cache and bars.

    Levels are expected values and jitter by a few hundredths between
    near-identical texts, so a drop within ``tolerance`` is accepted, unless
    it takes a symbol from passing the gate to failing it.
    """

    api_key: str | None
    conn: sqlite3.Connection
    baseline: Mapping[str, object]
    tolerance: float = 0.1

    def verdict(self, old: DefNode, original: str, new: DefNode, updated: str, key: str) -> str:
        """Return why the rewrite of one definition is refused, or ``""``.

        Parameters
        ----------
        old : DefNode
            The definition parsed from *original*.
        original : str
            File text before the edits.
        new : DefNode
            The same definition parsed from *updated*.
        updated : str
            File text after the edits.
        key : str
            Project-relative file key.

        Returns
        -------
        str
            ``""`` to keep the rewrite; otherwise a reason naming both levels.
        """
        state = state_for(new, updated, key)
        before = symbol_level(state_for(old, original, key), api_key=self.api_key, conn=self.conn)
        after = symbol_level(state, api_key=self.api_key, conn=self.conn)
        bar = min_score_for(state, self.baseline)
        # Without a grade there is no proof of non-regression, so refuse.
        if after is None:
            return f"the rewrite could not be graded (was {before})"
        # Nothing to regress from (e.g. an ungraded original): keep the fix.
        if before is None:
            return ""
        if after < before - self.tolerance:
            return f"Jev level fell from {before:.2f} to {after:.2f}"
        if after < bar <= before:
            return f"Jev level {before:.2f} -> {after:.2f} crosses the gate bar {bar:g}"
        return ""


def gate_baseline(config: PyqtestConfig) -> Mapping[str, object]:
    """Load the docstring gate's baseline payload (its score bars).

    Parameters
    ----------
    config : PyqtestConfig
        Active config naming the baseline directory.

    Returns
    -------
    Mapping[str, object]
        The envelope data; empty when the gate has no baseline yet, in which
        case the gate's default bars apply.
    """
    return load_envelope(config.baseline_path / BASELINE_FILENAME, kind="jev_docstring_audit").data


def failing_targets(
    paths: list[Path], config: PyqtestConfig, conn: sqlite3.Connection
) -> tuple[frozenset[tuple[str, int]], int]:
    """Return the symbols the docstring gate currently fails, from cache only.

    Parameters
    ----------
    paths : list[Path]
        Directories to scan.
    config : PyqtestConfig
        Active config, for the project root and baseline thresholds.
    conn : sqlite3.Connection
        Shared Jev response cache.

    Returns
    -------
    tuple[frozenset[tuple[str, int]], int]
        ``(file_key, line)`` of every symbol below its bar, and the number
        of symbols with no cached grade (those are not targeted).
    """
    data = gate_baseline(config)
    targets: set[tuple[str, int]] = set()
    ungraded = 0
    for base in paths:
        for state in extract_symbol_states(base, root=config.root):
            # Cache only: targeting must be free and must not change the cache.
            level = symbol_level(state, api_key=None, conn=conn)
            if level is None:
                ungraded += 1
            elif level < min_score_for(state, data):
                targets.add((state.file_key, state.lineno))
    return frozenset(targets), ungraded
