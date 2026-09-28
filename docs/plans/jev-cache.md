# Jev response cache + Jev comment audit

## Context
`test_jev_docstring_audit` ([pykissembed/checks/jev_docstring_audit.py](pykissembed/checks/jev_docstring_audit.py), commit 0999347) makes one OpenRouter Decisions call per documented symbol on every run. The goal is to pay only for text that changed: hash the exact `state` and `question` sent to `typesafe/jev-1.13`, and reuse the stored response when both hashes match. Any edit to code, a docstring or a comment changes the state hash and so triggers a new call. A second consumer check will grade inline comments with the user's 0–4 rubric (pass only above 2.25), using the same cache.

Decisions made with the user:
- **Store:** a stdlib `sqlite3` file, gitignored, shared by both checks.
- **Docstring audit:** stops grading inline comments.
- **Comment audit units:** functions and methods only.
- **Default flow:** a cache hit is graded from the stored response. A cache miss calls Jev and stores the result. With no key or with `--cached-only`, a miss is skipped.

## 1. New shared module `pykissembed/jev.py` (outside `checks/`, so the plugin doesn't collect it)
Move these out of `jev_docstring_audit.py` unchanged:
- `JEV_MODEL`, `DECISIONS_URL`, `API_KEY_ENV`, the retry and HTTP constants and `MAX_STATE_CHARS`
- `SymbolState`, `_is_overload_stub`, `_extract_symbol_states`
- `_load_api_key`, `_requests_api`, `_is_retryable`

Both checks import them from there. Keeping two copies would trip the repo's own similarity gate.

New code:
- **`parse_score(payload, question_id, n_levels) -> float | None`** is the current `_parse_score_answer` generalised to return the raw 0-based expected value. The docstring audit adds `+1` itself; the comment audit uses the value as is, because its levels already run 0–4.
- **`state_payload(state) -> dict`** builds `{file, symbol, kind, source}`. `lineno` is **dropped** from what is sent, so edits above a function don't invalidate its cache entry. Jev never needed the line number to grade anything.
- **Hashes:**
  - `state_hash = compute_content_hash(canonical_json(state_payload))`
  - `question_hash = compute_content_hash(canonical_json({"model": JEV_MODEL, "questions": questions}))`
  - `canonical_json` is `json.dumps(sort_keys=True, separators=(",", ":"))`.
  - `compute_content_hash` is reused from [pykissembed/similarity/ast_helpers.py:66](pykissembed/similarity/ast_helpers.py#L66), provided importing it is light. Otherwise it's a one-line `hashlib.sha256`.
  - Because the model id is part of the question hash, changing the model or any rubric text re-grades everything.
- **Cache:** `open_cache(config) -> sqlite3.Connection` opens `config.baseline_path / "jev_cache.sqlite3"` with `timeout=30` and `PRAGMA journal_mode=WAL`, which makes it safe for concurrent xdist workers. The table is:
  ```sql
  CREATE TABLE IF NOT EXISTS jev_responses(
    state_hash TEXT NOT NULL, question_hash TEXT NOT NULL,
    model TEXT NOT NULL, response TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (state_hash, question_hash))
  ```
  Only the last response is kept per pair (`INSERT OR REPLACE`), and each insert commits immediately, so an interrupted run keeps every response it paid for.
- **`ask_jev(state, questions, *, parse, api_key: str | None, conn) -> float | None`** runs in this order:
  1. Look up the hash pair. On a hit, return `parse(json.loads(response))`. Parsing happens at read time, so a later parser fix doesn't cost a new call.
  2. On a miss with no `api_key`, return `None`, which means skipped.
  3. Otherwise POST using the existing retry loop from `_query_jev`. Store the raw decoded JSON **only when `parse` succeeds**, so transport garbage and malformed replies are never cached.

## 2. `pykissembed/checks/jev_docstring_audit.py`
- Import the shared pieces from `pykissembed.jev`, and replace `_query_jev` with `ask_jev(..., parse=lambda p: _shift(parse_score(p, ...)))`.
- **Drop comment grading:**
  - delete `_COMMENT_GUIDANCE` and its concatenation into `_INSTRUCTIONS`
  - trim level 6 in `_SCORE_LEVELS` to remove "every inline comment is relevant…"
  - fix the module docstring and the `_COMMENT_GUIDANCE` rationale comment
- The test now takes the `cached_only` fixture ([pykissembed/plugin.py:152](pykissembed/plugin.py#L152)) and passes `api_key=None` when it is set.
- **Missing API key:**
  - Before: the whole test is skipped.
  - After: the audit still runs from cache, and misses are skipped. The test is skipped only when the run graded nothing and there is no key.
  - The failure header, and a skip or summary line, report how many symbols were ungraded.

## 3. New `pykissembed/checks/jev_comment_audit.py` → `TestJevCommentAudit.test_jev_comment_audit` (`@pytest.mark.jev`)
- **Units:** the output of `_extract_symbol_states` filtered to `kind == "function"`, which covers functions and methods. The source sent includes the docstring, because the rubric has to know whether intent is already explained there. Symbols without a docstring are **not** short-circuited, since comments can make up for a missing docstring.
- **One `score` question**, `comment_score`, with criteria ordered worst → best. The API's 0-index then equals the user's level, so no shift is needed:
  - 0 Major: a misleading claim likely to cause misuse, e.g. about thread safety, validation, ownership, units or destructive effects.
  - 1 Moderate: missing necessary context, or a materially inaccurate or stale comment. This includes a hidden constraint or opaque intent left unexplained, and a linter/type-checker suppression with no reason given.
  - 2 Minor: repeated narration of obvious operations, or confusing wording whose meaning can still be recovered.
  - 3 No violation: comments are accurate and useful, or omitting them is valid, or a workaround is concisely justified.
  - 4 No comments, and none needed: the code is straightforward, or it is complex but its intent is explained by the docstring.
- **Instructions** encode the user's five rules:
  1. Judge comments, not code, and check each comment's claims against the code; a contradiction is heavily penalised.
  2. Penalise a missing comment only by naming the specific hidden constraint and the misunderstanding it invites. It is illustrated by this worked example, stored as the constant `_ABSENCE_EXAMPLE` and appended to the rule verbatim:

     ````text
     Example of a justified absence penalty:

     ```python
     _TOKEN = re.compile(
         r"""(?P<quoted>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
         r"""|\((?:[^()'"]|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')*\)"""
     )
     while True:
         updated = _TOKEN.sub(
             lambda m: m.group(0) if m.group("quoted") is not None else "",
             text,
         )
         if updated == text:
             return text
         text = updated
     ```

     - The `if updated == text: return text` guard with no comment is
       1 (Moderate). The regex removes only innermost parenthesised groups,
       so the loop must repeat until nothing changes in order to remove
       nested ones. A plausible refactor to a single `.sub()` call would
       silently leave nested groups behind. The comment belongs at the
       guard; buried in the docstring it loses its placement.
     - The block's intent (strip parenthesised groups, nested ones
       included, while keeping quoted strings intact) is opaque from the
       regex. If the docstring states that intent, the missing comment is
       4. If nothing states it, it is 1 (Moderate).
     - Counter-example: a plain loop over a list with no comment is 4.
       "Complex code needs comments" alone never justifies a penalty.
     ````
  3. Never use density, length or coverage as a target.
  4. No penalty for missing optional metadata such as authors, dates, issue links, TODOs, examples or banners.
  5. Judge redundancy by its cost: brief orientation comments are fine; penalise repetition that adds clutter or can go stale.

  A symbol's level is that of its **worst** defect.
- **Baseline:** `tests/baselines/jev_comment_audit.json` with kind `jev_comment_audit` and `data.min_score = 2.25`. Scores must exceed the bar. It reuses the `_locked_envelope` / `_read_threshold` pattern, clamped to `[0, 4]`. There is no separate bar for test code (YAGNI).
- **Registration:**
  - add `"jev_comment_audit"` to `_CHECK_MODULES` ([pykissembed/plugin.py:38](pykissembed/plugin.py#L38)) and to [pykissembed/checks/__init__.py](pykissembed/checks/__init__.py)
  - add the kind to the enum in [pykissembed/schemas/baselines.v1.json](pykissembed/schemas/baselines.v1.json)
  - reuse the existing `jev` marker

## 4. Housekeeping
- Add `tests/baselines/jev_cache.sqlite3*` to `.gitignore`, which also covers the `-wal` and `-shm` sidecars.
- Update [tests/test_plugin_collection.py](tests/test_plugin_collection.py) if it counts or lists the check modules.
- Skipped: pruning cache rows that no current symbol uses anymore. Leave a `ponytail:` note saying the rows are small and a prune can be added when the file grows.

## 5. Tests
- **[tests/test_jev_docstring_check.py](tests/test_jev_docstring_check.py):**
  - update the imports and monkeypatch targets for the move to `pykissembed.jev`
  - remove `lineno` from the request-shape assertion
  - delete `test_guidance_never_penalises_absent_comments`
  - `_run_check` already points `get_config` at `tmp_path`, so each test gets a fresh cache
- **New `tests/test_jev_comment_check.py`, covering the comment rubric:**
  - scores 2.24 and 2.25 fail; 2.26 passes
  - only functions and methods are sent, never classes
  - a symbol with no docstring is still sent
  - the rubric is ordered and has 5 levels
- **New `tests/test_jev_cache.py`, using the counting `requests.post` double from the existing tests:**
  - A second run of the same source makes 0 POSTs.
  - An edit to the source or to a comment causes 1 new POST.
  - Changing the questions or the model causes a new POST.
  - Shifting a function down (lineno changes) makes 0 POSTs.
  - A malformed reply is not stored, so the next run POSTs again.
  - With no key: a cached symbol is graded and can fail, and an uncached one is skipped.
  - `cached_only=True` never POSTs.

## Verification
1. `uv sync --all-extras`, then `uv run pytest tests/test_jev_*.py tests/test_plugin_collection.py`.
2. Run the repo's own gates: ruff, pyright/ty, `test_no_suppressions_or_casts`, and similarity. If the Qwen-AST gate fails, A/B it against a stash, since three near-duplicate pairs are already known to fail on main.
3. **Live check** (needs `OPENROUTER_API_KEY`): run `uv run pytest -m jev pykissembed/checks -p no:cacheprovider` twice.
   - Run 1 fills `jev_cache.sqlite3`. Record the row count with `sqlite3 … 'select count(*) from jev_responses'`.
   - Run 2 should finish in seconds with the row count unchanged.
   - Edit one comment and rerun: exactly one new row should appear.
4. Report how many symbols the first live comment-audit run flags at 2.25, so the bar can be recalibrated the way the docstring bar was.
