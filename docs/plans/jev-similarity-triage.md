# Jev Consolidation Review for Similarity Failures: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send each pair that fails `test_providers_parallel` to a Jev score model, fail only the pairs that should change, and explain each failure.

**Architecture:** Embeddings stay the first-stage filter. Provider threads record the pairs they flag. After all threads finish, the main thread merges those pairs across providers, grades each one once through the existing `ask_jev` cache, and builds one report.

**Tech Stack:** Python 3.14, pytest, NumPy, SQLite Jev cache, OpenRouter Decisions API (`score` and `noul` questions).

**Spec:** The user's request, reproduced under "Decision Rubric".

## Decision Rubric

Use this text verbatim in the score question.

Question:

> What level of consolidation would simplify future changes to these two functions? Use the rules that must change together, each function's contract with its callers, and the branches or flags the shared implementation would require.

| Score | Decision | Criterion |
| --- | --- | --- |
| 0 | Keep separate | The similar code implements rules that can change independently, or sharing it would add more complexity than it removes. |
| 1 | Extract shared logic | Both functions implement a specific rule that should change in one place. Their workflows remain distinct. |
| 2 | Share the core workflow | Both functions follow the same core workflow, but their public contracts require separate entry points. |
| 3 | Consolidate | Both functions have the same responsibility. One function can serve both sets of callers without caller-specific flags or branches. |

## Findings That Shape the Design

- **Nothing fails today.** All 12 providers ran on `main` on 2026-11-21 with no pair at or above the configured 0.97 threshold. The closest pair is `TestJevDocstringAudit` vs `TestJevCommentAudit` at 96.57% (Gemini-AST).
- **Candidate volume by threshold**, as distinct pairs across providers: 0.97 → 0, 0.95 → 5, 0.93 → 10, 0.90 → 45, 0.88 → 92.
- **The Decisions API returns no written rationale.** It returns scores, probabilities and confidence. Explanations therefore come from the score distribution plus three yes/no evidence questions that mirror the rubric's three factors.
- **The model needs caller context.** The rubric asks about caller contracts, and a function's source does not show its callers.
- **SQLite connections are thread-bound.** Grading must run after the provider thread pool exits.

## Decisions

| Topic | Decision |
| --- | --- |
| Fail bar | A pair fails at score `>= 1.0`, the first level that recommends a code change. Config key: `consolidation_fail_score` in `tests/baselines/similarity.json`, default `1.0`, clamped to `[0, 3]`. |
| Clearing a pair | A pair clears only when `score < bar` and confidence is known and above `LOW_CONFIDENCE` (0.2). |
| Ungraded pairs | They stay failed. A missing key, cache miss, API error or refusal must not turn a failing gate green. |
| Embedding thresholds | Unchanged in this work. Lowering them is a separate decision after the validation run. |
| `test_combined_similarity` | Unchanged. It is outside the request. |

## Global Constraints

- No `# type: ignore`, `# noqa`, `cast`, or broad `Any`.
- Do not edit consumer projects such as `speechtext`, or write outside this workspace.
- Use `uv sync --all-extras` if the environment needs a resync.
- Do not commit, push, or call a live API unless the user approves it.
- New code must pass this repository's own pykissembed gates.

## Review Focus

1. Same-named functions in different files, such as `_load_callable`, must not share callers. Pinned in Task 2.
2. A missing key, `--cached-only` miss, or model refusal must keep the pair failing. Pinned in Task 2.
3. A pair flagged by several providers, in either order, must produce one request. Pinned in Task 2.
4. A very long first function must not push the second function out of the model's input. Pinned in Task 2.
5. A provider crash must still fail the test when every flagged pair clears. Pinned in Task 3.

---

### Task 1: Record flagged pairs

**Files:**
- Modify: `pykissembed/similarity/types.py`
- Modify: `pykissembed/similarity/checks.py:376-703`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Produces: `type FlaggedEdge = tuple[FunctionInfo, FunctionInfo, float]` in `types.py`.
- Produces: an optional eighth positional parameter `flagged: list[FlaggedEdge] | None = None` on `_find_matrix_violations`.
- Produces: keyword `flagged: list[FlaggedEdge] | None = None` on `run_provider_similarity_checks` and `run_jina_similarity_checks`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([[1, .95, .1], [.95, 1, .1], [.1, .1, 1]], [("a", "b")]),       # pair
        ([[1, .85, .85], [.85, 1, .1], [.85, .1, 1]], [("a", "b"), ("a", "c")]),  # neighbors
        ([[1, .85, .1], [.85, 1, .1], [.1, .1, 1]], []),                 # one neighbor only
    ],
)
def test_find_matrix_violations_records_flagged_edges(matrix, expected):
    flagged: list[FlaggedEdge] = []
    similarity_checks._find_matrix_violations(
        make_functions("a", "b", "c"), np.array(matrix, dtype=np.float32),
        0.9, 0.8, None, None, 0, flagged,
    )
    assert [(a.name, b.name) for a, b, _ in flagged] == expected
```

Add a local `make_functions` helper if the module has no equivalent.

- [ ] **Step 2: Verify it fails**

Run: `uv run pytest tests/test_similarity.py -k records_flagged_edges -q`
Expected: FAIL with a `TypeError` for the extra positional argument.

- [ ] **Step 3: Implement the sink**

Append each edge at or above the pair threshold. When a function has at least two neighbors, also append its neighbor edges. Keep the returned strings identical. Both `run_*` functions must pass `flagged` **positionally**, because an existing test replaces `_find_matrix_violations` with `lambda *_`.

- [ ] **Step 4: Verify it passes**

Run: `uv run pytest tests/test_similarity.py -q`
Expected: PASS.

---

### Task 2: Grade and explain pairs

**Files:**
- Create: `pykissembed/similarity/consolidation.py` (do not import it from `similarity/__init__.py`, because `pykissembed.jev` imports that package)
- Modify: `pykissembed/jev.py` (add `parse_noul`)
- Modify: `pykissembed/wrapper_analysis.py` (add `call_terminal_name = _call_terminal_name`, the re-export pattern `jev.py` uses)
- Test: `tests/test_similarity_consolidation.py`

**Interfaces:**
- Consumes: `FlaggedEdge`, `ask_jev`, `open_cache`, `load_api_key`, `grade_summary`, `is_low_confidence`, `parse_grade`, `parse_source_files`, `call_terminal_name`.
- Produces: `parse_noul(payload: object, question_id: str) -> float | None`, which returns a value in `[0, 1]` or `None`.
- Produces:

```python
def triage_flagged_pairs(
    flagged: Mapping[str, Sequence[FlaggedEdge]],  # provider label -> edges
    paths: list[Path],
    *,
    fail_score: float,
    cached_only: bool,
) -> str | None:
    """Return the failure report, or None when every pair cleared."""
```

**Question payload:**
- `consolidation_score`: type `score`, the Rubric question as instructions, four criteria formatted `"<n> <Decision>: <Criterion>"`.
- `shared_rule`: type `noul`. "Do both functions implement a specific rule that must change in one place?"
- `separate_contracts`: type `noul`. "Do their contracts with callers require separate entry points?"
- `needs_flags`: type `noul`. "Would a shared implementation need caller-specific flags or branches?"

**Pair state:** Build a `SymbolState` with `kind="function_pair"`.
- Order the two functions by `(file, name, start_line)`.
- Include each function's source, capped at 5,000 characters, and up to five call sites capped at 750 characters.
- Omit line numbers so moving code keeps the cached grade.
- Keep the total at or below `MAX_STATE_CHARS`.

**Caller lookup:** A call counts for a target when it is in the target's file, when its file has `from <module ending in target stem> import <name>`, or when it is written `<target stem>.<name>(...)`. Add this comment at the lookup: `# ponytail: static name matching misses aliases and instance calls across files; use a real call graph if grades prove caller-blind.` Tell the model the caller list may be incomplete.

**Report format for each failing pair:**

```text
pkg/a.py:_grade_comments vs pkg/b.py:_grade_docstrings
  Decision: level 1.24/3, confidence 0.62 (need <1): 1 Extract shared logic: ...
  Distribution: keep 12% | extract 58% | core 26% | consolidate 4%
  Why: shared rule: yes (84%); separate contracts: yes (71%); needs flags: no (18%)
  Flagged by: Gemini-AST 96.6%, Gemini-Text 96.1%
```

Group the report as **Needs change** (highest score first), **Uncertain**, then **Ungraded**. Cleared pairs go in one `UserWarning`, one line each, so a passing run still shows them.

- [ ] **Step 1: Write the failing tests**

Stub `requests.post` and config the way `tests/test_jev_comment_check.py` does.

| Test | Assertion |
| --- | --- |
| `test_rubric_matches_decision_table` | Four criteria start with `0 Keep separate`, `1 Extract shared logic`, `2 Share the core workflow`, `3 Consolidate`. Instructions contain the Rubric question. |
| `test_fail_bar_boundary[0.99, 1.0]` | `0.99` returns `None` and warns. `1.0` returns a report under `Needs change`. |
| `test_low_or_missing_confidence_never_clears[0.2, None]` | Score `0.4` returns a report under `Uncertain`. |
| `test_unavailable_grade_stays_failed` | With `cached_only=True`, `post` is never called and the report contains `Ungraded`. |
| `test_refused_score_stays_failed` | An answer without `score` is reported as `Ungraded` and is not cached. |
| `test_report_explains_failure` | Report contains `1 Extract shared logic`, `extract 58%`, `shared rule: yes (84%)`, `Gemini-AST 96.6%`. |
| `test_missing_evidence_keeps_grade` | A missing `needs_flags` shows `needs flags: no answer`, and the grade still counts. |
| `test_pair_sent_once_across_providers` | The same pair in reversed order from two providers makes one POST and lists both labels. |
| `test_same_named_functions_keep_their_own_callers` | Each `_load` function's section lists only its own file's caller. |
| `test_long_function_keeps_second_in_state` | A 20,000-character first function leaves the second's sentinel in the state, and the length is `<= MAX_STATE_CHARS`. |

- [ ] **Step 2: Verify they fail**

Run: `uv run pytest tests/test_similarity_consolidation.py -q`
Expected: FAIL during collection with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `parse_noul`, the re-export, and `consolidation.py`**

Grade sequentially on the calling thread with one `closing(open_cache(config))` connection. Use `api_key = None if cached_only else load_api_key()`.

- [ ] **Step 4: Verify they pass**

Run: `uv run pytest tests/test_similarity_consolidation.py tests/test_jev_cache.py tests/test_jev_comment_check.py -q`
Expected: PASS.

---

### Task 3: Wire the review into `test_providers_parallel`

**Files:**
- Modify: `pykissembed/checks/code_similarity.py:165-343`
- Modify: `README.md` (Similarity tests section)
- Test: `tests/test_similarity.py`

**Interfaces:**
- Consumes: `triage_flagged_pairs` and `read_threshold`.
- Produces: keyword `flagged: list[FlaggedEdge] | None = None` on `_run_similarity_test`.
- Produces: a new `pykissembed_paths: list[Path]` fixture parameter on `test_providers_parallel`.

- [ ] **Step 1: Write the failing tests**

Replace `_run_similarity_test` with a fake that appends to `flagged` and raises `pytest.fail.Exception`.

| Test | Assertion |
| --- | --- |
| `test_cleared_pairs_pass_the_gate` | With triage returning `None`, the test returns normally. |
| `test_confirmed_pair_fails_with_report` | Failure text contains the triage report and the provider's original message, including refactor priority. |
| `test_provider_crash_survives_cleared_pairs` | One provider raises `RuntimeError`, pairs clear, and failure text contains `Provider failures`. |
| `test_failure_without_edges_is_a_provider_error` | A `Failed` with an empty sink is reported under `Provider failures`. |
| `test_combined_similarity_skips_review` | With triage patched to raise, `test_combined_similarity` still fails with the original message. |

- [ ] **Step 2: Verify they fail**

Run: `uv run pytest tests/test_similarity.py -k "cleared_pairs or confirmed_pair or provider_crash or without_edges or skips_review" -q`
Expected: FAIL.

- [ ] **Step 3: Implement the wiring**

- Create `flagged = {p.label: [] for p in _PARALLEL_PROVIDERS}` before the pool, so each thread appends only to its own list.
- Treat an exception as a similarity failure only when it is `pytest.fail.Exception` **and** that provider's list is non-empty. Everything else stays in `errors`.
- After the pool exits, call `triage_flagged_pairs` when any list is non-empty. Read the bar with `read_threshold(config, key="consolidation_fail_score", default=1.0, max_level=3.0)`.
- Build one failure message from provider errors, the triage report, and the original similarity messages. Omit the similarity messages when triage returns `None`.
- Keep the existing all-providers-skipped behavior.
- Add a short README paragraph covering the review, the rubric, `consolidation_fail_score`, and that ungraded pairs fail.

- [ ] **Step 4: Verify they pass**

Run: `uv run pytest tests/test_similarity.py tests/test_similarity_consolidation.py tests/test_plugin_collection.py -q`
Expected: PASS.

---

### Task 4: Validate on real pairs

**Files:** No repository changes, unless a gate requires a fix.

- [ ] **Step 1: Run the offline suite and quality gates**

Run: `uv run pytest -q` and `uv run pytest --pykissembed-all --cached-only -q`
Expected: PASS, or skips only for embeddings of newly edited functions. Fix any new-code findings without suppressions.

- [ ] **Step 2: Ask the user to approve the live run**

The run sends the 10 candidate pairs' source and call sites to OpenRouter's free `perplexity/pplx-decider-v1.3:free` model. Because the edited functions have new hashes, it also calls the embedding providers and updates the embedding caches.

- [ ] **Step 3: Run the review at a temporary 0.93 threshold**

From a scratch test outside the repository, set `similarity_threshold_pair` and `similarity_threshold_neighbor` to `0.93` on the in-memory `shared_baselines["config"]`, then call `code_similarity.test_providers_parallel(..., update_baselines=False, cached_only=False)`.
Expected: roughly 10 pairs, each graded, cleared or explained. Do not change `tests/baselines/similarity.json`.

- [ ] **Step 4: Present the results**

Show a table of pair, decision, score, confidence and explanation. Recommend whether to keep `1.0` as the bar and whether to lower the 0.97 embedding threshold.
