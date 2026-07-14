Developer: ---
description: "Always-on workspace rules for the repo: fetch current upstream docs from context7 first when needed."
applyTo: "**"
---

# Workspace Rules — speechtext

The documentation lookup for any non-trivial change in this repo is:

1. **`context7`** — pull current upstream documentation for the libraries
    involved, so you reason against today's API, not yesterday's training
    cut-off.

This step is mandatory when the trigger conditions below hold.

---

## 1. Step 1 — `context7` for current documentation

This VS Code instance has the `context7` MCP server registered at the
**user profile** level (see
[`/home/alvaro/.vscode-server/data/User/profiles/-68d70264/mcp.json`](../.vscode/mcp.json)
— outside the workspace, but always available in this editor). It exposes:

> **Tool:** `mcp_context7_resolve-library-id`
> **Tool:** `mcp_context7_query-docs`

### When to call context7

Before writing or modifying code that touches a third-party API, call
context7 first **unless one of the explicit exceptions in “When NOT to call
context7” applies**. Concretely:

- Any new call into the OpenRouter Chat Completions API or to a provider
  SDK (AssemblyAI, Azure, ElevenLabs, Google, Mistral, OpenAI, …).
- Any change to test infrastructure under
  [`tests/`](../tests/) (pytest, pytest-asyncio, respx / httpx mocking).
- Any change to the Python environment declared in
  [`.vscode/settings.json`](../.vscode/settings.json)
  (ms-python.python:venv / uv).
- Whenever the user mentions a library by name and you are not 100% sure
  of its current surface — context7 is faster and more reliable than
  guessing.

### How to call it

1. **Resolve the library id** (unless the user supplied one in the form
   `/org/project`):

   ```text
   mcp_context7_resolve-library-id(
     query  = "Short description of what you need, e.g. 'async HTTP client with timeouts'",
     libraryName = "httpx"
   )
   ```

   Pick the result whose **name** matches the library, **Source Reputation**
   is `High` or `Medium`, and **Benchmark Score** is highest. Pin a specific
   version with `/org/project/version` when stability matters.

2. **Query the docs** with that id:

   ```text
   mcp_context7_query-docs(
     libraryId = "/encode/httpx",
     query     = "how to set per-request and client timeouts, and how to raise_for_status"
   )
   ```

   Pass a focused, single-concept query. Do not bundle multiple questions
   in one call — context7 degrades on multi-concept queries.

3. **Do not exceed 3 calls per question** for
    `resolve-library-id` and 3 calls for `query-docs`. If you still don't
    have what you need, stop and ask the user.

### When NOT to call context7

- Trivial edits (typos, formatting, single-line renames) where the local
  code, tests, or repo memory already give a definitive answer, even if the
  touched file references a third-party library.
- Pure execution of an unambiguous, user-specified plan when the required
  library usage is already explicit and stable in the local code or in prior
  context7 results for this task.
- Pure reasoning tasks with no third-party API surface (e.g. "what should
  we name this variable").
- Anything that would require a secret (API key, token, password) in the
  query — context7 logs nothing we control, so do not send secrets through
  it.

### Libraries this repo touches — pre-resolved ids

If you are working on one of these surfaces, you can skip step 1 and use
the id directly. Re-resolve if the id looks stale (404 / no snippets).

| Library | Library id | Used in |
|---|---|---|
| `pytest`, `pytest-asyncio` | `/pytest-dev/pytest` | [`tests/`](../tests/) |
| OpenRouter Chat Completions API | `/openrouter/awesome-openrouter` (or `/openrouterai/api-reference`) | any provider hitting OpenRouter |

Treat this table as a hint, not a source of truth — context7 ids drift
when publishers reorganise their docs.

---

## 2. Other conventions preserved from this workspace

- Python source lives at the repo root (`process_audio.py`,
  etc, providers in `*_provider.py`, …); tests live in
  [`tests/`](../tests/). New tests must be runnable via `pytest tests/`.
- The Python interpreter for this workspace is `.venv/bin/python`
  (configured in [`.vscode/settings.json`](../.vscode/settings.json)).
- Provider implementations follow the pattern in
  [`schema.py`](../schema.py) and [`tools.py`](../tools.py) — reuse the
  shared schemas instead of redefining them.
- When in doubt about provider-specific behaviour, prefer reading the
  provider module over guessing, and consult context7 before making
  architectural changes that span multiple providers.

# Role and Objective
- Act as a GitHub Copilot agent with expertise in Python 3.14 and WSL2 Ubuntu 24.04
# Working Context
- `<file>` refers to the file or files involved.
- Dependency manager: `uv` (or pixi only when you are dealing with GPU-accelerated libraries). Never use `pip`.
- Use the `context7` tool to verify current functionality and library usage when its trigger conditions apply.

# Implementation Rules
- Do not add new dependencies without asking first.
- Use NumPy-style docstrings.
- Before proceeding to implementation, ask 3–4 questions about design options unless the user provided an explicit step-by-step plan **and** the change is a single-file mechanical transformation with no design choices.
- Do not implement anything until the user approves the plan.

# Quality and Validation
- Tests: `uv run pytest <file>`
- Add a test for every new function.
- Lint and format: `ruff check --fix --preview --unsafe-fixes <file>` and `ruff format`
- Type checking: `uv run pyright <file>`

# Repository Instructions and Memory
1. Read relevant repository instructions and available Copilot Memory.
2. Do not store the Active Plan in Copilot Memory.
3. Store only durable repository facts in Memory, such as build commands, architecture constraints, naming conventions, or test rules.
4. Before storing memory, show the exact proposed memory and ask for approval.

# Active Plan
1. Produce a 3–7 step Active Plan before editing.
2. Keep the Active Plan in this chat/session context.
3. Before each coding step, restate:
   - the current plan step number
   - whether the step is unchanged, changed, or completed
   - the reason for any change
4. After each change, update the Active Plan status.

# Reasoning and Execution
- Think step by step internally.
- Do not begin implementation until design questions have been answered and the plan has been approved.