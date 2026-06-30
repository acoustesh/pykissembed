Developer: ---
description: "Always-on workspace rules for the repo: fetch current upstream docs from context7 first when needed, then consult the openrouter-advisor MCP tool for hard decisions."
applyTo: "**"
---

# Workspace Rules — speechtext

The decision pipeline for any non-trivial change in this repo is:

1. **`context7`** — pull current upstream documentation for the libraries
   involved, so the advisor (and you) reason against today's API, not
   yesterday's training cut-off.
2. **`openrouter-advisor`** — ask a stronger model for an opinion on the
   hard decision, citing the context7 results.

Both steps are mandatory when the trigger conditions below hold. The
advisor checkpoint must happen before producing a non-trivial Active Plan
or any code, not merely before editing files.

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
- Any change to the MCP server in
  [`advisor-mcp/advisor_mcp_server.py`](../advisor-mcp/advisor_mcp_server.py)
  (httpx, FastMCP, stdio transport).
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
   have what you need, stop and either escalate to the advisor or ask
   the user.

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
| `httpx` | `/encode/httpx` | [`advisor-mcp/advisor_mcp_server.py`](../advisor-mcp/advisor_mcp_server.py) |
| `mcp` (FastMCP) | `/modelcontextprotocol/python-sdk` | [`advisor-mcp/advisor_mcp_server.py`](../advisor-mcp/advisor_mcp_server.py) |
| `pytest`, `pytest-asyncio` | `/pytest-dev/pytest` | [`tests/`](../tests/) |
| OpenRouter Chat Completions API | `/openrouter/awesome-openrouter` (or `/openrouterai/api-reference`) | any provider hitting OpenRouter |

Treat this table as a hint, not a source of truth — context7 ids drift
when publishers reorganise their docs.

---

## 2. Step 2 — `openrouter-advisor` for hard decisions

This workspace exposes a single advisor tool via the `openrouter-advisor`
MCP server (defined in
[`.vscode/mcp.json`](../.vscode/mcp.json) and implemented in
[`advisor-mcp/advisor_mcp_server.py`](../advisor-mcp/advisor_mcp_server.py)):

> **Tool:** `mcp__openrouter-advisor__consult_advisor`
> **Model:** `@preset/glm52` (the `ADVISOR_MODEL` set in `.vscode/mcp.json`;
>   the hardcoded fallback default is the direct slug `z-ai/glm-5.2`)
> **Signature:** `consult_advisor(prompt: str) -> str`

**When the active model is a small / fast / preset model (for example
`@preset/minimax` or any `@preset/*` slug), and you face a hard decision —
ambiguous requirements, conflicting trade-offs, a design choice with
non-obvious consequences, or a task at the edge of the model's competence —
treat `mcp__openrouter-advisor__consult_advisor` as a required checkpoint
after consulting context7 (§1) and before producing a non-trivial Active
Plan or any code.** The advisor should see the context7 findings summarised
in the prompt. Use the advisor's reply to inform your reasoning; you do not
have to follow it verbatim, but you must consult it rather than guess.

### When to consult

Consult the advisor when **any** of the following is true:

- The current model is a `@preset/*` model (e.g. `@preset/minimax`) and the
  question is non-trivial.
- The decision affects architecture, public APIs, schema, or test strategy
  in this repo (see [`schema.py`](../schema.py), [`tools.py`](../tools.py),
  [`segmenter.py`](../segmenter.py), [`advisor-mcp/`](../advisor-mcp/)).
- Multiple plausible approaches exist and the trade-off is not obvious from
  the local code alone.
- The user is correcting you on a previous similar decision — escalate
  before repeating the mistake.
- A request is ambiguous and a clarifying question would lose context;
  consult the advisor first, then optionally clarify.

### When NOT to consult

Skip the advisor for:

- Trivial edits (typos, formatting, single-line renames) where the cost of
  a network round-trip exceeds the benefit.
- Tasks where the local code, tests, or repo memory, **plus** the context7
  results from §1, already give a definitive answer.
- The user provided an explicit step-by-step plan **and** the change is a
  single-file mechanical transformation with no design choices.
- Anything that would require a secret (API key, token, password) to be
  pasted into the prompt — the advisor logs nothing we control, so do not
  send secrets through it.

### How to call it

- Tool identifier (use this exact string): `mcp__openrouter-advisor__consult_advisor`.
- Pass the **full context the advisor needs** in a single `prompt` string:
  the question, the relevant code snippet (short), the constraints, the
  candidate options, and **a short summary of the context7 findings from
  §1** so the advisor reasons against today's API. A good prompt is
  self-contained — the advisor has no memory of this conversation.
- The tool returns a string. Treat that string as advice from a stronger
  model and weigh it against local context. Cite the advisor's key
  recommendation in your final answer when it materially shaped the
  decision.
- Do **not** attempt to call the deprecated server-side
  `openrouter:advisor` tool — it was removed because its output never
  reached `message.content`. Use the MCP tool above only.

### Example

```text
# Step 1 — context7 (already done above; summary you would paste in):
#   httpx 0.27+: AsyncClient(timeout=...) accepts float seconds;
#   AsyncClient.timeout can be set per-request via ``timeout=`` kwarg;
#   raise_for_status() raises on 4xx/5xx.
#
# Step 2 — advisor:
mcp__openrouter-advisor__consult_advisor(
  prompt = "Context7 says httpx supports both client-level and per-request "
           "timeouts via the ``timeout=`` kwarg, and raise_for_status() "
           "surfaces 4xx/5xx.\n\n"
           "I need to choose between asyncio.to_thread and a "
           "ProcessPoolExecutor for parallelizing per-provider "
           "transcription calls in speechtext/process_audio.py. Each "
           "provider call is HTTP-bound (httpx), there are ~8 providers, "
           "and the process is already inside an asyncio event loop. "
           "Trade-offs?"
)
```

---

## 3. Other conventions preserved from this workspace

- Python source lives at the repo root (`process_audio.py`,
  etc, providers in `*_provider.py`, …); tests live in
  [`tests/`](../tests/). New tests must be runnable via `pytest tests/`.
- The Python interpreter for this workspace is `.venv/bin/python`
  (configured in [`.vscode/settings.json`](../.vscode/settings.json)).
- Provider implementations follow the pattern in
  [`schema.py`](../schema.py) and [`tools.py`](../tools.py) — reuse the
  shared schemas instead of redefining them.
- When in doubt about provider-specific behaviour, prefer reading the
  provider module over guessing — and run the §1 → §2 pipeline (context7
  first, then advisor) before making architectural changes that span
  multiple providers. 

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
2. For every non-trivial Active Plan (3+ steps, or any step touching
   architecture, schema, public APIs, or tests), the first line must be one
   of:
   - `Advisory consulted: yes (consult_advisor, <ref-or-timestamp>)`
   - `Advisory consulted: no (exemption: <a|b|c>, reason: <one sentence>)`
3. If the advisory status is `no`, cite the specific exemption and
   one-sentence reason before any other plan content. The valid exemptions
   are:
   - `a`: trivial edit, such as a typo, formatting-only change, or
     single-line rename.
   - `b`: local code, tests, repo memory, and any applicable context7
     results already give a definitive answer.
   - `c`: the user provided an explicit step-by-step plan **and** the change
     is a single-file mechanical transformation with no design choices.
4. Keep the Active Plan in this chat/session context.
5. Before each coding step, restate:
   - the current plan step number
   - whether the step is unchanged, changed, or completed
   - the reason for any change
6. After each change, update the Active Plan status.

# Reasoning and Execution
- Think step by step internally.
- Do not begin implementation until design questions have been answered and the plan has been approved.