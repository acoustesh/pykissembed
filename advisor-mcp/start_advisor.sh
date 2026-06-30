#!/usr/bin/env bash
# Launch the openrouter-advisor MCP server with the API key loaded from
# ~/.config/openrouter/api-key (mode 600, owner-only).
#
# VS Code spawns MCP servers as non-interactive, non-login child processes
# that do NOT source ~/.bashrc or ~/.profile.  This wrapper ensures the
# key reaches the server regardless of how the process is launched.
set -euo pipefail

# --- locate workspace root -------------------------------------------------
# When VS Code expands ${workspaceFolder}, it passes an absolute path as
# the first argument.  Fall back to the script's own location otherwise.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# --- load API key ----------------------------------------------------------
KEY_FILE="${OPENROUTER_API_KEY_FILE:-$HOME/.config/openrouter/api-key}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -r "$KEY_FILE" ]; then
    # shellcheck disable=SC1090
    export OPENROUTER_API_KEY="$(<"$KEY_FILE")"
fi

# --- exec the server -------------------------------------------------------
exec "$WORKSPACE_ROOT/advisor-mcp/.venv/bin/python" \
     "$WORKSPACE_ROOT/advisor-mcp/advisor_mcp_server.py"
