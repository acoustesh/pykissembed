"""MCP server exposing a single "consult advisor" tool backed by OpenRouter.

The server forwards a prompt directly to a configurable advisor model via the
OpenRouter chat-completions endpoint and returns the advisor's reply.

Configuration is read from environment variables:

* ``OPENROUTER_API_KEY`` - required; your OpenRouter API key.
* ``ADVISOR_MODEL``      - optional; the advisor model to consult
                           (default: ``@preset/glm52``).
* ``ADVISOR_TIMEOUT_S``  - optional; HTTP timeout in seconds (default 180).

Earlier revisions of this module routed the call through OpenRouter's
``openrouter:advisor`` server-side tool, with a cheap executor model driving
the request. That approach proved unreliable: the executor's final
``message.content`` was returned to the client, not the advisor's inner text,
so callers could not tell whether advice had actually been used. This
implementation removes that indirection and calls the advisor directly.
"""

import logging
import os
from typing import Any, Final

import httpx
from mcp.server.fastmcp import FastMCP

_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
_API_KEY_ENV: Final[str] = "OPENROUTER_API_KEY"
_ADVISOR_ENV: Final[str] = "ADVISOR_MODEL"
_TIMEOUT_ENV: Final[str] = "ADVISOR_TIMEOUT_S"
_DEFAULT_ADVISOR: Final[str] = "@preset/glm52"
_DEFAULT_TIMEOUT_S: Final[float] = 180.0

mcp: Final[FastMCP] = FastMCP("openrouter-advisor")

_logger: Final[logging.Logger] = logging.getLogger("openrouter-advisor")


def _build_payload(prompt: str, advisor: str) -> dict[str, Any]:
    """Return a chat-completions body that asks the advisor model directly.

    Args:
        prompt: The question or decision the caller needs guidance on.
        advisor: The OpenRouter model slug to use as the advisor.

    Returns:
        The JSON-serialisable request body.
    """
    return {
        "model": advisor,
        "messages": [{"role": "user", "content": prompt}],
    }


def _extract_content(data: dict[str, Any]) -> str:
    """Pull the assistant message text out of an OpenRouter response.

    Args:
        data: The decoded JSON body returned by OpenRouter.

    Returns:
        The assistant message content, or a human-readable fallback.
    """
    choices: list[dict[str, Any]] = data.get("choices", [])
    if not choices:
        return "No response was returned by the advisor."
    message: dict[str, Any] = choices[0].get("message", {})
    content: Any = message.get("content")
    if isinstance(content, str) and content:
        return content
    return "The advisor returned an empty response."


@mcp.tool()
async def consult_advisor(prompt: str) -> str:
    """Consult the configured advisor model about a hard decision.

    Args:
        prompt: The question or decision the caller needs guidance on.

    Returns:
        The advisor model's reply, or an error message describing why the
        consultation failed.
    """
    api_key: str | None = os.environ.get(_API_KEY_ENV)
    if not api_key:
        return f"Missing {_API_KEY_ENV} environment variable."

    advisor: str = os.environ.get(_ADVISOR_ENV, _DEFAULT_ADVISOR)
    try:
        timeout_s: float = float(os.environ.get(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_S))
    except ValueError:
        timeout_s = _DEFAULT_TIMEOUT_S

    payload: dict[str, Any] = _build_payload(prompt, advisor)
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response: httpx.Response = await client.post(
                _ENDPOINT, json=payload, headers=headers
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
    except httpx.HTTPError as error:
        return f"Advisor request failed: {error}"

    return _extract_content(data)


def main() -> None:
    """Run the MCP server over the stdio transport."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    advisor: str = os.environ.get(_ADVISOR_ENV, _DEFAULT_ADVISOR)
    _logger.info("openrouter-advisor starting — model: %s", advisor)
    mcp.run()


if __name__ == "__main__":
    main()
