"""Unit tests for the cloud providers.

Covers attribute correctness, ``is_configured`` behaviour based on
``OPENROUTER_API_KEY``, and the request/response shape used by the
shared ``OpenAICompatProvider`` base class. Network I/O is mocked.

For live tests (real API calls), see ``tests/integration``.
"""
