"""Shared HTTP status classification for transient provider failures."""

_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 600


def is_retryable_http_error(
    exc: Exception,
    timeout_error: type[Exception],
    *,
    http_error: type[Exception] | None = None,
) -> bool:
    """Classify timeout, rate-limit, and server failures for retry.

    Parameters
    ----------
    exc : Exception
        Failure raised by a provider request.
    timeout_error : type[Exception]
        Provider timeout exception type.
    http_error : type[Exception] | None
        Required HTTP exception type, if the provider exposes one.

    Returns
    -------
    bool
        Whether the failure is transient under the provider's exception contract.
    """
    if isinstance(exc, timeout_error):
        return True
    # Jev accepts response-bearing failures from its generic request boundary;
    # Voyage requires the transport's HTTPError type before reading a status.
    if http_error is not None and not isinstance(exc, http_error):
        return False
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return isinstance(status_code, int) and (
        status_code == _HTTP_TOO_MANY_REQUESTS
        or _HTTP_SERVER_ERROR_MIN <= status_code < _HTTP_SERVER_ERROR_MAX
    )
