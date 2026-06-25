# errors.py
"""Custom exception classes for nkululeko."""


class NkululukoError(RuntimeError):
    """Raised when nkululeko encounters a fatal error.

    This replaces direct sys.exit() calls so that nkululeko can be used
    as a library without terminating the host process.
    """
