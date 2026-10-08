"""Parsing of the ``[MODEL] random_seed`` config value."""


def parse_seed(value):
    """Return ``[MODEL] random_seed`` as an int, or None if no seed is set.

    The config value is a raw INI string: ``False`` (or empty/``None``/``0``)
    means "do not seed", an integer string is the seed. Nothing is evaluated.

    Raises:
        ValueError: if the value is neither of those.
    """
    text = str(value).strip().lower()
    if text in ("", "false", "none", "0"):
        return None
    return int(text)
