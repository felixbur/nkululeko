# test_modules.py: basic tests for nkululeko module imports
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nkululeko
from nkululeko.constants import VERSION


class TestModules(unittest.TestCase):
    """Basic module tests for nkululeko package."""

    def test_version(self):
        """Test that version is correctly defined."""
        self.assertEqual(nkululeko.__version__, VERSION)
        self.assertTrue(isinstance(VERSION, str))
        self.assertTrue(len(VERSION) > 0)

    def test_import(self):
        """Test that all nkululeko modules can be imported.

        The nkululeko package exposes these top-level entry-point modules:
        ``nkululeko`` (``nkululeko.nkululeko``, aliased as ``nkululeko.train``),
        ``ensemble``, ``multidb``, ``explore``, ``augment``, ``aug_train``,
        ``predict``, ``segment``, ``resample``, ``optim`` and ``flags``.
        The core modules must import
        with the package's base dependencies; the optional ones are skipped
        gracefully when their optional extras (e.g. torch, torchaudio,
        pyannote, silero) are not installed.
        """
        # Core entry-point modules (base dependencies only).
        import nkululeko.aug_train  # noqa: F401
        import nkululeko.augment  # noqa: F401
        import nkululeko.explore  # noqa: F401
        import nkululeko.flags  # noqa: F401
        import nkululeko.multidb  # noqa: F401
        import nkululeko.nkululeko  # noqa: F401
        import nkululeko.optim  # noqa: F401
        import nkululeko.predict  # noqa: F401
        import nkululeko.train  # noqa: F401

        # Optional entry-point modules (may require optional extras).
        try:
            import nkululeko.resample  # noqa: F401
        except (ImportError, OSError):
            print("Skipping resample module import (requires torchaudio)")

        try:
            import nkululeko.segment  # noqa: F401
        except ImportError:
            print(
                "Skipping segment module import "
                "(may require optional dependencies)"
            )

        try:
            import nkululeko.ensemble  # noqa: F401
        except ImportError:
            print(
                "Skipping ensemble module import "
                "(may require optional dependencies)"
            )

        self.assertTrue(True)  # If imports succeed, test passes


if __name__ == "__main__":
    unittest.main()
