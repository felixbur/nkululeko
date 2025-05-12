import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nkululeko
from nkululeko.constants import VERSION


class TestBasic(unittest.TestCase):
    """Basic tests for nkululeko package."""

    def test_version(self):
        """Test that version is correctly defined."""
        self.assertEqual(nkululeko.__version__, VERSION)
        self.assertTrue(isinstance(VERSION, str))
        self.assertTrue(len(VERSION) > 0)

    def test_import(self):
        """Test that main modules can be imported."""
        import nkululeko.nkululeko
        import nkululeko.explore
        import nkululeko.augment
        import nkululeko.demo
        import nkululeko.predict
        import nkululeko.resample
        import nkululeko.segment
        import nkululeko.test
        import nkululeko.ensemble
        self.assertTrue(True)  # If imports succeed, test passes


if __name__ == '__main__':
    unittest.main()
