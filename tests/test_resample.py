"""Tests for nkululeko/resample.py main() CLI entry point.

Regression for https://github.com/felixbur/nkululeko/issues/409: --config
should default to exp.ini in the current directory instead of requiring the
user to always pass --file, --folder, or an explicit --config.
"""

import sys
from unittest.mock import patch

import pytest

from nkululeko.resample import main


class TestResampleConfigDefault:
    def test_config_defaults_to_exp_ini(self):
        """With no --config given, the tool must look for exp.ini in the
        current directory rather than erroring out immediately."""
        with patch.object(sys, "argv", ["resample"]):
            with patch(
                "nkululeko.resample.os.path.isfile", return_value=False
            ) as mock_isfile:
                with pytest.raises(SystemExit):
                    main()

        mock_isfile.assert_called_once_with("exp.ini")

    def test_no_longer_requires_file_folder_or_config_upfront(self, capsys):
        """The old eager "must provide one of --file/--folder/--config"
        check is gone -- omitting all three now falls through to the
        exp.ini default instead of erroring immediately."""
        with patch.object(sys, "argv", ["resample"]):
            with patch("nkululeko.resample.os.path.isfile", return_value=False):
                with pytest.raises(SystemExit):
                    main()

        captured = capsys.readouterr()
        assert "Either --file, --folder, or --config" not in captured.out
        assert "no such file: exp.ini" in captured.out

    def test_explicit_config_still_takes_precedence(self):
        """An explicitly passed --config is used as-is, not overridden by
        the exp.ini default."""
        with patch.object(sys, "argv", ["resample", "--config", "myconf.ini"]):
            with patch(
                "nkululeko.resample.os.path.isfile", return_value=False
            ) as mock_isfile:
                with pytest.raises(SystemExit):
                    main()

        mock_isfile.assert_called_once_with("myconf.ini")
