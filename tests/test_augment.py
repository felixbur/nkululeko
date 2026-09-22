from unittest.mock import patch

import nkululeko.augment as augment_module
from nkululeko.augment import main


class TestMainAcceptsPositionalConfig:
    """augment should accept the config file as a plain positional
    argument, not just via --config (matching nkululeko.py's and
    multidb.py's own main())."""

    def test_positional_config_file(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["augment", "my_conf.ini"])
        with patch.object(augment_module, "doit") as mock_doit:
            main()
        mock_doit.assert_called_once_with("my_conf.ini")

    def test_flag_config_still_works(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["augment", "--config", "my_conf.ini"])
        with patch.object(augment_module, "doit") as mock_doit:
            main()
        mock_doit.assert_called_once_with("my_conf.ini")

    def test_defaults_to_exp_ini_when_neither_given(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["augment"])
        with patch.object(augment_module, "doit") as mock_doit:
            main()
        mock_doit.assert_called_once_with("exp.ini")
