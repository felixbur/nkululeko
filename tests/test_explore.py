"""Tests for nkululeko/explore.py's CLI argument handling."""

import pytest


class TestExploreMainConfigArgument:
    def test_config_accepted_positionally(self, capsys, monkeypatch):
        """GH #458: explore should run without --config, like most other
        modules (e.g. nkululeko.augment), accepting the config file as a
        positional argument instead."""
        import nkululeko.explore as explore_mod

        monkeypatch.setattr("sys.argv", ["nkululeko.explore", "positional.ini"])

        with pytest.raises(SystemExit):
            explore_mod.main()

        captured = capsys.readouterr()
        assert "positional.ini" in captured.out

    def test_positional_config_takes_priority_over_default(self, capsys, monkeypatch):
        import nkululeko.explore as explore_mod

        monkeypatch.setattr("sys.argv", ["nkululeko.explore", "mine.ini"])

        with pytest.raises(SystemExit):
            explore_mod.main()

        captured = capsys.readouterr()
        assert "mine.ini" in captured.out
        assert "exp.ini" not in captured.out

    def test_no_args_falls_back_to_default_config(self, capsys, monkeypatch):
        import nkululeko.explore as explore_mod

        monkeypatch.setattr("sys.argv", ["nkululeko.explore"])

        with pytest.raises(SystemExit):
            explore_mod.main()

        captured = capsys.readouterr()
        assert "exp.ini" in captured.out
