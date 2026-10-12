"""Tests for nkululeko/explore.py's CLI argument handling."""

import os
import tempfile
from unittest.mock import MagicMock

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


class TestSavedExperimentIsStale:
    """Review on #468: explore's saved-experiment pickle (expr.load()) is a
    second cache layer separate from the traindf.csv/testdf.csv split
    cache -- loading it bypasses load_datasets()/Dataset.prepare()
    entirely, so should_reuse_split()'s source-mtime check never runs on
    this path. _saved_experiment_is_stale() applies the same check here."""

    def _util(self, data_files):
        util = MagicMock()
        util.config_val_data.side_effect = (
            lambda name, key, default: data_files.get(name, default)
        )
        return util

    def test_false_when_no_csv_backed_dataset(self):
        from nkululeko.explore import _saved_experiment_is_stale

        util = self._util({"mydb": "mydb"})  # audb-managed, no local file
        config = {"DATA": {"databases": "['mydb']"}}

        assert not _saved_experiment_is_stale("/no/such/save.pkl", config, util)

    def test_false_when_source_is_older_than_save_file(self):
        from nkululeko.explore import _saved_experiment_is_stale

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            save_path = os.path.join(tmp, "save.pkl")
            open(csv_path, "w").close()
            open(save_path, "w").close()
            os.utime(csv_path, (1, 1))

            util = self._util({"mydb": csv_path})
            config = {"DATA": {"databases": "['mydb']"}}

            assert not _saved_experiment_is_stale(save_path, config, util)

    def test_true_when_source_is_newer_than_save_file(self):
        from nkululeko.explore import _saved_experiment_is_stale

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            save_path = os.path.join(tmp, "save.pkl")
            open(save_path, "w").close()
            os.utime(save_path, (1, 1))
            open(csv_path, "w").close()

            util = self._util({"mydb": csv_path})
            config = {"DATA": {"databases": "['mydb']"}}

            assert _saved_experiment_is_stale(save_path, config, util)

    def test_false_when_save_file_does_not_exist_yet(self):
        """First run: there's nothing to compare against, so this isn't
        "stale" -- it's just not there, handled separately by the
        FileNotFoundError fallback."""
        from nkululeko.explore import _saved_experiment_is_stale

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            open(csv_path, "w").close()

            util = self._util({"mydb": csv_path})
            config = {"DATA": {"databases": "['mydb']"}}

            assert not _saved_experiment_is_stale(
                os.path.join(tmp, "no_such_save.pkl"), config, util
            )

    def test_false_when_databases_key_is_missing(self):
        from nkululeko.explore import _saved_experiment_is_stale

        util = self._util({})
        assert not _saved_experiment_is_stale("/no/such/save.pkl", {"DATA": {}}, util)
