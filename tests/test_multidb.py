"""Tests for nkululeko/multidb.py.

Regression for https://github.com/felixbur/nkululeko/issues/407: the
multidb heatmap and results.txt always labeled the metric "UAR", even for
regression experiments where the actual metric (MSE, MAE, CCC, ...) is
something else entirely.
"""

import ast
import configparser
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nkululeko.multidb import (
    _all_failed,
    _all_zero,
    _copy_cached_features,
    _metric_label,
    _no_reuse,
    main,
    plot_heatmap,
)
from nkululeko.utils.errors import NkululukoError


def _make_config(exp_type=None, measure=None):
    config = configparser.ConfigParser()
    config.add_section("EXP")
    if exp_type is not None:
        config["EXP"]["type"] = exp_type
    if measure is not None:
        config.add_section("MODEL")
        config["MODEL"]["measure"] = measure
    return config


class TestMetricLabel:
    def test_classification_default_is_uar(self):
        config = _make_config(exp_type="classification")
        assert _metric_label(config) == "UAR"

    def test_regression_default_is_mse(self):
        config = _make_config(exp_type="regression")
        assert _metric_label(config) == "MSE"

    def test_regression_custom_measure_mae(self):
        config = _make_config(exp_type="regression", measure="mae")
        assert _metric_label(config) == "MAE"

    def test_regression_custom_measure_ccc(self):
        config = _make_config(exp_type="regression", measure="ccc")
        assert _metric_label(config) == "CCC"

    def test_regression_custom_measure_pcc(self):
        config = _make_config(exp_type="regression", measure="pcc")
        assert _metric_label(config) == "PCC"

    def test_classification_custom_measure_eer(self):
        config = _make_config(exp_type="classification", measure="eer")
        assert _metric_label(config) == "EER"

    def test_missing_exp_type_defaults_to_classification(self):
        config = _make_config()
        assert _metric_label(config) == "UAR"

    def test_missing_exp_section_defaults_to_classification(self):
        config = configparser.ConfigParser()
        assert _metric_label(config) == "UAR"

    def test_missing_model_section_uses_default_measure(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config["EXP"]["type"] = "regression"
        assert _metric_label(config) == "MSE"


class TestPlotHeatmap:
    """Covers both outputs plot_heatmap produces: results.txt *and* the
    heatmap's own title/caption (ax.set_title()) -- the original bug report
    is specifically about the heatmap label, not just the text file.

    Plotting calls (plt.figure/savefig/close, sn.heatmap) are patched out:
    it keeps these tests fast and independent of a real matplotlib/seaborn
    rendering backend, while still letting us capture the exact caption
    string passed to ax.set_title().
    """

    def _run_plot_heatmap(self, config, tmp_path):
        results = np.array([[0.1, 0.2], [0.3, 0.4]])
        last_epochs = np.array([[1, 2], [3, 4]])
        mock_ax = MagicMock()
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=mock_ax),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            plot_heatmap(
                results,
                last_epochs,
                ["a", "b"],
                str(tmp_path / "heatmap.png"),
                config,
                ["a", "b"],
            )
        return mock_ax

    def test_results_txt_labels_regression_metric(self, tmp_path):
        config = _make_config(exp_type="regression", measure="mae")
        config["EXP"]["root"] = str(tmp_path)

        self._run_plot_heatmap(config, tmp_path)

        content = (tmp_path / "results.txt").read_text()
        assert "Mean MAE:" in content
        assert "Mean UAR:" not in content

    def test_heatmap_caption_labels_regression_metric(self, tmp_path):
        config = _make_config(exp_type="regression", measure="mae")
        config["EXP"]["root"] = str(tmp_path)

        mock_ax = self._run_plot_heatmap(config, tmp_path)

        caption = mock_ax.set_title.call_args.args[0]
        assert "Mean MAE:" in caption
        assert "UAR" not in caption

    def test_results_txt_still_labels_uar_for_classification(self, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)

        self._run_plot_heatmap(config, tmp_path)

        content = (tmp_path / "results.txt").read_text()
        assert "Mean UAR:" in content

    def test_heatmap_caption_still_labels_uar_for_classification(self, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)

        mock_ax = self._run_plot_heatmap(config, tmp_path)

        caption = mock_ax.set_title.call_args.args[0]
        assert "Mean UAR:" in caption

    def test_results_txt_omits_epochs_section_when_all_zero(self, tmp_path):
        """A non-ANN model forces epoch_num=1, so last_epochs is always
        exactly 0 everywhere -- that's not useful information and
        shouldn't clutter results.txt."""
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)
        results = np.array([[0.1, 0.2], [0.3, 0.4]])
        last_epochs = np.zeros((2, 2))
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            plot_heatmap(
                results, last_epochs, ["a", "b"], str(tmp_path / "heatmap.png"),
                config, ["a", "b"],
            )

        content = (tmp_path / "results.txt").read_text()
        assert "epochs" not in content.lower()

    def test_results_txt_keeps_epochs_section_when_not_all_zero(self, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)

        self._run_plot_heatmap(config, tmp_path)

        content = (tmp_path / "results.txt").read_text()
        assert "all epochs" in content


class TestPlotHeatmapMeanRowColumn:
    """issue #424: the plotted heatmap should get an extra "mean (cross)" row/column
    with the diagonal (self-result) excluded, not just the text summary's
    existing (diagonal-included) mean_non_diag figure."""

    def _run(self, results, labels, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)
        last_epochs = np.ones_like(results, dtype=int)
        mock_ax = MagicMock()
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=mock_ax) as mock_heatmap,
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            plot_heatmap(
                results, last_epochs, labels, str(tmp_path / "heatmap.png"), config, labels
            )
        df_cm = mock_heatmap.call_args.args[0]
        return df_cm

    def test_mean_row_and_column_present(self, tmp_path):
        results = np.array([[0.6, 0.3], [0.25, 0.4]])
        df_cm = self._run(results, ["a", "b"], tmp_path)

        assert list(df_cm.columns) == ["a", "b", "mean (cross)"]
        assert list(df_cm.index) == ["a", "b", "mean (cross)"]

    def test_mean_values_exclude_the_diagonal(self, tmp_path):
        # a's self-result (0.6) and b's self-result (0.4) must not count
        # towards either db's Mean row/column.
        results = np.array([[0.6, 0.3], [0.25, 0.4]])
        df_cm = self._run(results, ["a", "b"], tmp_path)

        assert df_cm.loc["a", "mean (cross)"] == pytest.approx(0.3)  # a trained, tested on b
        assert df_cm.loc["b", "mean (cross)"] == pytest.approx(0.25)  # b trained, tested on a
        assert df_cm.loc["mean (cross)", "a"] == pytest.approx(0.25)  # a tested, trained on b
        assert df_cm.loc["mean (cross)", "b"] == pytest.approx(0.3)  # b tested, trained on a
        assert df_cm.loc["mean (cross)", "mean (cross)"] == pytest.approx(0.275)  # overall cross mean

    def test_original_cells_unchanged(self, tmp_path):
        results = np.array([[0.6, 0.3], [0.25, 0.4]])
        df_cm = self._run(results, ["a", "b"], tmp_path)

        assert df_cm.loc["a", "a"] == pytest.approx(0.6)
        assert df_cm.loc["a", "b"] == pytest.approx(0.3)
        assert df_cm.loc["b", "a"] == pytest.approx(0.25)
        assert df_cm.loc["b", "b"] == pytest.approx(0.4)

    def test_single_database_does_not_crash(self, tmp_path):
        """dim == 1 has no cross-db result to average -- Mean row/column
        should be NaN, not a crash (division by zero or int(nan))."""
        results = np.array([[0.6]])
        df_cm = self._run(results, ["a"], tmp_path)

        assert list(df_cm.columns) == ["a", "mean (cross)"]
        assert np.isnan(df_cm.loc["a", "mean (cross)"])
        assert np.isnan(df_cm.loc["mean (cross)", "mean (cross)"])

    def test_failed_pair_does_not_poison_other_aggregates(self, tmp_path):
        """A single failed pair (main()'s per-pair try/except records it as
        NaN) must not turn every other row/column's Mean into NaN too --
        only the failed cell's own row/column loses that one data point."""
        results = np.array(
            [
                [0.6, 0.3, 0.2],
                [0.25, 0.4, np.nan],  # (train=b, test=c) failed
                [0.1, 0.35, 0.7],
            ]
        )
        df_cm = self._run(results, ["a", "b", "c"], tmp_path)

        # row 'a' Mean unaffected by b's failed cell
        assert df_cm.loc["a", "mean (cross)"] == pytest.approx((0.3 + 0.2) / 2)
        # column 'c' Mean: only 'a' contributes (b's cell is NaN)
        assert df_cm.loc["mean (cross)", "c"] == pytest.approx(0.2)
        # row 'b' Mean: only the (b, a) cell contributes
        assert df_cm.loc["b", "mean (cross)"] == pytest.approx(0.25)


class TestAllZero:
    """issue #424: "why is a zero matrix shown" -- multidb should refuse to
    plot a heatmap when every single result is exactly 0.0 (a silent
    all-empty-splits/no-labels failure), rather than write out a
    meaningless all-zero heatmap."""

    def test_all_zero_matrix(self):
        assert _all_zero(np.zeros((3, 3)))

    def test_one_nonzero_cell_is_not_all_zero(self):
        results = np.zeros((3, 3))
        results[1, 2] = 0.5
        assert not _all_zero(results)

    def test_normal_matrix_is_not_all_zero(self):
        assert not _all_zero(np.array([[0.6, 0.3], [0.25, 0.4]]))

    def test_all_nan_matrix_is_all_zero(self):
        """A fully-failed run (every pair NaN) also has nothing meaningful
        to plot, same as an all-0.0 run."""
        assert _all_zero(np.full((3, 3), np.nan))


class TestAllFailed:
    """Every pair failing (all NaN, from main()'s per-pair try/except) is a
    distinct case from _all_zero's "computed but 0.0" -- nothing was
    computed at all."""

    def test_all_nan_matrix(self):
        assert _all_failed(np.full((3, 3), np.nan))

    def test_one_success_is_not_all_failed(self):
        results = np.full((3, 3), np.nan)
        results[0, 0] = 0.6
        assert not _all_failed(results)

    def test_all_zero_but_no_nan_is_not_all_failed(self):
        assert not _all_failed(np.zeros((3, 3)))


class TestMainAcceptsPositionalConfig:
    """multidb should accept the config file as a plain positional
    argument, not just via --config (matching nkululeko.py's own
    main())."""

    def _run_with_argv(self, tmp_path, monkeypatch, argv):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a']\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        monkeypatch.setattr("nkululeko.multidb.nkulu", lambda tmp_config: (0.5, 1))
        monkeypatch.setattr(
            "sys.argv", ["multidb"] + [a.format(config=str(config_path)) for a in argv]
        )
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()
        return tmp_path

    def test_positional_config_file(self, tmp_path, monkeypatch):
        self._run_with_argv(tmp_path, monkeypatch, ["{config}"])
        assert (tmp_path / "results.txt").exists()

    def test_flag_config_still_works(self, tmp_path, monkeypatch):
        self._run_with_argv(tmp_path, monkeypatch, ["--config", "{config}"])
        assert (tmp_path / "results.txt").exists()


class TestMainSuppressesAllZeroEpochsPrint:
    """A non-ANN model forces epoch_num=1, so last_epochs is always
    exactly 0 for every pair -- printing that matrix is just confusing
    noise, easily mistaken for the (real, meaningful) results matrix
    also being all-zero."""

    def test_all_zero_last_epochs_not_printed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        monkeypatch.setattr("nkululeko.multidb.nkulu", lambda tmp_config: (0.5, 0))
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()

        out = capsys.readouterr().out
        assert "array([[0., 0.]" not in out


class TestMainPerPairFailureHandling:
    """A single pair's genuine data/config error (e.g. train/test label
    sets that don't overlap, raised as NkululukoError deep inside
    datasplitter) must not abort the whole multidb run before a heatmap
    for every *other* pair can be written."""

    def test_one_failing_pair_does_not_abort_the_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )

        def fake_nkulu(tmp_config):
            config = configparser.ConfigParser()
            config.read(tmp_config)
            if config["EXP"]["name"] == "b_vs_a":
                raise NkululukoError("fake: train/test label mismatch")
            return 0.5, 1

        monkeypatch.setattr("nkululeko.multidb.nkulu", fake_nkulu)
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()  # must not raise / sys.exit

        content = (tmp_path / "results.txt").read_text()
        assert "nan" in content.lower()


class TestNoReuse:
    def test_false_when_not_configured(self):
        config = configparser.ConfigParser()
        config.add_section("DATA")
        assert not _no_reuse(config)

    def test_true_when_data_no_reuse_set(self):
        config = configparser.ConfigParser()
        config.add_section("DATA")
        config["DATA"]["no_reuse"] = "True"
        assert _no_reuse(config)

    def test_true_when_feats_no_reuse_set(self):
        config = configparser.ConfigParser()
        config.add_section("FEATS")
        config["FEATS"]["no_reuse"] = "true"
        assert _no_reuse(config)

    def test_false_for_falsy_value(self):
        config = configparser.ConfigParser()
        config.add_section("DATA")
        config["DATA"]["no_reuse"] = "False"
        assert not _no_reuse(config)


class TestCopyCachedFeatures:
    """Sharing whole-database feature caches (<db>_<feats_type>_all.*)
    across multidb pairs avoids re-extracting the same database's features
    once per pair it appears in."""

    def test_copies_all_suffixed_files(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        (src / "emovo_os_all.pkl").write_text("fake features")

        _copy_cached_features(str(src), str(dst))

        assert (dst / "emovo_os_all.pkl").read_text() == "fake features"

    def test_does_not_overwrite_existing_destination_file(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "emovo_os_all.pkl").write_text("new")
        (dst / "emovo_os_all.pkl").write_text("existing")

        _copy_cached_features(str(src), str(dst))

        assert (dst / "emovo_os_all.pkl").read_text() == "existing"

    def test_ignores_split_specific_files(self, tmp_path):
        """feats_train.pkl/feats_test.pkl/traindf.csv/testdf.csv are
        pair-specific (the actual rows differ per pair) and must never be
        shared across pairs, unlike the whole-database `_all` cache."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        (src / "feats_train.pkl").write_text("train-specific")
        (src / "testdf.csv").write_text("test-specific")

        _copy_cached_features(str(src), str(dst))

        assert not (dst / "feats_train.pkl").exists()
        assert not (dst / "testdf.csv").exists()

    def test_missing_source_dir_is_a_no_op(self, tmp_path):
        dst = tmp_path / "dst"
        _copy_cached_features(str(tmp_path / "does_not_exist"), str(dst))
        assert not dst.exists()


class TestMainSharesFeatureCacheAcrossPairs:
    """issue follow-up: the same database's whole-database features
    shouldn't be re-extracted for every multidb pair it appears in."""

    def test_second_occurrence_of_a_database_reuses_cached_features(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        extraction_log = []

        def fake_nkulu(tmp_config):
            config = configparser.ConfigParser()
            config.read(tmp_config)
            name = config["EXP"]["name"]
            store = os.path.join(".", name, "store")
            os.makedirs(store, exist_ok=True)
            for db in ast.literal_eval(config["DATA"]["databases"]):
                cache_file = os.path.join(store, f"{db}_os_all.pkl")
                if os.path.isfile(cache_file):
                    extraction_log.append((name, db, "reused"))
                else:
                    extraction_log.append((name, db, "extracted"))
                    with open(cache_file, "w") as f:
                        f.write("fake")
            return 0.5, 1

        monkeypatch.setattr("nkululeko.multidb.nkulu", fake_nkulu)
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()

        for db in ("a", "b"):
            events = [e for e in extraction_log if e[1] == db]
            assert events[0][2] == "extracted", f"{db}: {events}"
            assert all(e[2] == "reused" for e in events[1:]), f"{db}: {events}"

    def test_no_reuse_disables_sharing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\n"
            "[DATA]\ntarget = emotion\nno_reuse = True\n[MODEL]\ntype = xgb\n"
        )
        extraction_log = []

        def fake_nkulu(tmp_config):
            config = configparser.ConfigParser()
            config.read(tmp_config)
            name = config["EXP"]["name"]
            store = os.path.join(".", name, "store")
            os.makedirs(store, exist_ok=True)
            for db in ast.literal_eval(config["DATA"]["databases"]):
                cache_file = os.path.join(store, f"{db}_os_all.pkl")
                extraction_log.append((name, db, os.path.isfile(cache_file)))
                with open(cache_file, "w") as f:
                    f.write("fake")
            return 0.5, 1

        monkeypatch.setattr("nkululeko.multidb.nkulu", fake_nkulu)
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()

        # DATA.no_reuse=True: no pair should ever find a pre-populated
        # cache file waiting for it (nothing was copied in).
        assert not any(already_there for _, _, already_there in extraction_log)
        assert not os.path.isdir("_feat_cache")
