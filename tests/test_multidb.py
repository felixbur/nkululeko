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
    _reuse_train,
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


class TestReuseTrainHelper:
    def test_false_when_not_configured(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        assert not _reuse_train(config)

    def test_false_when_no_exp_section(self):
        assert not _reuse_train(configparser.ConfigParser())

    def test_true_for_true(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config["EXP"]["reuse_train"] = "True"
        assert _reuse_train(config)

    def test_true_for_lowercase_true(self):
        """Regression: eval("true") raises NameError, so a lowercase
        ini value used to crash multidb entirely instead of just being
        falsy or truthy."""
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config["EXP"]["reuse_train"] = "true"
        assert _reuse_train(config)

    def test_false_for_false(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config["EXP"]["reuse_train"] = "False"
        assert not _reuse_train(config)


class TestMainReuseTrain:
    """EXP.reuse_train (opt-in): train each row once and reuse that saved
    model for every other column in the row, instead of retraining per
    (train, test) cell -- see main()'s reuse_train branch."""

    def _run(self, tmp_path, monkeypatch, ini_extra="", fake_nkulu=None, calls=None):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\nreuse_train = True\n"
            f"{ini_extra}"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        if calls is None:
            calls = []
        if fake_nkulu is None:

            def fake_nkulu(tmp_config):
                config = configparser.ConfigParser()
                config.read(tmp_config)
                call = dict(config["EXP"]) | dict(config["DATA"])
                if config.has_section("MODEL"):
                    call |= {f"model.{k}": v for k, v in config["MODEL"].items()}
                calls.append(call)
                is_diagonal = "tests" not in config["DATA"]
                return (0.1, 1) if is_diagonal else (0.5, 1)

        monkeypatch.setattr("nkululeko.multidb.nkulu", fake_nkulu)
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=MagicMock()),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            main()
        return calls

    def test_diagonal_runs_before_its_row_off_diagonal_cells(
        self, tmp_path, monkeypatch
    ):
        calls = self._run(tmp_path, monkeypatch)
        names = [c["name"] for c in calls]
        # Row 'a': diagonal 'a' must precede the 'a'-trained off-diagonal
        # cell (also named 'a' -- same EXP.name so the saved model resolves).
        a_indices = [k for k, n in enumerate(names) if n == "a"]
        assert len(a_indices) == 2
        assert "tests" not in calls[a_indices[0]]
        assert calls[a_indices[0]]["databases"] == "['a']"

    def test_off_diagonal_cell_uses_data_tests_not_pooled_databases(
        self, tmp_path, monkeypatch
    ):
        calls = self._run(tmp_path, monkeypatch)
        off_diagonal = [c for c in calls if "tests" in c]
        assert len(off_diagonal) == 2  # a-vs-b and b-vs-a
        for c in off_diagonal:
            # databases holds only the training db, never both -- that's
            # what makes get_save_name() resolve to the diagonal's model.
            assert c["databases"] in ("['a']", "['b']")
            assert c["tests"] in ("['a']", "['b']")
            assert c["databases"] != c["tests"]

    def test_off_diagonal_test_db_gets_full_split_strategy(self, tmp_path, monkeypatch):
        """Without this, the test database defaults to speaker_split
        (~20% of it), not the whole database."""
        calls = self._run(tmp_path, monkeypatch)
        off_diagonal = [c for c in calls if "tests" in c]
        for c in off_diagonal:
            test_db = ast.literal_eval(c["tests"])[0]
            assert c[f"{test_db}.split_strategy"] == "test"

    def test_diagonal_cell_forces_exp_and_model_save(self, tmp_path, monkeypatch):
        """EXP.save alone isn't enough: MODEL.save (also on by default,
        but a base config could turn it off) gates whether the per-epoch
        model weights get written at all -- reuse needs both forced."""
        calls = self._run(tmp_path, monkeypatch)
        diagonal = [c for c in calls if "tests" not in c]
        assert len(diagonal) == 2  # 'a' and 'b', each trained once
        for c in diagonal:
            assert c["save"] == "True"
            assert c["model.save"] == "True"

    def test_failed_diagonal_skips_its_off_diagonal_cells(self, tmp_path, monkeypatch):
        calls = []

        def fake_nkulu(tmp_config):
            config = configparser.ConfigParser()
            config.read(tmp_config)
            calls.append(dict(config["EXP"]) | dict(config["DATA"]))
            if config["EXP"]["name"] == "a" and "tests" not in config["DATA"]:
                raise NkululukoError("fake: diagonal training failed")
            return 0.5, 1

        self._run(tmp_path, monkeypatch, fake_nkulu=fake_nkulu, calls=calls)

        # nkulu is never called for an off-diagonal cell whose row's
        # diagonal failed -- it must not silently fall through and train
        # on 'a' again while reporting it as some other pair.
        assert not any(
            c["databases"] == "['a']" and "tests" in c for c in calls
        )
        content = (tmp_path / "results.txt").read_text()
        assert "nan" in content.lower()

    def test_reuse_train_rejects_train_extra(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\nreuse_train = True\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
            "[CROSSDB]\ntrain_extra = ['c']\n"
        )
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with pytest.raises(SystemExit):
            main()

    def test_reuse_train_rejects_augment(self, tmp_path, monkeypatch):
        """aug_train.doit() has no DATA.tests fast path -- combined with
        reuse_train, an off-diagonal cell would silently retrain from
        scratch instead of reusing the diagonal's saved model."""
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\nreuse_train = True\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
            "[AUGMENT]\naugment = ['noise']\n"
        )
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with pytest.raises(SystemExit):
            main()

    def test_reuse_train_rejects_use_splits(self, tmp_path, monkeypatch):
        """The off-diagonal reuse branch unconditionally sets
        <test>.split_strategy = "test", ignoring EXP.use_splits -- unlike
        the non-reuse branch, which uses as_test/as_train instead when
        use_splits is set. Silently diverging from that semantics is worse
        than rejecting the combination outright."""
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\nreuse_train = True\n"
            "use_splits = True\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        monkeypatch.setattr("sys.argv", ["multidb", "--config", str(config_path)])

        with pytest.raises(SystemExit):
            main()

    def test_default_behavior_unchanged_when_reuse_train_unset(
        self, tmp_path, monkeypatch
    ):
        """The non-reuse path (existing default) must still retrain for
        every cell, including same-training-data cells within a row."""
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "exp.ini"
        config_path.write_text(
            "[EXP]\nroot = .\ndatabases = ['a', 'b']\n"
            "[DATA]\ntarget = emotion\n[MODEL]\ntype = xgb\n"
        )
        calls = []

        def fake_nkulu(tmp_config):
            config = configparser.ConfigParser()
            config.read(tmp_config)
            calls.append(dict(config["DATA"]))
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

        # 2x2 matrix, every cell retrained: no DATA.tests anywhere, and
        # both off-diagonal cells pool both databases (the pre-existing
        # behavior reuse_train is opt-in to replace).
        assert len(calls) == 4
        assert all("tests" not in c for c in calls)
