"""aug_train: augment the training data only, name the added database after
the augmentation (including the RawBoost settings)."""

import configparser

import pytest

import nkululeko.aug_train as aug_train
import nkululeko.glob_conf as glob_conf
from nkululeko.experiment_context import get_context, set_context


class _Stop(Exception):
    pass


@pytest.fixture(autouse=True)
def _restore_context():
    """Creating an Experiment makes its context the current one; later tests
    (and the threads of some of them) expect the default one."""
    context = get_context()
    yield
    set_context(context)
    glob_conf.config = None


def _write_config(tmp_path, augment):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "x", "root": str(tmp_path)}
    config["DATA"] = {
        "databases": "['dba']",
        "dba": str(tmp_path / "dba.csv"),
        "dba.type": "csv",
        "target": "label",
    }
    config["FEATS"] = {"type": "['os']"}
    config["MODEL"] = {"type": "xgb"}
    config["AUGMENT"] = augment
    path = tmp_path / "exp.ini"
    with open(path, "w") as f:
        config.write(f)
    return str(path)


def _run(tmp_path, monkeypatch, augment):
    """Run aug_train up to the augmentation and return the settings it hands
    to it (the tmp.ini it writes)."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_augment(tmp_config):
        parsed = configparser.ConfigParser()
        parsed.read(tmp_config)
        seen["config"] = parsed
        raise _Stop

    monkeypatch.setattr(aug_train, "augment", fake_augment)
    with pytest.raises(_Stop):
        aug_train.doit(_write_config(tmp_path, augment))
    glob_conf.config = None
    return seen["config"]


class TestAugTrain:
    def test_only_the_train_data_is_augmented(self, tmp_path, monkeypatch):
        """Experiment.augment() reads DATA.sample_selection (default: all):
        augmented copies of test files would be added to the training data."""
        config = _run(tmp_path, monkeypatch, {"augment": "['rawboost']"})
        assert config["DATA"]["sample_selection"] == "train"
        assert config["AUGMENT"]["sample_selection"] == "train"

    def test_other_rawboost_settings_give_other_files(self, tmp_path, monkeypatch):
        names = set()
        for settings in (
            {"rawboost_algo": "3"},
            {"rawboost_algo": "5"},
            {"rawboost_algo": "5", "rawboost_snr_min": "5"},
        ):
            config = _run(
                tmp_path, monkeypatch, {"augment": "['rawboost']", **settings}
            )
            names.add(config["AUGMENT"]["result"])
        assert len(names) == 3
        assert "./augmented_rawboost-algo3.csv" in names

    def test_other_methods_keep_their_names(self, tmp_path, monkeypatch):
        config = _run(tmp_path, monkeypatch, {"augment": "['traditional']"})
        assert config["AUGMENT"]["result"] == "./augmented_traditional.csv"
