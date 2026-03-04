# test_naming.py - unit tests for nkululeko/utils/naming.py
import configparser
import sys
import unittest

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


def make_util(extra_sections=""):
    """Helper: create a Util with a minimal config."""
    c = configparser.ConfigParser()
    c.read_string(
        f"""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
[PLOT]
{extra_sections}
"""
    )
    glob_conf.config = c
    return Util("test")


class TestNamingMixin(unittest.TestCase):

    def test_get_model_description_basic(self):
        u = make_util()
        desc = u.get_model_description()
        self.assertIn("svm", desc)
        self.assertIn("os", desc)

    def test_get_exp_name(self):
        u = make_util()
        name = u.get_exp_name()
        self.assertIn("emodb", name)
        self.assertIn("emotion", name)
        self.assertIn("svm", name)
        # no double underscores
        self.assertNotIn("__", name)

    def test_get_exp_name_only_data(self):
        u = make_util()
        name = u.get_exp_name(only_data=True)
        self.assertEqual(name, "emodb")

    def test_get_target_name(self):
        u = make_util()
        self.assertEqual(u.get_target_name(), "emotion")

    def test_get_model_type(self):
        u = make_util()
        self.assertEqual(u.get_model_type(), "svm")

    def test_get_model_type_missing(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u.get_model_type(), "")

    def test_get_feat_type_string_plain(self):
        u = make_util()
        self.assertEqual(u._get_feat_type_string(), "os_")

    def test_get_feat_type_string_list(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = ["os", "mfcc"]
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_feat_type_string(), "os-mfcc_")

    def test_get_layer_string_empty(self):
        u = make_util()
        self.assertEqual(u._get_layer_string(), "")

    def test_get_layer_string_dict(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = mlp
layers = {"a": 64, "b": 128}
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        layer_str = u._get_layer_string()
        self.assertIn("64", layer_str)
        self.assertIn("128", layer_str)

    def test_get_layer_string_list(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = mlp
layers = [64, 128]
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        layer_str = u._get_layer_string()
        self.assertIn("64", layer_str)
        self.assertIn("128", layer_str)

    def test_adm_branch_suffix_tsp(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = adm
adm.branches = time,spectral,phase
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_adm_branch_suffix(), "_tsp")

    def test_adm_branch_suffix_ts(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = adm
adm.branches = time,spectral
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_adm_branch_suffix(), "_ts")

    def test_adm_branch_suffix_not_adm(self):
        u = make_util()
        self.assertEqual(u._get_adm_branch_suffix(), "")

    def test_aug_suffix_present(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
[AUGMENT]
augment = ["traditional"]
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_aug_suffix(), "_aug_traditional")

    def test_aug_suffix_absent(self):
        u = make_util()
        self.assertEqual(u._get_aug_suffix(), "")

    def test_aug_suffix_multiple(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
[AUGMENT]
augment = ["traditional", "auglib"]
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_aug_suffix(), "_aug_traditional_auglib")

    def test_get_plot_name_default(self):
        u = make_util()
        # no [PLOT] name key → falls back to get_exp_name
        self.assertEqual(u.get_plot_name(), u.get_exp_name())

    def test_get_plot_name_custom(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
[PLOT]
name = my_custom_plot
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u.get_plot_name(), "my_custom_plot")

    def test_get_data_name(self):
        u = make_util()
        self.assertEqual(u.get_data_name(), "emodb")

    def test_get_feattype_name(self):
        u = make_util()
        # get_feattype_name uses ast.literal_eval, so type must be a list literal
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = ["os"]
""")
        glob_conf.config = c
        u2 = Util("test")
        self.assertEqual(u2.get_feattype_name(), "os")


if __name__ == "__main__":
    unittest.main()
