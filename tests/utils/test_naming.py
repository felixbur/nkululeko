# test_naming.py - unit tests for nkululeko/utils/naming.py
import configparser
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

    # --- Additional tests for uncovered lines ---

    def test_get_save_name(self):
        u = make_util()
        save_name = u.get_save_name()
        self.assertIn("test", save_name)
        self.assertTrue(save_name.endswith(".pkl"))

    def test_get_pred_name(self):
        u = make_util()
        pred_name = u.get_pred_name()
        self.assertIn("pred", pred_name)
        self.assertIn("emotion", pred_name)

    def test_print_results_to_store(self):
        import tempfile
        import os

        u = make_util()
        with tempfile.TemporaryDirectory() as tmpdir:
            u.config["EXP"]["root"] = tmpdir
            u.config["EXP"]["name"] = "myexp"
            result_path = u.print_results_to_store("test", "test content")
            # file must exist
            self.assertTrue(os.path.exists(result_path))
            # must live under {root}/{name}/results/
            self.assertIn(os.path.join(tmpdir, "myexp", "results"), result_path)
            # file must contain the written text
            with open(result_path) as fh:
                self.assertIn("test content", fh.read())

    def test_get_value_descript_with_value(self):
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
C_val = 1.0
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        result = u._get_value_descript("MODEL", "C_val")
        self.assertIn("C_val", result)
        self.assertIn("1", result)

    def test_get_value_descript_without_value(self):
        u = make_util()
        result = u._get_value_descript("MODEL", "nonexistent")
        self.assertEqual(result, "")

    def test_get_exp_name_with_only_train(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb", "emodb2"]
trains = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        name = u.get_exp_name(only_train=True)
        self.assertIn("emodb", name)
        self.assertNotIn("emodb2", name)

    def test_adm_branch_suffix_empty(self):
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
adm.branches = 
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertEqual(u._get_adm_branch_suffix(), "")

    def test_model_description_includes_optimizer_for_ann(self):
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
optimizer = adamw
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertIn("optimizer-adamw", u.get_model_description())

    def test_model_description_excludes_optimizer_for_non_ann(self):
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
optimizer = adamw
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertNotIn("optimizer", u.get_model_description())

    def test_model_description_includes_learning_rate_for_xgb(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = xgb
learning_rate = 0.3
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertIn("learning_rate-0", u.get_model_description())

    def test_model_description_excludes_learning_rate_for_svm(self):
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
learning_rate = 0.3
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        self.assertNotIn("learning_rate", u.get_model_description())

    def test_model_description_includes_c_val_kernel_for_svm(self):
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
C_val = 1.0
kernel = linear
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        desc = u.get_model_description()
        self.assertIn("C_val-1", desc)
        self.assertIn("kernel-linear", desc)

    def test_model_description_excludes_c_val_kernel_for_xgb(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = xgb
C_val = 1.0
kernel = linear
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        desc = u.get_model_description()
        self.assertNotIn("C_val", desc)
        self.assertNotIn("kernel", desc)

    def test_model_description_excludes_drop_activation_loss_for_svm(self):
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
drop = 0.5
activation = relu
loss = cross
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        desc = u.get_model_description()
        self.assertNotIn("drop", desc)
        self.assertNotIn("activation", desc)
        self.assertNotIn("loss", desc)

    def test_aug_suffix_invalid_syntax(self):
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
augment = not_a_list
""")
        glob_conf.config = c
        u = Util("test")
        # Should handle invalid syntax gracefully
        result = u._get_aug_suffix()
        self.assertIn("aug", result)


if __name__ == "__main__":
    unittest.main()
