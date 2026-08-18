# naming.py - mixin for experiment/model naming helpers
import ast
import os

# MODEL.type values backed by an artificial neural network (see
# Model.is_ann() and the model_type "ann"/"finetuned" tags set by these
# models' __init__). These are the only ones reading MODEL.optimizer,
# MODEL.drop, MODEL.activation and MODEL.loss (learning_rate is shared
# with xgb below, which reads it too but as a boosting hyperparameter).
ANN_MODEL_TYPES = frozenset({"cnn", "mlp", "mlp_reg", "adm", "finetune"})
# MODEL.type values backed by a kernel SVM — only these read
# MODEL.C_val / MODEL.kernel.
SVM_MODEL_TYPES = frozenset({"svm", "svr"})

# Maps a MODEL.<key> naming option to the MODEL.type values it's actually
# read by, so result filenames only mention parameters the chosen model
# type uses. Keys absent from this map (e.g. MODEL.class_weight,
# MODEL.logo, MODEL.k_fold_cross, all FEATS.* options) apply regardless of
# model type and are always included when set.
MODEL_OPTION_TYPES = {
    "C_val": SVM_MODEL_TYPES,
    "kernel": SVM_MODEL_TYPES,
    "drop": ANN_MODEL_TYPES,
    "activation": ANN_MODEL_TYPES,
    "loss": ANN_MODEL_TYPES,
    "optimizer": ANN_MODEL_TYPES,
    "learning_rate": ANN_MODEL_TYPES | {"xgb"},
}


class NamingMixin:
    """Mixin providing experiment and model naming methods for Util."""

    def get_save_name(self):
        """Return a relative path to a name to save the experiment."""
        store = self.get_path("store")
        return f"{store}/{self.get_exp_name()}.pkl"

    def get_pred_name(self):
        results_dir = self.get_path("res_dir")
        target = self.get_target_name()
        pred_name = self.get_model_description()
        return f"{results_dir}/pred_{target}_{pred_name}"

    def print_results_to_store(self, name: str, contents: str) -> str:
        """Write contents to a result file.

        Args:
            name (str): the (sub) name of the file

        Returns:
            str: The path to the file
        """
        results_dir = self.get_path("res_dir")
        pred_name = self.get_model_description()
        path = os.path.join(results_dir, f"{name}_{pred_name}.txt")
        with open(path, "a") as f:
            f.write(contents)
        return path

    def _get_value_descript(self, section, name):
        if self.config_val(section, name, False):
            val = self.config_val(section, name, False)
            val = str(val).strip(".")
            return f"_{name}-{str(val)}"
        return ""

    def get_data_name(self):
        """Get a string as name from all databases that are used."""
        return "_".join(ast.literal_eval(self.config["DATA"]["databases"]))

    def get_feattype_name(self):
        """Get a string as name from all feature sets that are used."""
        return "_".join(ast.literal_eval(self.config["FEATS"]["type"]))

    def get_exp_name(self, only_train=False, only_data=False):
        trains_val = self.config_val("DATA", "trains", False)
        if only_train and trains_val:
            ds = "-".join(ast.literal_eval(self.config["DATA"]["trains"]))
        else:
            ds = "-".join(ast.literal_eval(self.config["DATA"]["databases"]))
        return_string = f"{ds}"
        if not only_data:
            mt = self.get_model_description()
            target = self.get_target_name()
            return_string = return_string + "_" + target + "_" + mt
        return return_string.replace("__", "_")

    def get_target_name(self):
        """Get a string as name from all target sets that are used."""
        return self.config["DATA"]["target"]

    def get_model_type(self):
        try:
            return self.config["MODEL"]["type"]
        except KeyError:
            return ""

    def _get_feat_type_string(self):
        """Return feature type as a dash-joined string with trailing underscore."""
        ft_value = self.config["FEATS"]["type"]
        if (
            isinstance(ft_value, str)
            and ft_value.startswith("[")
            and ft_value.endswith("]")
        ):
            return "-".join(ast.literal_eval(ft_value)) + "_"
        return ft_value + "_"

    def _get_layer_string(self):
        """Return sorted layer sizes as a dash-joined string."""
        layer_s = self.config_val("MODEL", "layers", False)
        if not layer_s:
            return ""
        layers = ast.literal_eval(layer_s)
        if isinstance(layers, list):
            layers = {str(i): v for i, v in enumerate(layers)}
        sorted_layers = sorted(layers.items(), key=lambda x: x[1])
        return "-".join(str(v) for _, v in sorted_layers)

    def _get_adm_branch_suffix(self):
        """Return ADM branch suffix (e.g. 'tsp', 'ts', 's') if model type is adm."""
        if self.get_model_type() != "adm":
            return ""
        branches_str = self.config_val("MODEL", "adm.branches", "time,spectral,phase")
        branches = [b.strip() for b in branches_str.split(",") if b.strip()]
        if not branches:
            return ""
        return "_" + "".join(b[0] for b in branches)

    def _get_aug_suffix(self):
        """Return augmentation suffix if [AUGMENT] augment is configured."""
        aug = self.config_val("AUGMENT", "augment", False)
        if not aug:
            return ""
        try:
            augmentings = "_".join(ast.literal_eval(aug))
        except (ValueError, SyntaxError):
            augmentings = aug
        return f"_aug_{augmentings}"

    def get_model_description(self):
        mt = self.config_val("MODEL", "type", "")
        ft = self._get_feat_type_string()
        layers = self._get_layer_string()
        return_string = f"{mt}_{ft}{layers}"

        options = [
            ["MODEL", "C_val"],
            ["MODEL", "kernel"],
            ["MODEL", "drop"],
            ["MODEL", "activation"],
            ["MODEL", "class_weight"],
            ["MODEL", "loss"],
            ["MODEL", "logo"],
            ["MODEL", "learning_rate"],
            ["MODEL", "optimizer"],
            ["MODEL", "k_fold_cross"],
            ["FEATS", "balancing"],
            ["FEATS", "scale"],
            ["FEATS", "set"],
            ["FEATS", "wav2vec2.layer"],
        ]
        for section, name in options:
            applicable_types = MODEL_OPTION_TYPES.get(name)
            if applicable_types is not None and mt not in applicable_types:
                continue
            return_string += self._get_value_descript(section, name).replace(
                ".", "-"
            )
            return_string = return_string.replace("__", "_").strip("_")

        return_string += self._get_adm_branch_suffix()
        return_string += self._get_aug_suffix()
        return return_string

    def get_plot_name(self):
        try:
            plot_name = self.config["PLOT"]["name"]
        except KeyError:
            plot_name = self.get_exp_name()
        return plot_name
