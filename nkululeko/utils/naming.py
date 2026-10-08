# naming.py - mixin for experiment/model naming helpers
import ast
import hashlib
import os
import re

# MODEL.type values backed by an artificial neural network (see
# Model.is_ann() and the model_type "ann"/"finetuned" tags set by these
# models' __init__). All six read MODEL.loss (mlp/cnn via the shared
# Model._setup_criterion(), mlp_reg/adm/finetune/aasist each directly); the
# other ANN-only options below are read by narrower subsets of this group.
ANN_MODEL_TYPES = frozenset({"cnn", "mlp", "mlp_reg", "adm", "finetune", "aasist"})
# MODEL.type values backed by a kernel SVM — only these read
# MODEL.C_val / MODEL.kernel.
SVM_MODEL_TYPES = frozenset({"svm", "svr"})
# MODEL.type values with a configurable layer stack — only these read
# MODEL.layers (adm, finetune and aasist have a fixed architecture instead).
LAYERED_MODEL_TYPES = frozenset({"cnn", "mlp", "mlp_reg"})
# ANN types that build their optimizer via optimizer_factory.get_optimizer()
# and thus read MODEL.optimizer. TunedModel (finetune) configures its own
# optimizer elsewhere and never reads it.
OPTIMIZER_MODEL_TYPES = frozenset({"cnn", "mlp", "mlp_reg", "adm", "aasist"})
# ANN types that read MODEL.drop for dropout. model_adm.py has no dropout
# config of its own.
DROPOUT_MODEL_TYPES = frozenset({"cnn", "mlp", "mlp_reg", "finetune"})
# ANN types with a configurable hidden-layer activation function. cnn, adm
# and finetune don't expose MODEL.activation.
ACTIVATION_MODEL_TYPES = frozenset({"mlp", "mlp_reg"})
# Types that implement domain-adversarial training (MODEL.dann_columns);
# every other type ignores the dann_* keys.
DANN_MODEL_TYPES = frozenset({"aasist", "mlp", "mlp_reg", "cnn", "adm"})



def parse_dann_columns(raw):
    """Parse MODEL.dann_columns (an INI string such as "['source_db']", or
    an already parsed list) into a list of column names; "" and "[]" mean
    DANN off. Raises ValueError if the value is not a list of names."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw.lower() in ("false", "none"):
            return []
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"cannot parse dann_columns {raw!r}") from e
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not all(
        isinstance(c, str) for c in raw
    ):
        raise ValueError(f"dann_columns must be a list of column names: {raw!r}")
    return list(raw)


# Maps a MODEL.<key> naming option to the MODEL.type values it's actually
# read by, so result filenames only mention parameters the chosen model
# type uses. Keys absent from this map (e.g. MODEL.class_weight,
# MODEL.logo, MODEL.k_fold_cross, all FEATS.* options) apply regardless of
# model type and are always included when set.
MODEL_OPTION_TYPES = {
    "C_val": SVM_MODEL_TYPES,
    "kernel": SVM_MODEL_TYPES,
    "drop": DROPOUT_MODEL_TYPES,
    "activation": ACTIVATION_MODEL_TYPES,
    "loss": ANN_MODEL_TYPES,
    "optimizer": OPTIMIZER_MODEL_TYPES,
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

    def safe_filename_component(self, name: str) -> str:
        """Return `name` with any path separators or other filesystem-unsafe
        characters replaced, so it's safe to embed as one component of a
        filename.

        A configurable string (e.g. PLOT.combine_per_speaker.col) otherwise
        gets inserted into result/plot filenames verbatim; if it contains
        "/" or "..", that can create nested paths, make savefig fail because
        the directory doesn't exist, or escape the intended output
        directory. Use this only for the filename, not for display text.
        """
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))
        return safe or "col"

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
        # [EXP] res_name overrides the auto-constructed name entirely (GH
        # #451): with many databases/features/model options, the default
        # name (databases + target + model description) can get too long
        # for a filesystem path.
        res_name = self.config_val("EXP", "res_name", False)
        if res_name:
            # Configuration-controlled and used verbatim in filesystem
            # paths below; sanitize so a stray "/" or ".." can't escape
            # the configured output directory or break a path join.
            return self.safe_filename_component(res_name)
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

    def _get_dann_suffix(self):
        """Return a DANN suffix (e.g. '_dann-source_db') when the model type
        uses MODEL.dann_columns, so DANN runs don't share result/checkpoint
        paths with plain runs or with runs using other DANN settings.
        Non-default lambda/weight/reverse are appended."""
        if self.get_model_type() not in DANN_MODEL_TYPES:
            return ""
        raw = self.config_val("MODEL", "dann_columns", "[]")
        try:
            columns = parse_dann_columns(raw)
        except ValueError:
            self.error(
                f"MODEL.dann_columns = {raw} is not a list of column names; "
                "write it like ['source_db', 'language']"
            )
        if not columns:
            return ""
        # Path-safe, and unambiguous: ['a+b'] and ['a', 'b'] must not share
        # a name, so if sanitizing changed any name, add a hash of the list.
        safe = [self.safe_filename_component(c) for c in columns]
        suffix = "_dann-" + "+".join(safe)
        if safe != columns or any("+" in c for c in columns):
            digest = hashlib.sha1(repr(columns).encode()).hexdigest()[:6]
            suffix += f"-{digest}"
        lambda_ = str(self.config_val("MODEL", "dann_lambda", "1.0"))
        weight = str(self.config_val("MODEL", "dann_weight", "1.0"))
        if float(lambda_) != 1.0:
            suffix += f"-l{lambda_.replace('.', '-')}"
        if float(weight) != 1.0:
            suffix += f"-w{weight.replace('.', '-')}"
        if not self.config_val_bool("MODEL", "dann_reverse", True):
            suffix += "-noreverse"
        return suffix

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
        layers = self._get_layer_string() if mt in LAYERED_MODEL_TYPES else ""
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
        return_string += self._get_dann_suffix()
        return_string += self._get_aug_suffix()
        return return_string

    def get_plot_name(self):
        try:
            plot_name = self.config["PLOT"]["name"]
        except KeyError:
            plot_name = self.get_exp_name()
        return plot_name
