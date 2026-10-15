# model.py
import ast
import pickle
import random

import audeer
import numpy as np
import pandas as pd
import sklearn.utils
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold

from nkululeko.constants import PREDICTED_LABEL_KEY
from nkululeko.experiment_context import ContextAware
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.pickle_integrity import (
    checksum_path,
    save_checksum,
    verify_checksum,
)
from nkululeko.utils.util import Util


def _finite_gradients(optimizer):
    """True if no gradient of the optimizer's parameters is NaN/inf."""
    import torch

    norms = [
        p.grad.detach().norm()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    return not norms or bool(torch.isfinite(torch.stack(norms)).all())


class Model(ContextAware):
    """Generic model class for linear (non-neural) algorithms."""

    dann_heads = None  # DannHeads when MODEL.dann_columns is set (see _init_dann)
    # (dataset, batch_size, loader kwargs) of a train loader that draws
    # domain-balanced batches (see _make_train_loader)
    _balanced_train = None
    _trainloader_eval = None

    def _init_dann(self, df_train, feat_dim, cfg=None):
        """Build the domain-adversarial heads (MODEL.dann_columns) on
        self.device; self.dann_heads stays None when DANN is off. Neural
        models call this once their network exists, before the optimizer."""
        from nkululeko.models.domain_adversarial import DannConfig, DannHeads

        self.dann_heads = DannHeads.build(
            df_train,
            feat_dim,
            cfg if cfg is not None else DannConfig.from_util(self.util),
            self.util,
            self.device,
        )

    def _dann_params(self, module):
        """All trainable parameters: `module`'s plus the DANN heads'. Use
        for the optimizer and for gradient clipping alike."""
        params = list(module.parameters())
        if self.dann_heads is not None:
            params += list(self.dann_heads.parameters())
        return params

    def _domain_labels(self, df):
        """(len(df), n_columns) domain labels for a train loader, or None
        when DANN is off."""
        return self.dann_heads.encode(df) if self.dann_heads is not None else None

    def _domain_sampler(self, df, batch_size):
        """A DomainBalancedBatchSampler over the rows of `df` (the train
        split, in the order of the train dataset) if MODEL.domain_balanced_sampling
        is set, else None. Falls back to None (with a warning) if the
        train split has a single domain."""
        if not self.util.config_val_bool("MODEL", "domain_balanced_sampling", False):
            return None
        from nkululeko.data.domain_sampler import DomainBalancedBatchSampler

        column = self.util.get_domain_balanced_column()
        if column not in df.columns:
            self.util.error(
                f"MODEL.domain_balanced_sampling needs the column '{column}' "
                f"in the train data, found: {list(df.columns)}. 'source_db' is "
                "added when several databases are pooled; a split cached "
                "before that needs [DATA] no_reuse = True once."
            )
        if df[column].nunique() < 2:
            self.util.warn(
                f"MODEL.domain_balanced_sampling: the train data has a single "
                f"value of '{column}', there is nothing to balance; using "
                "plain shuffling"
            )
            return None
        try:
            return DomainBalancedBatchSampler(df[column], batch_size)
        except ValueError as e:
            self.util.error(f"MODEL.domain_balanced_sampling: {e}")

    def _make_train_loader(self, dataset, df, batch_size, **loader_kwargs):
        """The DataLoader that trains the model: shuffled batches, or, with
        MODEL.domain_balanced_sampling, batches with the same number of
        samples of every domain. `df` has one row per item of `dataset`, in
        the same order.

        Such a loader yields repeated samples in random order, so it cannot
        be used to predict the train data: use `_eval_trainloader()` for
        that.
        """
        from torch.utils.data import DataLoader

        self._balanced_train = None
        self._trainloader_eval = None
        sampler = self._domain_sampler(df, batch_size)
        if sampler is None:
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
            )
        self.util.debug(
            f"domain-balanced batches: {len(sampler.domains)} domains "
            f"({', '.join(map(str, sampler.domains))}), {sampler.per_domain} "
            f"samples each per batch of {sampler.batch_size}"
        )
        self._balanced_train = (dataset, batch_size, loader_kwargs)
        return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)

    def _eval_trainloader(self):
        """A loader that goes over every train sample once (to predict the
        train data): the train loader itself, unless that one is
        domain-balanced."""
        if self._balanced_train is None:
            return self.trainloader
        if self._trainloader_eval is None:
            from torch.utils.data import DataLoader

            dataset, batch_size, kwargs = self._balanced_train
            kwargs = {k: v for k, v in kwargs.items() if k != "persistent_workers"}
            self._trainloader_eval = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, **kwargs
            )
        return self._trainloader_eval

    def _optimize(
        self,
        optimizer,
        compute_loss,
        clip_params=None,
        max_grad_norm=0.0,
        skip_non_finite=False,
        modules=(),
    ):
        """One optimization step for a neural model's train loop.

        `compute_loss` is a function without arguments that runs the forward
        pass and returns the loss (task loss plus, if used, the DANN loss).
        Returns that loss (a tensor), or None if the step was skipped because
        the loss, or a gradient, is NaN/inf (nothing is updated then). SAM
        always skips such steps, so callers must handle None (leave the batch
        out of the epoch loss); for a plain optimizer skipping is opt-in with
        `skip_non_finite`, so the other models keep their behavior.

        With a plain optimizer this is zero_grad, backward, optional
        gradient clipping (`clip_params` to `max_grad_norm`) and step. With
        SAM (MODEL.sam) `compute_loss` is run twice, because SAM needs the
        gradient at the current weights and at the perturbed ones; both
        runs must therefore do the same computation on the same batch. Pass
        the network(s) as `modules` so that their BatchNorm statistics are
        only updated by the first run.
        """
        import torch

        from nkululeko.optimizers.sam import frozen_batchnorm_stats, is_sam_optimizer

        def backward(loss):
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_grad_norm)

        n_passes = 0

        def closure():
            nonlocal n_passes
            n_passes += 1
            optimizer.zero_grad()
            # only the first pass runs at the real weights; any later one
            # (SAM has exactly one more) runs at the perturbed weights
            with frozen_batchnorm_stats(modules if n_passes > 1 else ()):
                loss = compute_loss()
                backward(loss)
            return loss

        if is_sam_optimizer(optimizer):
            loss = optimizer.step(closure)
            if optimizer.last_step_skipped:  # SAM skipped a non-finite step
                return None
        else:
            loss = compute_loss()
            if skip_non_finite and not torch.isfinite(loss):
                return None
            optimizer.zero_grad()
            backward(loss)
            if skip_non_finite and not _finite_gradients(optimizer):
                optimizer.zero_grad()
                return None
            optimizer.step()
        return loss

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        """Constructor taking the configuration and all dataframes."""
        self.name = "undefined"
        self.df_train, self.df_test, self.feats_train, self.feats_test = (
            df_train,
            df_test,
            feats_train,
            feats_test,
        )
        self.model_type = "classic"
        self.util = Util("model", context=context)
        self.context = self.util.context
        self.target = self.util.config_val("DATA", "target", "emotion")
        self.run = 0
        self.epoch = 0
        self.logo = self.util.config_val("MODEL", "logo", False)
        self.xfoldx = self.util.config_val("MODEL", "k_fold_cross", False)
        self.n_jobs = int(self.util.config_val("MODEL", "n_jobs", "8"))

    def _get_label_smoothing(self):
        """Parse label_smoothing config value for cross-entropy loss.

        Returns 0.0 (no smoothing) by default.
        Accepts a float in [0.0, 1.0] or the string "true" (shorthand for 0.1).
        Invalid or out-of-range values fall back to 0.0 with a warning.
        """
        ls = self.util.config_val("MODEL", "label_smoothing", "False")
        ls_str = str(ls).strip().lower()

        # Explicit false/disabled
        if ls_str in ("false", "no", "none", ""):
            return 0.0

        # Boolean shorthand "true" → default 0.1
        if ls_str == "true":
            self.util.debug("using label smoothing: 0.1 (default)")
            return 0.1

        # Try to parse as float
        try:
            smoothing = float(ls_str)
        except (ValueError, TypeError):
            self.util.warn(
                f"invalid MODEL.label_smoothing value '{ls}', "
                "falling back to 0.0 (no label smoothing)"
            )
            return 0.0

        # Validate range: PyTorch requires 0.0 <= label_smoothing <= 1.0
        if smoothing < 0.0 or smoothing > 1.0:
            self.util.warn(
                f"MODEL.label_smoothing={smoothing} is out of range [0.0, 1.0], "
                "falling back to 0.0"
            )
            return 0.0

        if smoothing > 0.0:
            self.util.debug(f"using label smoothing: {smoothing}")
        return smoothing

    def _setup_criterion(self):
        """Set up the loss criterion from config. Shared by MLP and CNN models.

        Requires self.class_num to be set before calling.
        Sets self.criterion.
        Returns the criterion name string for debug logging.
        """
        import torch

        from nkululeko.losses.loss_softf1loss import SoftF1Loss

        criterion = self.util.config_val("MODEL", "loss", "cross")
        if criterion == "cross":
            label_smoothing = self._get_label_smoothing()
            self.criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
            )
        elif criterion == "f1":
            self.criterion = SoftF1Loss(
                num_classes=self.class_num, weight=None, epsilon=1e-7
            )
        else:
            self.util.error(f"unknown loss function: {criterion}")
        self.util.debug(f"using model with {criterion} loss function")
        return criterion

    def set_model_type(self, type):
        self.model_type = type

    def is_ann(self):
        if (self.model_type == "ann") or (self.model_type == "finetuned"):
            return True
        else:
            return False

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df

    def reset_test(self, df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test

    def set_id(self, run, epoch):
        self.run = run
        self.epoch = epoch
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        self.store_path = dir + name

    def _x_fold_cross(self):
        # ignore train and test sets and do a x-fold-cross  evaluation
        self.util.debug(
            f"ignoring splits and doing {self.xfoldx} fold cross validation"
        )
        feats = pd.concat([self.feats_train, self.feats_test])
        annos = pd.concat([self.df_train, self.df_test])
        targets = annos[self.target]
        _skf = StratifiedKFold(n_splits=int(self.xfoldx))
        class_weight = self.util.config_val("MODEL", "class_weight", False)
        truths, preds, results = [], [], []
        g_index = 0
        # leave-one-speaker loop
        for train_index, test_index in _skf.split(
            feats,
            targets,
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets[train_index]
            with parallel_backend("threading", n_jobs=self.n_jobs):
                if class_weight:
                    sample_weight = sklearn.utils.class_weight.compute_sample_weight(
                        class_weight="balanced", y=train_y
                    )
                    self.clf.fit(train_x, train_y, sample_weight=sample_weight)
                else:
                    self.clf.fit(train_x, train_y)
            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(
                truth_y.astype(float),
                predict_y,
                self.run,
                self.epoch,
                context=self.context,
            )
            self.util.debug(
                f"result for fold {g_index}: {report.get_result().get_test_result()} "
            )
            results.append(float(report.get_result().test))
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1

        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = "truth"
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name="prediction",
        )
        self.truths = truth
        self.preds = pred
        results = np.asarray(results)
        self.util.debug(
            f"KFOLD: {self.xfoldx} folds: mean {results.mean():.3f}, std:"
            f" {results.std():.3f}"
        )

    def _do_logo(self):
        # ignore train and test sets and do a "leave one speaker group out"  evaluation
        logo = int(self.logo)
        feats = pd.concat([self.feats_train, self.feats_test])
        annos = pd.concat([self.df_train, self.df_test])
        targets = annos[self.target]
        _logo = LeaveOneGroupOut()
        class_weight = self.util.config_val("MODEL", "class_weight", False)
        truths, preds, results = [], [], []
        # get unique list of speakers
        speakers = annos["speaker"].unique()
        # check for folds columns
        if "fold" not in annos.columns:
            self.util.debug(f"creating random folds for {logo} groups")
            # create a random dictionary of groups
            sdict = {}
            # randomize the speaker order
            random.shuffle(speakers)
            folds = list(range(logo))
            for i, s in enumerate(speakers):
                sdict[s] = folds[i % len(folds)]
            # add this to the annotations
            annos["fold"] = annos["speaker"].apply(lambda x: str(sdict[x]))
            fold_count = self.logo
        else:
            fold_count = annos["fold"].nunique()
            self.util.debug(f"using existing folds for {fold_count} groups")
        g_index = 0
        self.util.debug(f"ignoring splits and doing LOGO with {fold_count} groups")
        # leave-one-group loop
        for train_index, test_index in _logo.split(
            feats,
            targets,
            groups=annos["fold"],
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets.iloc[train_index]
            with parallel_backend("threading", n_jobs=self.n_jobs):
                if class_weight:
                    sample_weight = sklearn.utils.class_weight.compute_sample_weight(
                        class_weight="balanced", y=train_y
                    )
                    self.clf.fit(train_x, train_y, sample_weight=sample_weight)
                else:
                    self.clf.fit(train_x, train_y)

            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets.iloc[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(
                truth_y.astype(float),
                predict_y,
                self.run,
                self.epoch,
                context=self.context,
            )
            result = report.get_result().get_test_result()
            self.util.debug(f"result for speaker group {g_index}: {result} ")
            results.append(float(report.get_result().test))
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1

        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = "truth"
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name="prediction",
        )
        self.truths = truth
        self.preds = pred
        results = np.asarray(results)
        self.util.debug(
            f"LOGO: {self.logo} folds: mean {results.mean():.3f}, std:"
            f" {results.std():.3f}"
        )
        report.print_logo(results)

    def train(self):
        """Train the model."""
        # # first check if the model already has been trained
        # if os.path.isfile(self.store_path):
        #     self.load(self.run, self.epoch)
        #     self.util.debug(f'reusing model: {self.store_path}')
        #     return

        # then if leave one speaker group out validation is wanted
        if self.logo:
            self._do_logo()
            return
        # then if x fold cross validation is wanted
        if self.xfoldx:
            self._x_fold_cross()
            return

        self.feats_train = self._handle_model_nan(self.feats_train, "Model, train")
        # remove labels from features
        feats = self.feats_train.to_numpy()
        # compute class weights
        if self.util.config_val("MODEL", "class_weight", False):
            self.classes_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight="balanced", y=self.df_train[self.target]
            )

        tuning_params = self.util.config_val("MODEL", "tuning_params", False)
        with parallel_backend("threading", n_jobs=self.n_jobs):
            if tuning_params:
                # tune the model meta parameters
                tuning_params = ast.literal_eval(tuning_params)
                tuned_params = {}
                try:
                    scoring = self.context.config["MODEL"]["scoring"]
                except KeyError:
                    self.util.error("got tuning params but no scoring")
                for param in tuning_params:
                    values = ast.literal_eval(self.context.config["MODEL"][param])
                    tuned_params[param] = values
                self.util.debug(f"tuning on {tuned_params}")
                self.clf = GridSearchCV(
                    self.clf, tuned_params, refit=True, verbose=3, scoring=scoring
                )
                try:
                    class_weight = eval(
                        self.util.config_val("MODEL", "class_weight", "False")
                    )
                    if class_weight:
                        self.util.debug("using class weight")
                        self.clf.fit(
                            feats,
                            self.df_train[self.target],
                            sample_weight=self.classes_weights,
                        )
                    else:
                        self.clf.fit(feats, self.df_train[self.target])
                except KeyError:
                    self.clf.fit(feats, self.df_train[self.target])
                self.util.debug(f"winner parameters: {self.clf.best_params_}")
            else:
                class_weight = self.util.config_val("MODEL", "class_weight", False)
                if class_weight:
                    self.util.debug("using class weight")
                    self.clf.fit(
                        feats,
                        self.df_train[self.target],
                        sample_weight=self.classes_weights,
                    )
                else:
                    labels = self.df_train[self.target]
                    self.clf.fit(feats, labels)

    def get_predictions(self):
        #        predictions = self.clf.predict(self.feats_test.to_numpy())
        if self.util.exp_is_classification():
            # make a dataframe for the class probabilities
            proba_d = {}
            for c in self.clf.classes_:
                proba_d[c] = []
            # get the class probabilities
            class_probas = self.clf.predict_proba(self.feats_test.to_numpy())
            for i, c in enumerate(self.clf.classes_):
                proba_d[c] = list(class_probas.T[i])
            probas = pd.DataFrame(proba_d)
            probas = probas.set_index(self.feats_test.index)
            # Predicted labels come from predict(), not argmax(predict_proba()):
            # for calibrated classifiers (e.g. SVC(probability=True)),
            # predict_proba is a separate Platt-scaling fit that can re-learn
            # its own class prior independently of the decision function -
            # so for SVC(class_weight="balanced") on imbalanced data,
            # class_weight shifts the decision function correctly, but
            # argmax(predict_proba) threw that shift away and fell back to
            # predicting the majority class (GH #440). predict_proba is
            # still the right source for the probability/uncertainty
            # columns themselves, just not for which label "wins".
            predictions = self.clf.predict(self.feats_test.to_numpy())
        else:
            predictions = self.clf.predict(self.feats_test.to_numpy())
            probas = None

        return predictions, probas

    def predict(self):
        self.feats_test = self._handle_model_nan(self.feats_test, "Model, test")
        if self.logo or self.xfoldx:
            report = Reporter(
                self.truths.astype(float),
                self.preds,
                self.run,
                self.epoch,
                context=self.context,
            )
            return report
        """Predict the whole eval feature set"""
        predictions, probas = self.get_predictions()

        report = Reporter(
            self.df_test[self.target].to_numpy().astype(float),
            predictions,
            self.run,
            self.epoch,
            probas=probas,
            context=self.context,
        )
        report.print_probabilities()
        return report

    def _handle_model_nan(self, feats, context):
        """Handle feature NaNs without changing row alignment with labels."""
        return self.util.handle_nan(
            feats,
            context=context,
            strategy=self.util.config_val("MODEL", "nan_strategy", "zero"),
            allow_drop=False,
        )

    def get_type(self):
        return "generic"

    def predict_sample(self, features):
        """Predict one sample"""
        prediction = {}
        if self.util.exp_is_classification():
            # get the class probabilities
            predictions = self.clf.predict_proba(features)
            for i in range(len(self.clf.classes_)):
                cat = self.clf.classes_[i]
                prediction[cat] = predictions[0][i]
            # The winning label comes from predict(), not
            # argmax(predict_proba) - see get_predictions() above for why
            # (GH #440). Stashed under a sentinel key; callers must pop it
            # out before treating the rest of this dict as per-class
            # probabilities.
            prediction[PREDICTED_LABEL_KEY] = self.clf.predict(features)[0]
        else:
            predictions = self.clf.predict(features)
            prediction = predictions[0]
        return prediction

    def store(self):
        with open(self.store_path, "wb") as handle:
            pickle.dump(self.clf, handle)
        save_checksum(self.store_path)

    @staticmethod
    def sidecar_paths(model_path):
        """Files written next to a stored model that go away with it."""
        return [checksum_path(model_path)]

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        try:
            path = dir + name
            verify_checksum(path)
            with open(path, "rb") as handle:
                self.clf = pickle.load(handle)
        except FileNotFoundError as fe:
            self.util.error(
                f"Did you forget to store your models? needs: \n[MODEL]\nsave=True\n{fe}"
            )

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        verify_checksum(path)
        with open(path, "rb") as handle:
            self.clf = pickle.load(handle)

    # next function exports the model to onnx
    def export_onnx(self, onnx_path, input_shape=None):
        """Export the trained sklearn model to ONNX format.

        Args:
            onnx_path (str): Path to save the ONNX model.
            input_shape (tuple, optional): Shape of the input features. If None, inferred from feats_train.
        """
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        if not hasattr(self, "clf"):
            self.util.error("No trained model found to export.")
            return

        if input_shape is None:
            n_features = self.feats_train.shape[1]
            initial_type = [("input", FloatTensorType([None, n_features]))]
        else:
            initial_type = [("input", FloatTensorType(input_shape))]

        onnx_model = convert_sklearn(self.clf, initial_types=initial_type)
        with open(audeer.path(onnx_path), "wb") as f:
            f.write(onnx_model.SerializeToString())
        self.util.debug(f"Model exported to ONNX at {onnx_path}")
