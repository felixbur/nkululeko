# model_mlp.py
import ast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

from nkululeko.models.mlp_layers import MLPBlock, load_mlp_state_dict, parse_drop_config
from nkululeko.models.model import Model
from nkululeko.optimizers import get_optimizer
from nkululeko.reporting.reporter import Reporter


class MLPModel(Model):
    """MLP = multi layer perceptron."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        """Constructor, taking all dataframes.

        Args:
            df_train (pd.DataFrame): The train labels.
            df_test (pd.DataFrame): The test labels.
            feats_train (pd.DataFrame): The train features.
            feats_test (pd.DataFrame): The test features.
        """
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("ann")
        self.name = "mlp"
        self.target = self.context.config["DATA"]["target"]
        manual_seed = eval(self.util.config_val("MODEL", "random_seed", "False"))
        if manual_seed:
            self.util.debug(f"seeding random to {manual_seed}")
            torch.manual_seed(int(manual_seed))
        labels = self.context.labels
        self.class_num = len(labels)
        # set up loss criterion
        self._setup_criterion()
        activation, act_func = self._get_activation()
        self.util.debug(f"using activation function: {act_func}")
        # set up the model, use GPU if availabe
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        try:
            layers_string = self.context.config["MODEL"]["layers"]
        except KeyError as ke:
            self.util.error(f"Please provide MODEL layers: {ke}")
        self.util.debug(f"using layers {layers_string}")
        layers = ast.literal_eval(layers_string)
        # with dropout?
        drop = parse_drop_config(self.util, num_layers=len(layers))
        if drop:
            self.util.debug(f"init: training with dropout: {drop}")
        self.model = MLPBlock(
            feats_train.shape[1], layers, self.class_num, drop, activation
        ).to(self.device)

        # domain-adversarial heads on the last hidden layer (MODEL.dann_columns)
        self._init_dann(df_train, self.model.feat_dim)
        params = self._dann_params(self.model)

        # set up optimizer
        self.optimizer, self.learning_rate = get_optimizer(
            params,
            self.util,
            default_lr=0.0001,
            default_optimizer="adamw",
        )

        # batch size
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", 8))
        # number of parallel processes
        self.num_workers = self.n_jobs
        feats_train = self._handle_model_nan(feats_train, "Model, train")
        feats_test = self._handle_model_nan(feats_test, "Model, test")
        # set up the data_loaders
        self.trainloader = self.get_loader(
            feats_train,
            df_train,
            True,
            domain_labels=self._domain_labels(df_train),
        )
        self.testloader = self.get_loader(feats_test, df_test, False)

    def _get_activation(self):
        act_func = self.util.config_val("MODEL", "activation", "relu")
        if act_func == "relu":
            activation = torch.nn.ReLU()
        elif act_func == "tanh":
            activation = torch.nn.Tanh()
        elif act_func == "sigmoid":
            activation = torch.nn.Sigmoid()
        elif act_func == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        else:
            self.util.error(f"unknown activation function: {act_func}")
        return activation, act_func

    def set_testdata(self, data_df, feats_df):
        self.df_test = data_df
        self.feats_test = feats_df
        self.testloader = self.get_loader(feats_df, data_df, False)

    def reset_test(self, df_test, feats_test):
        self.testloader = self.get_loader(feats_test, df_test, False)
        self.df_test = df_test
        self.feats_test = feats_test

    def train(self):
        self.model.train()
        losses = []
        for batch in self.trainloader:
            domain_labels = None
            if self.dann_heads is not None:
                features, labels, domain_labels = batch
            else:
                features, labels = batch

            # moved to the device once: SAM runs compute_loss twice per batch
            features = features.to(self.device)
            labels = labels.to(self.device, dtype=torch.int64)
            if domain_labels is not None:
                domain_labels = domain_labels.to(self.device)

            def compute_loss():
                if domain_labels is None:
                    return self.criterion(self.model(features), labels)
                logits, hidden = self.model(features, return_features=True)
                return self.criterion(logits, labels) + self.dann_heads.loss(
                    hidden, domain_labels
                )

            loss = self._optimize(
                self.optimizer, compute_loss, modules=[self.model]
            )
            if loss is None:  # SAM skipped a non-finite step
                continue
            losses.append(loss.item())
        # an epoch in which SAM skipped every step has no loss
        self.loss = float(np.mean(losses)) if losses else 0.0

    def evaluate(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset), self.class_num)
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        losses = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                # the DANN train loader has a third item (domain labels)
                features, labels = batch[0], batch[1]
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index, :] = model(features.to(device))
                targets[start_index:end_index] = labels
                loss = self.criterion(
                    logits[start_index:end_index, :].to(device),
                    labels.to(device, dtype=torch.int64),
                )
                losses.append(loss.item())

        self.loss_eval = (np.asarray(losses)).mean()
        predictions = logits.argmax(dim=1)
        uar = recall_score(targets.numpy(), predictions.numpy(), average="macro")
        return uar, targets, predictions, logits

    def get_probas(self, logits):
        # make a dataframe for probabilities: softmax first, since logits
        # aren't themselves probabilities (not bounded to [0, 1], don't sum
        # to 1) - the saved per-class columns, and anything derived from
        # them (uncertainty, session averaging, calibration, a probability
        # threshold on ROC) were silently wrong without this (GH #446).
        # argmax (the predicted label) is unaffected either way, since
        # softmax is monotonic.
        probs = torch.softmax(logits, dim=1).numpy()
        # Columns follow the network's outputs, i.e. every encoded class
        # 0..class_num-1, even if the test split lacks some of them.
        proba_d = {c: list(probs.T[c]) for c in range(self.class_num)}
        probas = pd.DataFrame(proba_d)
        probas = probas.set_index(self.df_test.index)
        return probas

    def predict(self):
        _, truths, predictions, logits = self.evaluate(
            self.model, self.testloader, self.device
        )
        uar, _, _, _ = self.evaluate(self.model, self._eval_trainloader(), self.device)
        probas = self.get_probas(logits)
        report = Reporter(
            truths,
            predictions,
            self.run,
            self.epoch,
            probas=probas,
            context=self.context,
        )
        try:
            report.result.loss = self.loss
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        try:
            report.result.loss_eval = self.loss_eval
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        report.result.train = uar
        return report

    def get_predictions(self):
        _, _, predictions, logits = self.evaluate(
            self.model, self.testloader, self.device
        )
        return (predictions.numpy(), self.get_probas(logits))

    def get_loader(self, df_x, df_y, shuffle, domain_labels=None):
        # df_x.values re-materializes the whole (n_samples, n_features)
        # array on every access; when df_x isn't a single consolidated
        # block (e.g. after feature balancing concatenates several), that
        # materialization itself is O(n) -- doing it once per row inside
        # this loop made building the loader O(n^2) (GH #444: 16114 rows
        # after balancing = ros took over 10 minutes; ~30s once hoisted).
        x_values = df_x.values
        y_values = df_y[self.target].values
        data = []
        for i in range(len(df_x)):
            item = [x_values[i], y_values[i]]
            if domain_labels is not None:
                item.append(domain_labels[i])
            data.append(item)
        if shuffle:  # the train loader
            return self._make_train_loader(data, df_y, self.batch_size)
        return torch.utils.data.DataLoader(
            data, shuffle=False, batch_size=self.batch_size
        )

    def predict_shap(self, features):
        # predict outputs for all samples in SHAP format (pd. dataframe)
        results = []
        for index, row in features.iterrows():
            feats = row.values
            res_dict = self.predict_sample(feats)
            class_key = max(res_dict, key=res_dict.get)
            results.append(class_key)
        return results

    def predict_sample(self, features):
        """Predict one sample."""
        with torch.no_grad():
            features = torch.from_numpy(features)
            features = features.reshape(-1, 1).T
            logits = self.model(features.to(self.device))
            # logits = self.model(features)
        # if tensor conver to cpu
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu()
        a = logits.numpy()
        res = {}
        for i in range(len(a[0])):
            res[i] = a[0][i]
        return res

    def store(self):
        torch.save(self.model.state_dict(), self.store_path)

    def store_as_onnx(self):
        pass

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        # name = f'{self.util.get_exp_name()}_{run}_{epoch:03d}.model'
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        layers = ast.literal_eval(self.context.config["MODEL"]["layers"])
        self.store_path = dir + name
        drop = parse_drop_config(self.util, num_layers=len(layers))
        if drop:
            self.util.debug(f"loading: dropout set to: {drop}")
        activation, _ = self._get_activation()
        self.model = MLPBlock(
            self.feats_train.shape[1], layers, self.class_num, drop, activation
        ).to(self.device)
        load_mlp_state_dict(self.model, self.store_path, self.util)
        self.model.eval()

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        from pathlib import Path

        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        layers = ast.literal_eval(self.context.config["MODEL"]["layers"])
        self.store_path = path
        drop = parse_drop_config(self.util, num_layers=len(layers))
        if drop:
            self.util.debug(f"dropout set to: {drop}")
        activation, _ = self._get_activation()
        self.model = MLPBlock(
            self.feats_train.shape[1], layers, self.class_num, drop, activation
        ).to(self.device)
        load_mlp_state_dict(self.model, self.store_path, self.util)
        self.model.eval()
