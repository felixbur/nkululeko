# model_mlp.py
import ast
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

import nkululeko.glob_conf as glob_conf
from nkululeko.losses.loss_softf1loss import SoftF1Loss
from nkululeko.models.model import Model
from nkululeko.reporting.reporter import Reporter


class MLPModel(Model):
    """MLP = multi layer perceptron."""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor, taking all dataframes.

        Args:
            df_train (pd.DataFrame): The train labels.
            df_test (pd.DataFrame): The test labels.
            feats_train (pd.DataFrame): The train features.
            feats_test (pd.DataFrame): The test features.
        """
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("ann")
        self.name = "mlp"
        self.target = glob_conf.config["DATA"]["target"]
        labels = glob_conf.labels
        self.class_num = len(labels)
        # set up loss criterion
        criterion = self.util.config_val("MODEL", "loss", "cross")
        if criterion == "cross":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion == "f1":
            self.criterion = SoftF1Loss(
                num_classes=self.class_num, weight=None, epsilon=1e-7
            )
        else:
            self.util.error(f"unknown loss function: {criterion}")
        self.util.debug("using model with cross entropy loss function")
        # set up the model, use GPU if availabe
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        try:
            layers_string = glob_conf.config["MODEL"]["layers"]
        except KeyError as ke:
            self.util.error(f"Please provide MODEL layers: {ke}")
        self.util.debug(f"using layers {layers_string}")
        layers = ast.literal_eval(layers_string)
        # with dropout?
        drop = self.util.config_val("MODEL", "drop", False)
        if drop:
            self.util.debug(f"init: training with dropout: {drop}")
        self.model = self.MLP(feats_train.shape[1], layers, self.class_num, drop).to(
            self.device
        )
        self.learning_rate = float(
            self.util.config_val("MODEL", "learning_rate", 0.0001)
        )
        # set up regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        # batch size
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", 8))
        # number of parallel processes
        self.num_workers = self.n_jobs
        if feats_train.isna().to_numpy().any():
            self.util.debug(
                f"Model, train: replacing {feats_train.isna().sum().sum()} NANs"
                " with 0"
            )
            feats_train = feats_train.fillna(0)
        if feats_test.isna().to_numpy().any():
            self.util.debug(
                f"Model, test: replacing {feats_test.isna().sum().sum()} NANs" " with 0"
            )
            feats_test = feats_test.fillna(0)
        # set up the data_loaders
        self.trainloader = self.get_loader(feats_train, df_train, True)
        self.testloader = self.get_loader(feats_test, df_test, False)

    def set_testdata(self, data_df, feats_df):
        self.testloader = self.get_loader(feats_df, data_df, False)

    def reset_test(self, df_test, feats_test):
        self.testloader = self.get_loader(feats_test, df_test, False)

    def train(self):
        self.model.train()
        losses = []
        for features, labels in self.trainloader:
            logits = self.model(features.to(self.device))
            loss = self.criterion(logits, labels.to(self.device, dtype=torch.int64))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def evaluate(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset), self.class_num)
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        losses = []
        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
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
        # make a dataframe for probabilites (logits)
        proba_d = {}
        classes = self.df_test[self.target].unique()
        classes.sort()
        for c in classes:
            proba_d[c] = []
        for i, c in enumerate(classes):
            proba_d[c] = list(logits.numpy().T[i])
        probas = pd.DataFrame(proba_d)
        probas = probas.set_index(self.df_test.index)
        return probas

    def predict(self):
        _, truths, predictions, logits = self.evaluate(
            self.model, self.testloader, self.device
        )
        uar, _, _, _ = self.evaluate(self.model, self.trainloader, self.device)
        probas = self.get_probas(logits)
        report = Reporter(truths, predictions, self.run, self.epoch, probas=probas)
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
        _, _, predictions, _ = self.evaluate(self.model, self.testloader, self.device)
        return predictions.numpy()

    def get_loader(self, df_x, df_y, shuffle):
        data = []
        for i in range(len(df_x)):
            data.append([df_x.values[i], df_y[self.target].iloc[i]])
        return torch.utils.data.DataLoader(
            data, shuffle=shuffle, batch_size=self.batch_size
        )

    class MLP(torch.nn.Module):
        def __init__(self, i, layers, o, drop):
            super().__init__()
            sorted_layers = sorted(layers.items(), key=lambda x: x[1])
            layers = OrderedDict()
            layers["0"] = torch.nn.Linear(i, sorted_layers[0][1])
            layers["0_r"] = torch.nn.ReLU()
            for i in range(0, len(sorted_layers) - 1):
                layers[str(i + 1)] = torch.nn.Linear(
                    sorted_layers[i][1], sorted_layers[i + 1][1]
                )
                if drop:
                    layers[str(i) + "_d"] = torch.nn.Dropout(float(drop))
                layers[str(i) + "_r"] = torch.nn.ReLU()
            layers[str(len(sorted_layers) + 1)] = torch.nn.Linear(
                sorted_layers[-1][1], o
            )
            self.linear = torch.nn.Sequential(layers)

        def forward(self, x):
            # x: (batch_size, channels, samples)
            x = x.squeeze(dim=1).float()
            return self.linear(x)

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
            features = np.reshape(features, (-1, 1)).T
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
        layers = ast.literal_eval(glob_conf.config["MODEL"]["layers"])
        self.store_path = dir + name
        drop = self.util.config_val("MODEL", "drop", False)
        if drop:
            self.util.debug(f"loading: dropout set to: {drop}")
        self.model = self.MLP(
            self.feats_train.shape[1], layers, self.class_num, drop
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.store_path))
        self.model.eval()

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, "rb") as handle:
            cuda = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = self.util.config_val("MODEL", "device", cuda)
            layers = ast.literal_eval(glob_conf.config["MODEL"]["layers"])
            self.store_path = path
            drop = self.util.config_val("MODEL", "drop", False)
            if drop:
                self.util.debug(f"dropout set to: {drop}")
            self.model = self.MLP(
                self.feats_train.shape[1], layers, self.class_num, drop
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.store_path))
            self.model.eval()
