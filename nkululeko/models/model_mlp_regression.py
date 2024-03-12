# model_mlp.py
from nkululeko.utils.util import Util
import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model
from nkululeko.reporter import Reporter
import torch
import ast
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import OrderedDict
from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
import os


class MLP_Reg_model(Model):
    """MLP = multi layer perceptron"""

    is_classifier = False

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("ann")
        self.target = glob_conf.config["DATA"]["target"]
        labels = glob_conf.labels
        self.class_num = len(labels)
        # set up loss criterion
        criterion = self.util.config_val("MODEL", "loss", "mse")
        if criterion == "mse":
            self.criterion = torch.nn.MSELoss()
        elif criterion == "mae":
            self.criterion = torch.nn.L1Loss()
        elif criterion == "1-ccc":
            self.criterion = ConcordanceCorCoeff()
        else:
            self.util.error(f"unknown loss function: {criterion}")
        self.util.debug(f"training model with {criterion} loss function")
        # set up the model
        self.device = self.util.config_val("MODEL", "device", "cpu")
        layers_string = glob_conf.config["MODEL"]["layers"]
        self.util.debug(f"using layers {layers_string}")
        try:
            layers = ast.literal_eval(layers_string)
        except KeyError as ke:
            self.util.error(f"Please provide MODEL layers: {ke}")
        drop = self.util.config_val("MODEL", "drop", False)
        if drop:
            self.util.debug(f"training with dropout: {drop}")
        self.model = self.MLP(feats_train.shape[1], layers, 1, drop).to(self.device)
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
        self.num_workers = int(self.util.config_val("MODEL", "num_workers", 5))
        # set up the data_loaders
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
        self.trainloader = self.get_loader(feats_train, df_train, True)
        self.testloader = self.get_loader(feats_test, df_test, False)

    def set_testdata(self, data_df, feats_df):
        self.testloader = self.get_loader(feats_df, data_df, False)

    def train(self):
        loss = self.train_epoch(
            self.model,
            self.trainloader,
            self.device,
            self.optimizer,
        )
        return loss

    def predict(self):
        _, truths, predictions = self.evaluate_model(
            self.model, self.testloader, self.device
        )
        result, _, _ = self.evaluate_model(self.model, self.trainloader, self.device)
        report = Reporter(truths.numpy(), predictions.numpy(), self.run, self.epoch)
        try:
            report.result.loss = self.loss
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        try:
            report.result.loss_eval = self.loss_eval
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        report.result.train = result
        return report

    def get_loader(self, df_x, df_y, shuffle):
        data_set = self.Dataset(df_y, df_x, self.target)
        loader = torch.utils.data.DataLoader(
            dataset=data_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return loader

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, df, features, label: str):
            super().__init__()
            self.df = df
            self.df_features = features
            self.label = label

        def __len__(self):
            return len(self.df)

        def __getitem__(self, item):
            index = self.df.index[item]
            features = self.df_features.loc[index, :].values.astype("float32").squeeze()
            labels = (
                np.array([self.df.loc[index, self.label]]).astype("float32").squeeze()
            )
            return features, labels

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

    def train_epoch(self, model, loader, device, optimizer):
        # first check if the model already has been trained
        # if os.path.isfile(self.store_path):
        #     self.load(self.run, self.epoch)
        #     self.util.debug(f'reusing model: {self.store_path}')
        #     return
        self.model.train()
        losses = []
        for features, labels in loader:
            logits = model(features.to(device)).reshape(-1)
            loss = self.criterion(logits, labels.to(device))
            # print(f'loss: {loss.item()}')
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def evaluate_model(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset))
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        losses = []
        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index] = model(features.to(device)).reshape(-1)
                targets[start_index:end_index] = labels
                loss = self.criterion(
                    logits[start_index:end_index].to(
                        device,
                    ),
                    labels.to(device),
                )
                losses.append(loss.item())
        self.loss_eval = (np.asarray(losses)).mean()

        predictions = logits
        measure = self.util.config_val("MODEL", "measure", "mse")
        if measure == "mse":
            result = mean_squared_error(targets.numpy(), predictions.numpy())
        elif measure == "mae":
            result = mean_absolute_error(targets.numpy(), predictions.numpy())
        elif measure == "ccc":
            result = Reporter.ccc(targets.numpy(), predictions.numpy())
        else:
            self.util.error(f"unknown measure: {measure}")
        return result, targets, predictions

    def store(self):
        torch.save(self.model.state_dict(), self.store_path)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{run}_{epoch:03d}.model"
        self.store_path = dir + name
        self.device = self.util.config_val("MODEL", "device", "cpu")
        layers = ast.literal_eval(glob_conf.config["MODEL"]["layers"])
        drop = self.util.config_val("MODEL", "drop", False)
        if drop:
            self.util.debug(f"training with dropout: {drop}")
        self.model = self.MLP(self.feats_train.shape[1], layers, 1, drop).to(
            self.device
        )
        self.model.load_state_dict(torch.load(dir + name))
        self.model.eval()

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, "rb") as handle:
            self.device = self.util.config_val("MODEL", "device", "cpu")
            layers = ast.literal_eval(glob_conf.config["MODEL"]["layers"])
            self.store_path = path
            drop = self.util.config_val("MODEL", "drop", False)
            if drop:
                self.util.debug(f"training with dropout: {drop}")
            self.model = self.MLP(
                self.feats_train.shape[1], layers, self.class_num, drop
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.store_path))
            self.model.eval()

    def predict_sample(self, features):
        """Predict one sample"""
        with torch.no_grad():
            features = torch.from_numpy(features)
            features = np.reshape(features, (-1, 1)).T
            logits = self.model(features.to(self.device)).reshape(-1)
        a = logits.numpy()
        return a
