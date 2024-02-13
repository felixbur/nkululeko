""" 
model_cnn.py

Inspired by code from Su Lei

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import ast
import numpy as np
from sklearn.metrics import recall_score
from collections import OrderedDict
from PIL import Image

from nkululeko.utils.util import Util
import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model
from nkululeko.reporter import Reporter
from nkululeko.losses.loss_softf1loss import SoftF1Loss


class CNN_model(Model):
    """CNN = convolutional neural net"""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("ann")
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
        self.util.debug(f"using model with cross entropy loss function")
        # set up the model
        self.device = self.util.config_val("MODEL", "device", "cpu")
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
        self.model = myCNN(layers, self.class_num).to(self.device)
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

        # Define transformations
        transformations = transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        train_set = self.Dataset_image(
            feats_train, df_train, self.target, transformations
        )
        test_set = self.Dataset_image(feats_test, df_test, self.target, transformations)
        # Define data loaders
        self.trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.testloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    class Dataset_image(Dataset):
        def __init__(self, df_feats, df_labels, target, transform=None):
            self.df_feats = df_feats
            self.df_labels = df_labels
            self.transform = transform
            self.target = target

        def __len__(self):
            return len(self.df_feats)

        def __getitem__(self, idx):
            # Load the image file
            img_path = self.df_feats.iloc[idx, 0]
            image = Image.open(img_path)
            # Get emotion label
            label = self.df_labels[self.target].iloc[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    def set_testdata(self, data_df, feats_df):
        test_set = self.Dataset_image(feats_df, data_df)
        # Define data loaders
        self.testloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def reset_test(self, df_test, feats_test):
        self.set_testdata(df_test, feats_test)

    def train(self):
        self.model.train()
        losses = []
        for images, labels in self.trainloader:
            logits = self.model(images.to(self.device))
            loss = self.criterion(logits, labels.to(self.device, dtype=torch.int64))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def evaluate_model(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset), self.class_num)
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        losses = []
        with torch.no_grad():
            for index, (images, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index, :] = model(images.to(device))
                targets[start_index:end_index] = labels
                loss = self.criterion(
                    logits[start_index:end_index, :],
                    labels.to(self.device, dtype=torch.int64),
                )
                losses.append(loss.item())

        self.loss_eval = (np.asarray(losses)).mean()
        predictions = logits.argmax(dim=1)
        uar = recall_score(targets.numpy(), predictions.numpy(), average="macro")
        return uar, targets, predictions

    def predict(self):
        _, truths, predictions = self.evaluate_model(
            self.model, self.testloader, self.device
        )
        uar, _, _ = self.evaluate_model(self.model, self.trainloader, self.device)
        report = Reporter(truths, predictions, self.run, self.epoch)
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
        _, truths, predictions = self.evaluate_model(
            self.model, self.testloader, self.device
        )
        return predictions.numpy()

    def predict_sample(self, features):
        """Predict one sample"""
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features).to(self.device))
        a = logits.numpy()
        res = {}
        for i in range(len(a[0])):
            res[i] = a[0][i]
        return res

    def store(self):
        torch.save(self.model.state_dict(), self.store_path)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        # name = f'{self.util.get_exp_name()}_{run}_{epoch:03d}.model'
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        self.device = self.util.config_val("MODEL", "device", "cpu")
        layers = ast.literal_eval(glob_conf.config["MODEL"]["layers"])
        self.store_path = dir + name
        drop = self.util.config_val("MODEL", "drop", False)
        if drop:
            self.util.debug(f"loading: dropout set to: {drop}")
        self.model = myCNN(layers, self.class_num).to(self.device)
        self.model.load_state_dict(torch.load(self.store_path))
        self.model.eval()

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, "rb") as handle:
            self.device = self.util.config_val("MODEL", "device", "cpu")
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


class myCNN(torch.nn.Module):
    def __init__(self, layers, class_num):
        sorted_layers = sorted(layers.items(), key=lambda x: x[1])
        l1 = sorted_layers[0][1]
        l2 = sorted_layers[1][1]
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, class_num)

    def forward(self, x):
        # -> n, 3, 256, 256
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 126, 126
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 61, 61
        x = x.view(-1, 16 * 61 * 61)  # -> n, 59536
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 7
        return x
