# model_mlp.py
from util import Util 
import glob_conf
from model import Model
from reporter import Reporter
from result import Result
import torch
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


class MLP_model(Model):
    """MLP = multi layer perceptron"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        Model.__init__(self, df_train, df_test, feats_train, feats_test)
        self.util = Util()
        self.target = glob_conf.config['DATA']['target']
        labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
        self.class_num = len(labels)
        # set up loss criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        # set up the data_loaders
        self.trainloader = self.get_loader(feats_train.df, df_train)
        self.testloader = self.get_loader(feats_test.df, df_test)
        # set up the model
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.model = self.MLP(self.class_num, feats_train.df.shape[1]).to(self.device)
        # set up regularization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)


    def train(self):
        self.model.train()
        losses = []
        for features, labels in self.trainloader:
            logits = self.model(features.to(self.device))
            loss = self.criterion(logits, labels.to(self.device))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def predict(self):        
        _, truths, predictions = self.evaluate_model(self.model, self.testloader, self.device)
        uar, _, _ = self.evaluate_model(self.model, self.trainloader, self.device)
        report = Reporter(truths, predictions)
        report.result.loss = self.loss
        report.result.train = uar
        return report

    def get_loader(self, df_x, df_y):
        data=[]
        for i in range(len(df_x)):
            data.append([df_x.values[i], df_y[self.target][i]])
        return torch.utils.data.DataLoader(data, shuffle=True, batch_size=8)


    class MLP(torch.nn.Module):
        def __init__(self, out_size, in_size):
            super().__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(in_size, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, out_size)
            )
        def forward(self, x):
            # x: (batch_size, channels, samples)
            x = x.squeeze(dim=1)
            return self.linear(x)


    def evaluate_model(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset), self.class_num)
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index, :] = model(features.to(device))
                targets[start_index:end_index] = labels

        predictions = logits.argmax(dim=1)
        uar = recall_score(targets.numpy(), predictions.numpy(), average='macro')
        return uar, targets, predictions
