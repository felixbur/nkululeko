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
from collections import OrderedDict


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
        layers = ast.literal_eval(glob_conf.config['MODEL']['layers'])
        self.model = self.MLP(feats_train.df.shape[1], layers, self.class_num).to(self.device)
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

    class MLP_test(torch.nn.Module):
        def __init__(self, i, layers, o):
            super().__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(i, layers['l1']),
                torch.nn.ReLU(),
                torch.nn.Linear(layers['l1'], layers['l2']),
                torch.nn.ReLU(),
                torch.nn.Linear(layers['l2'], o)
            )
        def forward(self, x):
            # x: (batch_size, channels, samples)
            x = x.squeeze(dim=1)
            return self.linear(x)

    class MLP(torch.nn.Module):
        def __init__(self, i, layers, o):
            super().__init__()
            sorted_layers = sorted(layers.items(), key=lambda x: x[1])
            layers = OrderedDict()
            layers['0'] = torch.nn.Linear(i, sorted_layers[0][1])
            layers['0_r'] = torch.nn.ReLU()
            for i in range(0, len(sorted_layers)-1):         
                layers[str(i+1)] = torch.nn.Linear(sorted_layers[i][1], sorted_layers[i+1][1])
                layers[str(i)+'_r'] = torch.nn.ReLU()
            layers[str(len(sorted_layers)+1)] = torch.nn.Linear(sorted_layers[-1][1], o)
            self.linear = torch.nn.Sequential(layers)
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
