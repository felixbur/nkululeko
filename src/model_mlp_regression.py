# model_mlp.py
from util import Util 
import glob_conf
from model import Model
from reporter import Reporter
import torch
import ast
import numpy as np
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from concordance_cor_coeff import ConcordanceCorCoeff

class MLP_Reg_model(Model):
    """MLP = multi layer perceptron"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        Model.__init__(self, df_train, df_test, feats_train, feats_test)
        self.util = Util()
        self.target = glob_conf.config['DATA']['target']
        labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
        self.class_num = len(labels)
        # set up loss criterion
        criterion = self.util.config_val('MODEL', 'loss', 'mse')
        if criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif criterion == '1-ccc':
            self.criterion = ConcordanceCorCoeff()
        else:
            self.util.error(f'unknown loss function: {criterion}')
        self.util.debug(f'training model with {criterion} loss function')
        # set up the data_loaders
        self.trainloader = self.get_loader(feats_train.df, df_train)
        self.testloader = self.get_loader(feats_test.df, df_test)
        # set up the model
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        layers = ast.literal_eval(glob_conf.config['MODEL']['layers'])
        self.model = self.MLP(feats_train.df.shape[1], layers, 1).to(self.device)
        self.learning_rate = float(self.util.config_val('MODEL', 'learning_rate', 0.0001))
        # set up regularization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def train(self):
        loss = self.train_epoch(self.model, self.trainloader, self.device, self.optimizer, self.criterion)
        return loss

    def predict(self):        
        _, truths, predictions = self.evaluate_model(self.model, self.testloader, self.device)
        result, _, _ = self.evaluate_model(self.model, self.trainloader, self.device)
        report = Reporter(truths.numpy(), predictions.numpy(), self.run, self.epoch)
        try:
            report.result.loss = self.loss
        except AttributeError: # if the model was loaded from disk the loss is unknown
            pass 
        report.result.train = result
        return report

    def get_loader(self, df_x, df_y):
        data_set = self.Dataset(df_y, df_x, self.target)
        loader = torch.utils.data.DataLoader(
            dataset=data_set,
            batch_size=8,
            shuffle=True,
            num_workers=3
        )
        return loader


    class Dataset(torch.utils.data.Dataset):
        def __init__(self, df, features,
                    label: str):
            super().__init__()
            self.df = df
            self.df_features = features
            self.label = label
        def __len__(self):
            return len(self.df)

        def __getitem__(self, item):
            index = self.df.index[item]
            features = self.df_features.loc[index, :].values.astype('float32').squeeze()
            labels = np.array([self.df.loc[index, self.label]]).astype('float32').squeeze()
            return features, labels


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


    def train_epoch(self, model, loader, device, optimizer, criterion):
        model.train()
        losses = []
        for features, labels in loader:
            logits = model(features.to(device)).reshape(-1)
            loss = criterion(logits, labels.to(device))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def evaluate_model(self, model, loader, device):
        logits = torch.zeros(len(loader.dataset))
        targets = torch.zeros(len(loader.dataset))
        model.eval()
        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index] = model(features.to(device)).reshape(-1)
                targets[start_index:end_index] = labels

        predictions = logits
        measure = self.util.config_val('MODEL', 'measure', 'mse')
        if measure == 'mse':
            result = mean_squared_error(targets.numpy(), predictions.numpy())
        elif measure == 'ccc':
            result = Reporter.ccc(targets.numpy(), predictions.numpy())
        else:
            self.util.error(f'unknown measure: {measure}')
        return result, targets, predictions

    def store(self):
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{self.run}_{self.epoch:03d}.model'
        torch.save(self.model.state_dict(), dir+name)
        
    def load(self, run, epoch):
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{run}_{epoch:03d}.model'
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        layers = ast.literal_eval(glob_conf.config['MODEL']['layers'])
        self.model = self.MLP(self.feats_train.df.shape[1], layers, 1).to(self.device)
        self.model.load_state_dict(torch.load(dir+name))
        self.model.eval()