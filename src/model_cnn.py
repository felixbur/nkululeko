# cnnmodel.py

from model import Model
import torch
import audpann
from sklearn.metrics import mean_squared_error
import glob_conf
from reporter import Reporter
import numpy as np
import ast

class CNN_model(Model):
    """A CNN (convolutional neural net) model"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        Model.__init__(self, df_train, df_test, feats_train, feats_test)
        self.util.debug(f'initializing model')
        self.device = glob_conf.config['MODEL']['device']
        store = self.util.get_path('store')
        state = torch.load(store+'gender_state.pth.tar')
        state.pop('out.gender.weight')
        state.pop('out.gender.bias')
        state['fc1.weight'] = state.pop('fc1.gender.weight')
        state['fc1.bias'] = state.pop('fc1.gender.bias')
        model = audpann.Cnn10(sampling_rate=16000, output_dim=1)
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device)       

    def train(self):
        """Train the model one epoch"""
        losses = []
        self.util.debug(f'training model')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.train()
        for features, labels in self.feats_train:
            logits = self.model(features.to(self.device).float()).squeeze(1)
            loss = criterion(logits, labels.float().to(self.device))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def predict(self):
        """Predict the whole eval feature set"""
        # evaluate on dev set
        _, truths, predictions = self.evaluate_model(False)
        # evaluate on train set
        mse, _, _ = self.evaluate_model(True)
        report = Reporter(truths, predictions)
        try:
            report.result.loss = self.loss
        except AttributeError: # if the model was loaded from disk the loss is unknown
            pass 
        report.result.train = mse
        return report
    

    def evaluate_model(self, on_train=False):
        if on_train:
            loader = self.feats_train
        else:
            loader = self.feats_test
        logits = torch.zeros(len(loader.dataset))
        targets = torch.zeros(len(loader.dataset))
        self.model.eval()
        with torch.no_grad():
            for index, (features, labels) in enumerate(loader):
                start_index = index * loader.batch_size
                end_index = (index + 1) * loader.batch_size
                if end_index > len(loader.dataset):
                    end_index = len(loader.dataset)
                logits[start_index:end_index] =  self.model(features.to(self.device).float()).squeeze(1)
                targets[start_index:end_index] = labels
        truth = targets.numpy()
        pred = logits.numpy()
        # truth = self.scaler.inverse_transform(targets.numpy().reshape(-1, 1)).flatten()
        # pred = self.scaler.inverse_transform(logits.numpy().reshape(-1, 1)).flatten()
        mse = mean_squared_error(truth, pred)
        return mse, truth, pred

    def store(self):
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{self.run}_{self.epoch:03d}.model'
        torch.save(self.model.state_dict(), dir+name)
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        # self.model.to(self.device)

    def load(self, run, epoch):
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{run}_{epoch:03d}.model'
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.model = audpann.Cnn10(sampling_rate=16000, output_dim=1)
        state_dict = torch.load(dir+name, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)   