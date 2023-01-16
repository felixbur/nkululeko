# cnnmodel.py

from sklearn.utils import resample
from nkululeko.model import Model
import torch
from sklearn.metrics import mean_squared_error
import nkululeko.glob_conf as glob_conf
from nkululeko.reporter import Reporter
import numpy as np
from loss_ccc import ConcordanceCorCoeff

class CNN_model(Model):
    """A CNN (convolutional neural net) model"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        Model.__init__(self, df_train, df_test, feats_train, feats_test)
        self.util.debug(f'initializing model')
        self.device = glob_conf.config['MODEL']['device']
        pretrained_state = self.util.config_val('MODEL', 'pre_train', 'gender_state.pth.tar')
        state = torch.load(pretrained_state)
        state.pop('out.gender.weight')
        state.pop('out.gender.bias')
        state['fc1.weight'] = state.pop('fc1.gender.weight')
        state['fc1.bias'] = state.pop('fc1.gender.bias')
        model = audpann.Cnn10(sampling_rate=16000, output_dim=1)
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device)       
        self.loss_func = self.util.config_val('MODEL', 'loss', 'mse')
        if self.loss_func == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_func == '1-ccc':
            self.criterion = ConcordanceCorCoeff()
        else:
            self.util.error(f'unknown loss function: {self.loss_func}')
        self.util.debug(f'training model with {self.loss_func} loss function')
        self.learning_rate = float(self.util.config_val('MODEL', 'learning_rate', 0.0001))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """Train the model one epoch"""
        losses = []
        # first check if the model already has been trained
        # if os.path.isfile(self.store_path):
        #     self.load(self.run, self.epoch)
        #     self.util.debug(f'reusing model: {self.store_path}')
        #     return
            
        self.model.train()

        for features, labels in self.feats_train:
            logits = self.model(features.to(self.device).float()).squeeze(1)
            loss = self.criterion(logits, labels.float().to(self.device))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss = (np.asarray(losses)).mean()

    def predict(self):
        """Predict the whole eval feature set"""
        # evaluate on dev set
        _, truths, predictions = self.evaluate_model(False)
        # evaluate on train set, if there is one
        result = 0
        if self.feats_train != None:
            result, _, _ = self.evaluate_model(True)
        report = Reporter(truths.numpy(), predictions.numpy(), self.run, self.epoch)
        try:
            report.result.loss = self.loss
        except AttributeError: # if the model was loaded from disk the loss is unknown
            pass 
        report.result.train = result
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

        measure = self.util.config_val('MODEL', 'measure', 'mse')
        if measure == 'mse':
            result = mean_squared_error(targets.numpy(), logits.numpy())
        elif measure == 'ccc':
            result = Reporter.ccc(targets.numpy(), logits.numpy())
        else:
            self.util.error(f'unknown measure: {measure}')
        return result, targets, logits

    def store(self):
        torch.save(self.model.state_dict(), self.store_path)
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        # self.model.to(self.device)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name(only_train=True)}_{run}_{epoch:03d}.model'
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.store_path = dir+name
        self.model = audpann.Cnn10(sampling_rate=16000, output_dim=1)
        state_dict = torch.load(dir+name, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)   