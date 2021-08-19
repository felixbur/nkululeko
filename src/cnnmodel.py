# cnnmodel.py

from model import Model
import torch
import audpann
from sklearn.metrics import mean_squared_error
import glob_conf

class CNN_model(Model):
    """A CNN model"""


    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        Model.__init__(self, df_train, df_test, feats_train, feats_test)
        self.util.debug(f'initializing model')
        self.device = config['MODEL']['device']
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
        self.util.debug(f'training model')
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.train()
        for features, labels in self.feats_train:
            logits = self.model(features.to(self.device).float()).squeeze(1)
            loss = criterion(logits, labels.float().to(self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self):
        """Predict the whole eval feature set"""
        mse, truth, pred = self.evaluate_model(False)
        return pred

    def predict_train(self):
        """Predict the whole eval feature set"""
        mse, truth, pred = self.evaluate_model(True)
        return pred
    

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