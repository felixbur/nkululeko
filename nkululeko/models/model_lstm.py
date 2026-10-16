"""LSTMModel: [MODEL] type = lstm.

LSTM classifier over frame sequences from the feature pipeline
(FEATS.level = frame), one label per segment (GH #256, #471). The
network, data loading and training are shared with lstm_reg, see
rnn_core.py.
"""

import pandas as pd
import torch
from sklearn.metrics import recall_score

from nkululeko.models.rnn_core import RNNModel
from nkululeko.reporting.reporter import Reporter


class LSTMModel(RNNModel):
    """LSTM over frame sequences, for classification."""

    is_classifier = True
    type_name = "lstm"

    def _setup_task(self):
        self.class_num = len(self.context.labels)
        self.out_dim = self.class_num
        self._setup_criterion()

    def evaluate(self, loader):
        logits, targets, loss_eval = self._run(loader)
        predictions = logits.argmax(dim=1)
        uar = (
            recall_score(targets.numpy(), predictions.numpy(), average="macro")
            if len(targets)
            else 0.0
        )
        return uar, targets, predictions, logits, loss_eval

    def get_probas(self, logits):
        probs = torch.softmax(logits, dim=1).numpy()
        # Columns follow the network's outputs, i.e. every encoded class
        # 0..class_num-1, even if the test split lacks some of them.
        proba_d = {c: list(probs.T[c]) for c in range(self.class_num)}
        return pd.DataFrame(proba_d).set_index(self.df_test.index)

    def predict(self):
        _, truths, predictions, logits, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        uar, _, _, _, _ = self.evaluate(self.trainloader)
        report = Reporter(
            truths.numpy().astype(float),
            predictions.numpy(),
            self.run,
            self.epoch,
            probas=self.get_probas(logits),
            context=self.context,
        )
        try:
            report.result.loss = self.loss
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        report.result.loss_eval = self.loss_eval
        report.result.train = uar
        return report

    def get_predictions(self):
        _, _, predictions, logits, _ = self.evaluate(self.testloader)
        return predictions.numpy(), self.get_probas(logits)
