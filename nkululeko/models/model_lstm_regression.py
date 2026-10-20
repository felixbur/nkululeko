"""LSTM_Reg_model: [MODEL] type = lstm_reg.

LSTM regressor over frame sequences from the feature pipeline
(FEATS.level = frame), one value per segment. Loss (MODEL.loss) and
evaluation measure (MODEL.measure) work as for mlp_reg; the network, data
loading and training are shared with lstm, see rnn_core.py.
"""

import numpy as np
import torch
from audmetric import (
    concordance_cc,
    mean_absolute_error,
    mean_squared_error,
    pearson_cc,
)

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
from nkululeko.losses.loss_pcc import PearsonCorCoeff
from nkululeko.models.rnn_core import RNNModel
from nkululeko.reporting.reporter import Reporter


class LSTM_Reg_model(RNNModel):
    """LSTM over frame sequences, for a continuous target."""

    is_classifier = False
    type_name = "lstm_reg"
    label_dtype = np.float32

    def _setup_task(self):
        self.out_dim = 1
        criterion = self.util.config_val("MODEL", "loss", "mse")
        if criterion == "mse":
            self.criterion = torch.nn.MSELoss()
        elif criterion == "mae":
            self.criterion = torch.nn.L1Loss()
        elif criterion == "1-ccc":
            self.criterion = ConcordanceCorCoeff()
        elif criterion == "1-pcc":
            self.criterion = PearsonCorCoeff()
        else:
            self.util.error(f"unknown loss function: {criterion}")
        self.util.debug(f"training model with {criterion} loss function")

    def _task_loss(self, out, labels):
        return self.criterion(out.reshape(-1), labels)

    def evaluate(self, loader):
        out, targets, loss_eval = self._run(loader)
        predictions = out.reshape(-1)
        truths, preds = targets.numpy(), predictions.numpy()
        measure = self.util.config_val("MODEL", "measure", "mse")
        if measure == "mse":
            result = mean_squared_error(truths, preds)
        elif measure == "mae":
            result = mean_absolute_error(truths, preds)
        elif measure == "ccc":
            result = concordance_cc(truths, preds)
        elif measure == "pcc":
            result = pearson_cc(truths, preds)
        else:
            self.util.error(f"unknown measure: {measure}")
        return result, targets, predictions, loss_eval

    def predict(self):
        _, truths, predictions, loss_eval = self.evaluate(self.testloader)
        self.loss_eval = loss_eval
        result, _, _, _ = self.evaluate(self._eval_trainloader())
        report = Reporter(
            truths.numpy(),
            predictions.numpy(),
            self.run,
            self.epoch,
            context=self.context,
        )
        try:
            report.result.loss = self.loss
        except AttributeError:  # if the model was loaded from disk the loss is unknown
            pass
        report.result.loss_eval = self.loss_eval
        report.result.train = result
        return report

    def get_predictions(self):
        _, _, predictions, _ = self.evaluate(self.testloader)
        return predictions.numpy(), None
