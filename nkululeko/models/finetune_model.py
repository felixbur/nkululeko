import dataclasses
import typing

import torch
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)


class ConcordanceCorCoeff(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):

        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (
            self.sqrt(self.sum(v_pred**2)) * self.sqrt(self.sum(v_gt**2))
        )
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator

        return 1 - ccc


@dataclasses.dataclass
class ModelOutput(transformers.file_utils.ModelOutput):

    logits_cat: torch.FloatTensor = None
    hidden_states: typing.Tuple[torch.FloatTensor] = None
    cnn_features: torch.FloatTensor = None


class ModelHead(torch.nn.Module):

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class Model(Wav2Vec2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.cat = ModelHead(config, 2)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def pooling(
        self,
        hidden_states,
        attention_mask,
    ):

        if attention_mask is None:  # For evaluation with batch_size==1
            outputs = torch.mean(hidden_states, dim=1)
        else:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            hidden_states = hidden_states * torch.reshape(
                attention_mask,
                (-1, attention_mask.shape[-1], 1),
            )
            outputs = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            outputs = outputs / torch.reshape(attention_sum, (-1, 1))

        return outputs

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        return_hidden=False,
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
        )

        cnn_features = outputs.extract_features
        hidden_states_framewise = outputs.last_hidden_state
        hidden_states = self.pooling(
            hidden_states_framewise,
            attention_mask,
        )
        logits_cat = self.cat(hidden_states)

        if not self.training:
            logits_cat = torch.softmax(logits_cat, dim=1)

        if return_hidden:

            # make time last axis
            cnn_features = torch.transpose(cnn_features, 1, 2)

            return ModelOutput(
                logits_cat=logits_cat,
                hidden_states=hidden_states,
                cnn_features=cnn_features,
            )

        else:

            return ModelOutput(
                logits_cat=logits_cat,
            )


class ModelWithPreProcessing(Model):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_values,
    ):
        # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm():
        # normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)

        mean = input_values.mean()

        # var = input_values.var()
        # raises: onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for the node ReduceProd_3:ReduceProd(11)

        var = torch.square(input_values - mean).mean()
        input_values = (input_values - mean) / torch.sqrt(var + 1e-7)

        output = super().forward(
            input_values,
            return_hidden=True,
        )

        return (
            output.hidden_states,
            output.logits_cat,
            output.cnn_features,
        )
