"""
Code based on @jwagner.
"""

import dataclasses
import json
import os
import pickle
import typing

import audeer
import audiofile
import audmetric
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model as BaseModel
from nkululeko.reporting.reporter import Reporter


class TunedModel(BaseModel):

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes."""
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("finetuned")
        self.name = "finetuned_wav2vec2"
        self.target = glob_conf.config["DATA"]["target"]
        labels = glob_conf.labels
        self.class_num = len(labels)
        # device = self.util.config_val("MODEL", "device", "cpu")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", "8"))
        if self.device != "cpu":
            self.util.debug(f"running on device {self.device}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # self.device
        self.df_train, self.df_test = df_train, df_test
        self.epoch_num = int(self.util.config_val("EXP", "epochs", 1))

        self._init_model()

    def _init_model(self):
        model_path = "facebook/wav2vec2-large-robust-ft-swbd-300h"
        self.num_layers = None
        self.sampling_rate = 16000
        self.max_duration_sec = 8.0
        self.accumulation_steps = 4
        # create dataset

        dataset = {}
        target_name = glob_conf.target
        data_sources = {
            "train": pd.DataFrame(self.df_train[target_name]),
            "dev": pd.DataFrame(self.df_test[target_name]),
        }

        for split in ["train", "dev"]:
            df = data_sources[split]
            y = df[target_name].astype("float")
            y.name = "targets"
            df = y.reset_index()
            df.start = df.start.dt.total_seconds()
            df.end = df.end.dt.total_seconds()
            ds = datasets.Dataset.from_pandas(df)
            dataset[split] = ds

        self.dataset = datasets.DatasetDict(dataset)

        # load pre-trained model
        le = glob_conf.label_encoder
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        target_mapping = {k: int(v) for k, v in mapping.items()}
        target_mapping_reverse = {
            value: key for key,
            value in target_mapping.items()}

        self.config = transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(target_mapping),
            label2id=target_mapping,
            id2label=target_mapping_reverse,
            finetuning_task=target_name,
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        setattr(self.config, "sampling_rate", self.sampling_rate)
        setattr(self.config, "data", self.util.get_data_name())

        vocab_dict = {}
        with open("vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        tokenizer = transformers.Wav2Vec2CTCTokenizer("./vocab.json")
        tokenizer.save_pretrained(".")

        feature_extractor = transformers.Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = transformers.Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        assert self.processor.feature_extractor.sampling_rate == self.sampling_rate

        self.model = Model.from_pretrained(
            model_path,
            config=self.config,
        )
        self.model.freeze_feature_extractor()
        self.model.train()
        self.model_initialized = True

    def set_model_type(self, type):
        self.model_type = type

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df

    def reset_test(self, df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test

    def set_id(self, run, epoch):
        self.run = run
        self.epoch = epoch
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        self.store_path = dir + name

    def data_collator(self, data):
        files = [d["file"] for d in data]
        starts = [d["start"] for d in data]
        ends = [d["end"] for d in data]
        targets = [d["targets"] for d in data]

        signals = []
        for file, start, end in zip(
            files,
            starts,
            ends,
        ):
            offset = start
            duration = end - offset
            if self.max_duration_sec is not None:
                duration = min(duration, self.max_duration_sec)
            signal, _ = audiofile.read(
                file,
                offset=offset,
                duration=duration,
            )
            signals.append(signal.squeeze())

        input_values = self.processor(
            signals,
            sampling_rate=self.sampling_rate,
            padding=True,
        )
        batch = self.processor.pad(
            input_values,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(targets)

        return batch

    def compute_metrics(self, p: transformers.EvalPrediction):

        metrics = {
            "UAR": audmetric.unweighted_average_recall,
            "ACC": audmetric.accuracy,
        }

        # truth = p.label_ids[:, 0].astype(int)
        truth = p.label_ids
        preds = p.predictions
        preds = np.argmax(preds, axis=1)
        scores = {}
        for name, metric in metrics.items():
            scores[f"{name}"] = metric(truth, preds)
        return scores

    def train(self):
        """Train the model."""
        model_root = self.util.get_path("model_dir")
        log_root = os.path.join(self.util.get_exp_dir(), "log")
        audeer.mkdir(log_root)
        self.torch_root = audeer.path(model_root, "torch")
        conf_file = os.path.join(self.torch_root, "config.json")
        if os.path.isfile(conf_file):
            self.util.debug(f"reusing finetuned model: {conf_file}")
            self.load(self.run, self.epoch_num)
            return
        targets = pd.DataFrame(self.dataset["train"]["targets"])
        counts = targets[0].value_counts().sort_index()
        train_weights = 1 / counts
        train_weights /= train_weights.sum()
        self.util.debug("train weights: {train_weights}")
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor(train_weights).to(self.device),
        )
        # criterion = torch.nn.CrossEntropyLoss()

        class Trainer(transformers.Trainer):
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
            ):
                targets = inputs.pop("labels").squeeze()
                targets = targets.type(torch.long)

                outputs = model(**inputs)
                logits = outputs[0].squeeze()

                loss = criterion(logits, targets)

                return (loss, outputs) if return_outputs else loss

        num_steps = (
            len(self.dataset["train"])
            // (self.batch_size * self.accumulation_steps)
            // 5
        )
        num_steps = max(1, num_steps)
        # print(num_steps)

        training_args = transformers.TrainingArguments(
            output_dir=model_root,
            logging_dir=log_root,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            evaluation_strategy="steps",
            num_train_epochs=self.epoch_num,
            fp16=self.device == "cuda",
            save_steps=num_steps,
            eval_steps=num_steps,
            logging_steps=num_steps,
            learning_rate=1e-4,
            save_total_limit=2,
            metric_for_best_model="UAR",
            greater_is_better=True,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["dev"],
            tokenizer=self.processor.feature_extractor,
            callbacks=[transformers.integrations.TensorBoardCallback()],
        )
        trainer.train()
        trainer.save_model(self.torch_root)
        self.load(self.run, self.epoch)

    def get_predictions(self):
        results = []
        for (file, start, end), _ in audeer.progress_bar(
            self.df_test.iterrows(),
            total=len(self.df_test),
            desc=f"Predicting {len(self.df_test)} audiofiles",
        ):
            if end == pd.NaT:
                signal, sr = audiofile.read(file, offset=start)
            else:
                signal, sr = audiofile.read(
                    file, duration=end - start, offset=start, always_2d=True
                )
            assert sr == self.sampling_rate
            predictions = self.model.predict(signal)
            results.append(predictions.argmax())
        return results

    def predict(self):
        """Predict the whole eval feature set"""
        predictions = self.get_predictions()
        report = Reporter(
            self.df_test[self.target].to_numpy().astype(float),
            predictions,
            self.run,
            self.epoch_num,
        )
        return report

    def predict_sample(self, signal):
        """Predict one sample"""
        prediction = {}
        if self.util.exp_is_classification():
            # get the class probabilities
            predictions = self.model.predict(signal)
            # pred = self.clf.predict(features)
            for i in range(len(self.labels)):
                cat = self.labels[i]
                prediction[cat] = predictions[i]
        else:
            predictions = self.model.predict(signal)
            prediction = predictions
        return prediction

    def store(self):
        self.util.debug("stored: ")

    def load(self, run, epoch):
        self.set_id(run, epoch)
        self.model = Model.from_pretrained(
            self.torch_root,
            config=self.config,
        )
        # print(f"loaded model type {type(self.model)}")

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, "rb") as handle:
            self.clf = pickle.load(handle)


@dataclasses.dataclass
class ModelOutput(transformers.file_utils.ModelOutput):

    logits_cat: torch.FloatTensor = None
    hidden_states: typing.Tuple[torch.FloatTensor] = None
    cnn_features: torch.FloatTensor = None


class ModelHead(torch.nn.Module):

    def __init__(self, config):

        super().__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

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
        self.cat = ModelHead(config)
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

    def predict(self, signal):
        result = self(torch.from_numpy(signal))
        result = result[0].detach().numpy()[0]
        return result


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
        # raises: onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented:
        # [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an
        # implementation for the node ReduceProd_3:ReduceProd(11)

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
