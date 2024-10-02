"""Code based on @jwagner."""

import ast
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
    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes."""
        super().__init__(df_train, df_test, feats_train, feats_test)
        super().set_model_type("finetuned")
        self.df_test, self.df_train, self.feats_test, self.feats_train = (
            df_test,
            df_train,
            feats_test,
            feats_train,
        )
        self.name = "finetuned_wav2vec2"
        self.target = glob_conf.config["DATA"]["target"]
        self.labels = glob_conf.labels
        self.class_num = len(self.labels)
        device = self.util.config_val("MODEL", "device", False)
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device != "cpu":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        self.util.debug(f"running on device {self.device}")
        self.is_classifier = self.util.exp_is_classification()
        if self.is_classifier:
            self.measure = "uar"
        else:
            self.measure = self.util.config_val("MODEL", "measure", "ccc")
        self.util.debug(f"evaluation metrics: {self.measure}")
        self.batch_size = int(self.util.config_val("MODEL", "batch_size", "8"))
        self.util.debug(f"batch size: {self.batch_size}")
        self.learning_rate = float(
            self.util.config_val("MODEL", "learning_rate", "0.0001")
        )
        self.max_duration = float(self.util.config_val("MODEL", "max_duration", "8.0"))
        self.df_train, self.df_test = df_train, df_test
        self.epoch_num = int(self.util.config_val("EXP", "epochs", 1))
        self.util.debug(f"num of epochs: {self.epoch_num}")
        drop = self.util.config_val("MODEL", "drop", False)
        self.drop = 0.1
        if drop:
            self.drop = float(drop)
        self.util.debug(f"init: training with dropout: {self.drop}")
        self.push = eval(self.util.config_val("MODEL", "push_to_hub", "False"))
        self.balancing = self.util.config_val("MODEL", "balancing", False)
        self._init_model()

    def _init_model(self):
        model_path = "facebook/wav2vec2-large-robust-ft-swbd-300h"
        pretrained_model = self.util.config_val("MODEL", "pretrained_model", model_path)
        self.num_layers = None
        self.sampling_rate = 16000
        self.max_duration_sec = self.max_duration
        self.accumulation_steps = 4

        # print finetuning information via debug
        self.util.debug(f"Finetuning from model: {pretrained_model}")

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
            #     ds = datasets.Dataset.from_pandas(df)
            #     dataset[split] = ds

            # self.dataset = datasets.DatasetDict(dataset)
            if split == "train" and self.balancing:
                if self.balancing == "ros":
                    from imblearn.over_sampling import RandomOverSampler

                    sampler = RandomOverSampler(random_state=42)
                elif self.balancing == "smote":
                    from imblearn.over_sampling import SMOTE

                    sampler = SMOTE(random_state=42)
                elif self.balancing == "adasyn":
                    from imblearn.over_sampling import ADASYN

                    sampler = ADASYN(random_state=42)
                else:
                    self.util.error(f"Unknown balancing algorithm: {self.balancing}")

                X_resampled, y_resampled = sampler.fit_resample(
                    df[["start", "end"]], df["targets"]
                )
                df = pd.DataFrame(
                    {
                        "start": X_resampled["start"],
                        "end": X_resampled["end"],
                        "targets": y_resampled,
                    }
                )

                # print the before and after class distribution
                self.util.debug(
                    f"balanced with: {self.balancing}, new size: {len(df)}, was {len(data_sources[split])}"
                )

            ds = datasets.Dataset.from_pandas(df)
            dataset[split] = ds

        self.dataset = datasets.DatasetDict(dataset)

        # load pre-trained model
        if self.is_classifier:
            self.util.debug("Task is classification.")
            le = glob_conf.label_encoder
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            target_mapping = {k: int(v) for k, v in mapping.items()}
            target_mapping_reverse = {
                value: key for key, value in target_mapping.items()
            }
            self.config = transformers.AutoConfig.from_pretrained(
                pretrained_model,
                num_labels=len(target_mapping),
                label2id=target_mapping,
                id2label=target_mapping_reverse,
                finetuning_task=target_name,
            )
        else:
            self.util.debug("Task is regression.")
            self.config = transformers.AutoConfig.from_pretrained(
                pretrained_model,
                num_labels=1,
                finetuning_task=target_name,
            )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        self.config.final_dropout = self.drop
        setattr(self.config, "sampling_rate", self.sampling_rate)
        setattr(self.config, "data", self.util.get_data_name())
        setattr(self.config, "is_classifier", self.is_classifier)

        vocab_dict = {}
        with open("vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        tokenizer = transformers.Wav2Vec2CTCTokenizer("./vocab.json")
        tokenizer.save_pretrained(".")

        # uoload tokenizer to hub if true
        if self.push:
            tokenizer.push_to_hub(self.util.get_name())

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
            pretrained_model,
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

        batch["labels"] = torch.Tensor(targets)

        return batch

    def compute_metrics(self, p: transformers.EvalPrediction):
        metrics = {
            "UAR": audmetric.unweighted_average_recall,
            "ACC": audmetric.accuracy,
        }
        metrics_reg = {
            "PCC": audmetric.pearson_cc,
            "CCC": audmetric.concordance_cc,
            "MSE": audmetric.mean_squared_error,
            "MAE": audmetric.mean_absolute_error,
        }

        # truth = p.label_ids[:, 0].astype(int)
        truth = p.label_ids
        preds = p.predictions
        preds = np.argmax(preds, axis=1)
        scores = {}
        if self.is_classifier:
            for name, metric in metrics.items():
                scores[f"{name}"] = metric(truth, preds)
        else:
            for name, metric in metrics_reg.items():
                scores[f"{name}"] = metric(truth, preds)

        return scores

    def train(self):
        """Train the model."""
        model_root = self.util.get_path("model_dir")
        self.log_root = os.path.join(self.util.get_exp_dir(), "log")
        audeer.mkdir(self.log_root)
        self.torch_root = audeer.path(model_root, "torch")
        conf_file = os.path.join(self.torch_root, "config.json")
        if os.path.isfile(conf_file):
            self.util.debug(f"reusing finetuned model: {conf_file}")
            self.load(self.run, self.epoch_num)
            return
        targets = pd.DataFrame(self.dataset["train"]["targets"])

        if self.is_classifier:
            criterion = self.util.config_val("MODEL", "loss", "cross")
            if criterion == "cross":
                if self.util.config_val("MODEL", "class_weight", False):
                    counts = targets[0].value_counts().sort_index()
                    train_weights = 1 / counts
                    train_weights /= train_weights.sum()
                    self.util.debug(f"train weights: {train_weights}")
                    criterion = torch.nn.CrossEntropyLoss(
                        weight=torch.Tensor(train_weights).to("cuda"),
                    )
                else:
                    criterion = torch.nn.CrossEntropyLoss()
            else:
                self.util.error(f"criterion {criterion} not supported for classifier")
        else:
            criterion = self.util.config_val("MODEL", "loss", "1-ccc")
            if criterion == "1-ccc":
                criterion = ConcordanceCorCoeff()
            elif criterion == "mse":
                criterion = torch.nn.MSELoss()
            elif criterion == "mae":
                criterion = torch.nn.L1Loss()
            else:
                self.util.error(f"criterion {criterion} not supported for regressor")

        # set push_to_hub value, default false
        # push = eval(self.util.config_val("MODEL", "push_to_hub", "False"))

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

        metrics_for_best_model = self.measure.upper()
        if metrics_for_best_model == "UAR":
            greater_is_better = True
        elif metrics_for_best_model == "CCC":
            greater_is_better = True
        elif metrics_for_best_model == "MSE":
            greater_is_better = False
        elif metrics_for_best_model == "MAE":
            greater_is_better = False
        else:
            self.util.error(f"unknown metric/measure: {metrics_for_best_model}")

        training_args = transformers.TrainingArguments(
            output_dir=model_root,
            logging_dir=self.log_root,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            evaluation_strategy="steps",
            num_train_epochs=self.epoch_num,
            fp16=self.device != "cpu",
            use_cpu=self.device == "cpu",
            save_steps=num_steps,
            eval_steps=num_steps,
            logging_steps=num_steps,
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            save_total_limit=2,
            metric_for_best_model=metrics_for_best_model,
            greater_is_better=greater_is_better,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            report_to="none",
            push_to_hub=self.push,
            hub_model_id=f"{self.util.get_name()}",
            overwrite_output_dir=True,
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
        log_file = os.path.join(
            self.log_root,
            "log.txt",
        )
        with open(log_file, "w") as text_file:
            print(trainer.state.log_history, file=text_file)
        self.util.debug(f"saved best model to {self.torch_root}")
        self.load(self.run, self.epoch)

    def get_predictions(self):
        results = [[]].pop(0)
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
            prediction = self.model.predict(signal)
            results.append(prediction)
            # results.append(predictions.argmax())
        predictions = np.asarray(results)
        if self.util.exp_is_classification():
            # make a dataframe for the class probabilities
            proba_d = {}
            for c in range(self.class_num):
                proba_d[c] = []
            # get the class probabilities
            # predictions = self.clf.predict_proba(self.feats_test.to_numpy())
            # pred = self.clf.predict(features)
            for i in range(self.class_num):
                proba_d[i] = list(predictions.T[i])
            probas = pd.DataFrame(proba_d)
            probas = probas.set_index(self.df_test.index)
            predictions = probas.idxmax(axis=1).values
        else:
            predictions = predictions.flatten()
            probas = None
        return predictions, probas

    def predict(self):
        """Predict the whole eval feature set"""
        predictions, probas = self.get_predictions()
        report = Reporter(
            self.df_test[self.target].to_numpy().astype(float),
            predictions,
            self.run,
            self.epoch_num,
            probas=probas,
        )
        self._plot_epoch_progression(report)
        return report

    def _plot_epoch_progression(self, report):
        log_file = os.path.join(
            self.log_root,
            "log.txt",
        )
        with open(log_file, "r") as file:
            data = file.read()
        data = data.strip().replace("nan", "0")
        list = ast.literal_eval(data)
        epochs, vals, loss = [], [], []
        for index, tp in enumerate(list):
            try:
                epochs.append(tp["epoch"])
                measure = self.measure.upper()
                vals.append(tp[f"eval_{measure}"])
                loss.append(tp["eval_loss"])
            except KeyError:
                del epochs[-1]
                # print(f'no value at {index}')
        df = pd.DataFrame({"results": vals, "losses": loss}, index=epochs)
        report.plot_epoch_progression_finetuned(df)

    def predict_sample(self, signal):
        """Predict one sample"""
        prediction = {}
        if self.is_classifier:
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
    logits: torch.FloatTensor = None
    hidden_states: typing.Tuple[torch.FloatTensor] = None
    cnn_features: torch.FloatTensor = None


@dataclasses.dataclass
class ModelOutputReg(transformers.file_utils.ModelOutput):
    logits: torch.FloatTensor
    hidden_states: typing.Tuple[torch.FloatTensor] = None
    attentions: typing.Tuple[torch.FloatTensor] = None
    logits_framewise: torch.FloatTensor = None
    hidden_states_framewise: torch.FloatTensor = None
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
        if not hasattr(config, "add_adapter"):
            setattr(config, "add_adapter", False)

        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.head = ModelHead(config)
        self.is_classifier = config.is_classifier
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

            epsilon = 1e-6  # to avoid division by zero and numerical instability
            outputs = outputs / (torch.reshape(attention_sum, (-1, 1)) + epsilon)

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
        logits = self.head(hidden_states)
        if not self.training:
            logits = torch.softmax(logits, dim=1)

        if return_hidden:
            # make time last axis
            cnn_features = torch.transpose(cnn_features, 1, 2)
            if self.is_classifier:
                return ModelOutput(
                    logits=logits,
                    hidden_states=hidden_states,
                    cnn_features=cnn_features,
                )
            else:
                return ModelOutputReg(
                    logits=logits,
                    hidden_states=hidden_states,
                    cnn_features=cnn_features,
                )
        else:
            if self.is_classifier:
                return ModelOutput(
                    logits=logits,
                )
            else:
                return ModelOutputReg(
                    logits=logits,
                )

    def predict(self, signal):
        result = self(torch.from_numpy(signal))
        result = result[0].detach().numpy()[0]
        return result


class ConcordanceCorCoeff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        ground_truth = ground_truth.float()
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
