"""Code based on @jwagner."""

import ast
import dataclasses
import inspect
import json
import os
import pickle
import re
import typing

import audeer
import audiofile
import audmetric
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import HubertModel, WavLMModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
from nkululeko.losses.loss_pcc import PearsonCorCoeff
from nkululeko.models.finetune_config import FinetuneConfig
from nkululeko.models.model import Model as BaseModel
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.pickle_integrity import verify_checksum


class TunedModel(BaseModel):
    def __init__(self, df_train, df_test, feats_train, feats_test, context=None):
        """Constructor taking the configuration and all dataframes."""
        super().__init__(df_train, df_test, feats_train, feats_test, context=context)
        super().set_model_type("finetuned")
        self.df_test, self.df_train, self.feats_test, self.feats_train = (
            df_test,
            df_train,
            feats_test,
            feats_train,
        )
        self.name = "finetuned_wav2vec2"
        self.target = self.context.config["DATA"]["target"]
        self.labels = self.context.labels
        self.class_num = len(self.labels)
        self.is_classifier = self.util.exp_is_classification()
        self.cfg = FinetuneConfig.from_util(self.util, self.is_classifier)

        self.device, cuda_visible_devices = self._resolve_device(self.cfg.device)
        if cuda_visible_devices is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.util.debug(f"running on device {self.device}")

        self.measure = self.cfg.measure
        self.util.debug(f"evaluation metrics: {self.measure}")
        self.batch_size = self.cfg.batch_size
        self.util.debug(f"batch size: {self.batch_size}")
        self.learning_rate = self.cfg.learning_rate
        self.max_duration = self.cfg.max_duration
        self.df_train, self.df_test = df_train, df_test
        self.epoch_num = int(self.util.config_val("EXP", "epochs", 1))
        self.util.debug(f"num of epochs: {self.epoch_num}")
        self.drop = self.cfg.drop
        self.util.debug(f"init: training with dropout: {self.drop}")
        self.push = self.cfg.push_to_hub
        self.balancing = self.cfg.balancing
        # Set unconditionally (not just inside train()) so a freshly
        # constructed instance can load() an already-trained checkpoint and
        # predict() without training first - e.g. runmanager.get_best_model()
        # and eval_specific_model() under [EXP] traindevtest=True build new
        # TunedModel instances purely to reload the best-dev-epoch checkpoint
        # and evaluate it on the real test set. Both paths are deterministic
        # (derived from the experiment dir), so setting them here is safe
        # even before any training has actually happened.
        self.torch_root = audeer.path(self.util.get_path("model_dir"), "torch")
        self.log_root = os.path.join(self.util.get_exp_dir(), "log")
        audeer.mkdir(self.log_root)
        self._init_model()

    def _resolve_device(self, device):
        """Resolve a [FINETUNE] device value into (torch_device, cuda_visible_devices).

        CUDA_VISIBLE_DEVICES only ever accepts physical GPU indices (e.g.
        "0" or "0,1"), never a torch device string like "cuda" or "cuda:0" -
        setting it to those hides every GPU instead of selecting one. This
        extracts the index (if any) for CUDA_VISIBLE_DEVICES separately from
        the torch-facing device string used everywhere else in this class,
        which after masking to a single GPU is always plain "cpu" or "cuda".
        """
        if device == "cpu":
            return "cpu", None
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match:
            return "cuda", match.group(1)
        if re.fullmatch(r"\d+(,\d+)*", device):
            return "cuda", device
        if device == "cuda":
            return "cuda", None
        self.util.warn(f"unrecognized device '{device}'; falling back to 'cuda'")
        return "cuda", None

    @staticmethod
    def _eval_strategy_key():
        """Name of the TrainingArguments kwarg that selects the eval strategy.

        transformers renamed evaluation_strategy -> eval_strategy at some
        point (and later versions may drop the old name entirely) - detect
        which one this installed version actually accepts instead of
        hardcoding either.
        """
        params = inspect.signature(transformers.TrainingArguments.__init__).parameters
        return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    def _build_callbacks(self, evals_per_epoch):
        """Trainer callbacks: TensorBoard, plus early stopping if configured.

        MODEL.patience is shared with every other model type (SVM/MLP/CNN
        etc. via modelrunner.do_epochs()'s own patience loop) and is in
        epochs there. Finetuning instead evaluates `evals_per_epoch` times
        per epoch (see train()), so patience is scaled to the matching
        number of evaluation calls for transformers' EarlyStoppingCallback -
        without this, load_best_model_at_end only picks the best checkpoint
        after training runs for the full configured epoch count; it never
        actually stops training early the way patience does for other models.
        """
        callbacks = [transformers.integrations.TensorBoardCallback()]
        patience = self.util.config_val("MODEL", "patience", False)
        if patience:
            early_stopping_patience = int(patience) * evals_per_epoch
            callbacks.append(
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            )
            self.util.debug(
                f"finetune: early stopping after {patience} epochs "
                f"({early_stopping_patience} evaluation calls) without dev improvement"
            )
        return callbacks

    @staticmethod
    def _cast_targets(targets, is_classifier):
        """Cast the HF Trainer's label tensor to the dtype its loss needs.

        Classification losses (CrossEntropyLoss) need integer class indices.
        Regression losses need float targets - casting to long
        unconditionally (as this used to do) truncates every continuous
        target to an integer (e.g. a grbas_score of 0.67 becomes 0),
        silently destroying nearly all signal for every regression loss.
        """
        return targets.type(torch.long) if is_classifier else targets.float()

    @staticmethod
    def _match_loss_dtype(targets, logits, is_classifier):
        """Align regression targets with the model's actual output dtype.

        Under fp16 training, logits come out of the model as Half - torch
        losses (MSELoss, L1Loss) require both operands to share a dtype
        exactly, unlike CrossEntropyLoss, which accepts Long targets against
        Half/Float logits without complaint. Classification targets are left
        untouched; only regression targets need to track whatever precision
        this forward pass actually used.
        """
        return targets if is_classifier else targets.to(logits.dtype)

    def _init_model(self):
        pretrained_model = self.cfg.pretrained_model
        self.num_layers = self.cfg.num_layers
        self.sampling_rate = 16000
        self.max_duration_sec = self.max_duration
        self.accumulation_steps = 4

        # print finetuning information via debug
        self.util.debug(f"Finetuning from model: {pretrained_model}")

        if any(
            emotion_model in pretrained_model
            for emotion_model in ["emotion2vec", "iic/emotion2vec"]
        ):
            self._init_emotion2vec_model(pretrained_model)
            return

        self._init_huggingface_model(pretrained_model)

    def _init_huggingface_model(self, pretrained_model):
        """Initialize HuggingFace transformer model for finetuning."""
        # create dataset
        dataset = {}
        target_name = self.context.target
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

            if split == "train" and self.balancing:
                df = self._apply_balancing(df, data_sources[split])

            ds = datasets.Dataset.from_pandas(df)
            dataset[split] = ds

        self.dataset = datasets.DatasetDict(dataset)

        # load pre-trained model
        if self.is_classifier:
            self.util.debug("Task is classification.")
            le = self.context.label_encoder
            if le is None:
                self.util.error(
                    "Label encoder is not available. Make sure to set up data loading properly."
                )
                raise ValueError(
                    "Label encoder is missing. Initialization cannot proceed. Ensure data loading is correctly configured."
                )
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
        original_num_layers = self.config.num_hidden_layers
        self._validate_layer_config(original_num_layers)
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
            self.util.debug(f"truncating model to {self.num_layers} encoder layers")
        self.config.final_dropout = self.drop
        setattr(self.config, "sampling_rate", self.sampling_rate)
        setattr(self.config, "data", self.util.get_data_name())
        setattr(self.config, "is_classifier", self.is_classifier)
        setattr(self.config, "head_layers", self.cfg.head_layers)
        setattr(self.config, "head_activation", self.cfg.head_activation)
        setattr(self.config, "pooling", self.cfg.pooling)
        # "eager" attention is universally supported; SDPA (the newer
        # default) isn't implemented for every architecture in every
        # transformers version (e.g. WavLM lacks it as of transformers 5) -
        # avoid depending on per-architecture/per-version SDPA support.
        self.config._attn_implementation = "eager"

        vocab_dict = {}
        with open("vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)
        tokenizer = transformers.Wav2Vec2CTCTokenizer("./vocab.json")
        tokenizer.save_pretrained(".")

        # uoload tokenizer to hub if true
        if self.push:
            tokenizer.push_to_hub(self.util.get_name())

        feature_extractor = self._build_feature_extractor(pretrained_model)
        self.processor = transformers.Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        assert self.processor.feature_extractor.sampling_rate == self.sampling_rate  # type: ignore

        self.model = Model.from_pretrained(  # type: ignore
            pretrained_model,
            config=self.config,
        )
        self.model.freeze_feature_extractor()  # type: ignore
        self._freeze_encoder_layers(self.model, self.cfg.freeze_layers)
        self.model.train()  # type: ignore
        self.model_initialized = True

    def _build_feature_extractor(self, pretrained_model):
        """Build the Wav2Vec2FeatureExtractor matching `pretrained_model`.

        Loads the checkpoint's own preprocessor config (e.g. `do_normalize`,
        which genuinely differs between checkpoints - some wav2vec2/HuBERT
        models expect it True, some WavLM models expect it False) instead
        of hardcoding wav2vec2-robust's settings for every backbone. Falls
        back to those hardcoded settings only if the checkpoint has none
        (e.g. a local/custom model with no preprocessor_config.json).

        return_attention_mask is always forced True regardless of source:
        pooling() (below) falls back to naively meaning over ALL frames,
        including padding, when no attention mask is present - correct only
        for batch_size==1, so nkululeko's own design requires this.
        """
        try:
            feature_extractor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
                pretrained_model
            )
        except Exception as e:
            self.util.warn(
                f"could not load feature extractor config for {pretrained_model} "
                f"({e}); falling back to default wav2vec2-style settings"
            )
            feature_extractor = transformers.Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
            )
        feature_extractor.return_attention_mask = True
        return feature_extractor

    def _validate_layer_config(self, original_num_layers):
        """Validate num_layers/freeze_layers against the pretrained model's depth.

        Enforces 0 <= freeze_layers < effective_num_layers <= original_num_layers,
        so num_layers never exceeds what the pretrained checkpoint provides, and
        freeze_layers never freezes the entire (possibly truncated) backbone,
        leaving at least one encoder layer trainable.
        """
        if self.num_layers is not None and not (0 < self.num_layers <= original_num_layers):
            self.util.error(
                f"num_layers={self.num_layers} must be between 1 and the "
                f"pretrained model's {original_num_layers} encoder layers"
            )
        effective_num_layers = (
            self.num_layers if self.num_layers is not None else original_num_layers
        )
        if not (0 <= self.cfg.freeze_layers < effective_num_layers):
            self.util.error(
                f"freeze_layers={self.cfg.freeze_layers} must be less than the "
                f"resulting model's {effective_num_layers} encoder layers "
                "(at least one layer must stay trainable)"
            )

    def _freeze_encoder_layers(self, model, freeze_layers):
        """Freeze the first `freeze_layers` transformer encoder layers.

        Leaves the remaining encoder layers and the head trainable. A
        `freeze_layers` of 0 (the default) freezes nothing here, matching
        full finetuning; freeze_feature_extractor() above always freezes
        the CNN feature extractor regardless of this setting.
        """
        if freeze_layers < 0:
            self.util.warn(
                f"freeze_layers={freeze_layers} is negative; freezing nothing "
                "(Python slicing would otherwise freeze from the end of the list)"
            )
            return
        if not freeze_layers:
            return
        layers = model.wav2vec2.encoder.layers
        if freeze_layers > len(layers):
            self.util.warn(
                f"freeze_layers={freeze_layers} exceeds the model's "
                f"{len(layers)} encoder layers; freezing all of them"
            )
        for layer in layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        self.util.debug(f"froze the first {min(freeze_layers, len(layers))} encoder layers")

    def _init_emotion2vec_model(self, pretrained_model):
        """Initialize emotion2vec model for finetuning."""
        try:
            from funasr import AutoModel
        except ImportError:
            self.util.error(
                "FunASR is required for emotion2vec finetuning. "
                "Please install with: pip install funasr"
            )
            return

        if self.cfg.freeze_layers:
            self.util.warn(
                "freeze_layers is not supported for emotion2vec models "
                "(the funasr backbone isn't exposed as freezable layers); ignoring it"
            )
        if self.num_layers is not None:
            self.util.warn(
                "num_layers is not supported for emotion2vec models "
                "(the funasr backbone isn't built from a HF encoder config); ignoring it"
            )

        model_mapping = {
            "emotion2vec": "emotion2vec/emotion2vec_base",
            "emotion2vec-base": "emotion2vec/emotion2vec_base",
            "emotion2vec-seed": "emotion2vec/emotion2vec_plus_seed",
            "emotion2vec-large": "emotion2vec/emotion2vec_plus_large",
        }

        if pretrained_model in model_mapping:
            model_path = model_mapping[pretrained_model]
        else:
            model_path = pretrained_model

        self._create_emotion2vec_dataset()

        self.emotion2vec_backbone = AutoModel(
            model=model_path,
            hub="hf",  # Use HuggingFace Hub instead of ModelScope
        )

        if self.is_classifier:
            le = self.context.label_encoder
            if le is None:
                self.util.error("Label encoder not available for classification")
                return
            num_labels = len(le.classes_)
            label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
            self.config = EmotionVecConfig(
                num_labels=num_labels,
                label2id=label_mapping,
                id2label={v: k for k, v in label_mapping.items()},
                is_classifier=True,
                sampling_rate=self.sampling_rate,
                final_dropout=self.drop,
                model_name=pretrained_model,
            )
        else:
            self.config = EmotionVecConfig(
                num_labels=1,
                is_classifier=False,
                sampling_rate=self.sampling_rate,
                final_dropout=self.drop,
                model_name=pretrained_model,
            )

        self.model = Emotion2vecModel(self.emotion2vec_backbone, self.config)
        self.model.train()
        self.model_initialized = True

        self.processor = None

    def _create_emotion2vec_dataset(self):
        """Create dataset for emotion2vec training."""
        dataset = {}
        target_name = self.context.target
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

            if split == "train" and self.balancing:
                df = self._apply_balancing(df, data_sources[split])

            ds = datasets.Dataset.from_pandas(df)
            dataset[split] = ds

        self.dataset = datasets.DatasetDict(dataset)

    def _apply_balancing(self, df, original_df):
        """Apply data balancing to training dataset."""
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
            return df

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

        self.util.debug(
            f"balanced with: {self.balancing}, new size: {len(df)}, was {len(original_df)}"
        )
        return df

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
        for file, start, end in zip(files, starts, ends):
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

        if hasattr(self, "emotion2vec_backbone"):
            max_length = max(len(s) for s in signals)
            padded_signals = []
            for s in signals:
                if len(s) < max_length:
                    padded = np.pad(s, (0, max_length - len(s)), mode="constant")
                else:
                    padded = s[:max_length]
                padded_signals.append(padded)

            batch = {
                "input_values": torch.stack(
                    [torch.tensor(s, dtype=torch.float32) for s in padded_signals]
                ),
                "labels": torch.tensor(
                    targets,
                    dtype=torch.float32 if not self.is_classifier else torch.long,
                ),
            }
        else:
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

        if isinstance(preds, tuple):
            if len(preds) > 0:
                preds = preds[0]  # Extract logits from tuple
            else:
                raise ValueError(f"Empty predictions tuple received: {preds}")

        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        elif hasattr(preds, "detach"):
            preds = preds.detach().numpy()

        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        else:
            preds = preds.flatten()
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
        conf_file = os.path.join(self.torch_root, "config.json")
        if os.path.isfile(conf_file):
            self.util.debug(f"reusing finetuned model: {conf_file}")
            self.load(self.run, self.epoch_num)
            return
        targets = pd.DataFrame(self.dataset["train"]["targets"])

        if self.is_classifier:
            criterion = self.cfg.loss
            if criterion == "cross":
                label_smoothing = self._get_label_smoothing()
                if self.cfg.class_weight:
                    counts = targets[0].value_counts().sort_index()
                    train_weights = 1 / counts
                    train_weights /= train_weights.sum()
                    self.util.debug(f"train weights: {train_weights}")
                    criterion = torch.nn.CrossEntropyLoss(
                        weight=torch.Tensor(train_weights).to(self.device),
                        label_smoothing=label_smoothing,
                    )
                else:
                    criterion = torch.nn.CrossEntropyLoss(
                        label_smoothing=label_smoothing,
                    )
            else:
                self.util.error(f"criterion {criterion} not supported for classifier")
        else:
            criterion = self.cfg.loss
            if criterion == "1-ccc":
                criterion = ConcordanceCorCoeff()
            elif criterion == "1-pcc":
                criterion = PearsonCorCoeff()
            elif criterion == "mse":
                criterion = torch.nn.MSELoss()
            elif criterion == "mae":
                criterion = torch.nn.L1Loss()
            else:
                self.util.error(f"criterion {criterion} not supported for regressor")

        # Captured by name (not `self.is_classifier`) since compute_loss's
        # own `self` below is the inner Trainer instance, shadowing this
        # method's TunedModel `self`.
        is_classifier = self.is_classifier

        class Trainer(transformers.Trainer):
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                targets = inputs.pop("labels").squeeze()
                targets = TunedModel._cast_targets(targets, is_classifier)

                outputs = model(**inputs)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits.squeeze()
                else:
                    logits = outputs[0].squeeze()

                targets = TunedModel._match_loss_dtype(targets, logits, is_classifier)

                loss = criterion(logits, targets)

                return (loss, outputs) if return_outputs else loss

        # eval/save happen every `num_steps` (~1/evals_per_epoch of an
        # epoch's worth of steps) - see early stopping below, which scales
        # MODEL.patience (in epochs, matching every other model type) to
        # this many evaluation calls.
        evals_per_epoch = 5
        num_steps = (
            len(self.dataset["train"])
            // (self.batch_size * self.accumulation_steps)
            // evals_per_epoch
        )
        num_steps = max(1, num_steps)

        metrics_for_best_model = self.measure.upper()
        if metrics_for_best_model == "UAR":
            greater_is_better = True
        elif metrics_for_best_model == "CCC":
            greater_is_better = True
        elif metrics_for_best_model == "PCC":
            greater_is_better = True
        elif metrics_for_best_model == "MSE":
            greater_is_better = False
        elif metrics_for_best_model == "MAE":
            greater_is_better = False
        else:
            self.util.error(f"unknown metric/measure: {metrics_for_best_model}")

        training_kwargs = dict(
            output_dir=model_root,
            logging_dir=self.log_root,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            num_train_epochs=self.epoch_num,
            fp16=self.device != "cpu",
            use_cpu=self.device == "cpu",
            save_steps=num_steps,
            eval_steps=num_steps,
            logging_steps=num_steps,
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            warmup_ratio=self.cfg.warmup_ratio,
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
        training_kwargs[self._eval_strategy_key()] = "steps"
        training_args = transformers.TrainingArguments(**training_kwargs)

        trainer_kwargs = {
            "model": self.model,
            "data_collator": self.data_collator,
            "args": training_args,
            "compute_metrics": self.compute_metrics,
            "train_dataset": self.dataset["train"],
            "eval_dataset": self.dataset["dev"],
            "callbacks": self._build_callbacks(evals_per_epoch),
        }

        if self.processor is not None:
            trainer_kwargs["tokenizer"] = self.processor.feature_extractor

        trainer = Trainer(**trainer_kwargs)

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

    def _normalize_signal(self, signal):
        """Normalize raw audio the same way every training/eval batch does.

        data_collator() (used for both training and HF's own internal eval
        loop) always runs raw audio through self.processor before it ever
        reaches the model - Wav2Vec2FeatureExtractor's do_normalize=True
        zero-mean/unit-variance normalization. get_predictions() and
        predict_sample() instead fed audiofile.read()'s raw signal straight
        into Model.predict(), skipping that normalization entirely: the
        model was finetuned to expect normalized input, so every dev/test
        report (and every predict_sample()/demo call) was silently scored
        against wrong-scale input, corrupting the reported metrics
        independently of - and in addition to - the missing eval() mode bug.
        """
        if self.processor is None:
            return signal
        squeezed = np.asarray(signal).squeeze()
        processed = self.processor(
            squeezed, sampling_rate=self.sampling_rate, padding=False
        )
        # Model.predict() does `self(torch.from_numpy(signal))` with no
        # unsqueeze of its own and indexes the result as a batch
        # (`result[0].detach().numpy()[0]`), so the returned array must keep
        # the (1, seq_len) batch dimension the raw signal already had (e.g.
        # from audiofile.read(..., always_2d=True)) - not the bare (seq_len,)
        # array self.processor's own output shape would otherwise give.
        return np.asarray(processed["input_values"][0], dtype=np.float32)[None, :]

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
            signal = self._normalize_signal(signal)
            prediction = self.model.predict(signal)  # type: ignore
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
            context=self.context,
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
        signal = self._normalize_signal(signal)
        if self.is_classifier:
            # get the class probabilities
            predictions = self.model.predict(signal)  # type: ignore
            # pred = self.clf.predict(features)
            for i in range(len(self.labels)):
                cat = self.labels[i]
                prediction[cat] = predictions[i]
        else:
            predictions = self.model.predict(signal)  # type: ignore
            prediction = predictions
        return prediction

    def store(self):
        self.util.debug("stored: ")

    def load(self, run, epoch):
        self.set_id(run, epoch)
        if hasattr(self, "emotion2vec_backbone"):
            model_path = os.path.join(self.torch_root, "pytorch_model.bin")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
        else:
            self.model = Model.from_pretrained(
                self.torch_root,
                config=self.config,
            )
            # A freshly constructed/loaded nn.Module defaults to train mode,
            # so without this, dropout stays active during every subsequent
            # predict() call - corrupting exactly the dev/test evaluation
            # this reload exists for (confirmed: the reloaded weights here
            # match the true best-dev-CCC checkpoint bit-for-bit, but the
            # reported dev/test metrics were far below that checkpoint's
            # live-observed eval score until this was added).
            self.model.eval()
        # print(f"loaded model type {type(self.model)}")

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        verify_checksum(path)
        with open(path, "rb") as handle:
            self.clf = pickle.load(handle)


@dataclasses.dataclass
class ModelOutput:
    logits: typing.Optional[torch.Tensor] = None
    hidden_states: typing.Optional[torch.Tensor] = None
    cnn_features: typing.Optional[torch.Tensor] = None

    def __getitem__(self, index):
        """Make ModelOutput subscriptable for HuggingFace compatibility."""
        if isinstance(index, slice):
            items = [self.logits, self.hidden_states, self.cnn_features]
            result = items[index]
            filtered_result = [item for item in result if item is not None]

            if not filtered_result and self.logits is not None:
                return (self.logits,)

            return tuple(filtered_result)
        elif index == 0:
            return self.logits
        elif index == 1:
            return self.hidden_states
        elif index == 2:
            return self.cnn_features
        else:
            raise IndexError(f"Index {index} out of range for ModelOutput")

    def __len__(self):
        """Return the number of available outputs."""
        return 3


@dataclasses.dataclass
class ModelOutputReg:
    logits: torch.Tensor
    hidden_states: typing.Optional[torch.Tensor] = None
    attentions: typing.Optional[torch.Tensor] = None
    logits_framewise: typing.Optional[torch.Tensor] = None
    hidden_states_framewise: typing.Optional[torch.Tensor] = None
    cnn_features: typing.Optional[torch.Tensor] = None

    def __getitem__(self, index):
        """Make ModelOutputReg subscriptable for HuggingFace compatibility."""
        if isinstance(index, slice):
            items = [
                self.logits,
                self.hidden_states,
                self.attentions,
                self.logits_framewise,
                self.hidden_states_framewise,
                self.cnn_features,
            ]
            result = items[index]
            filtered_result = [item for item in result if item is not None]

            if not filtered_result and self.logits is not None:
                return (self.logits,)

            return tuple(filtered_result)
        elif index == 0:
            return self.logits
        elif index == 1:
            return self.hidden_states
        elif index == 2:
            return self.attentions
        elif index == 3:
            return self.logits_framewise
        elif index == 4:
            return self.hidden_states_framewise
        elif index == 5:
            return self.cnn_features
        else:
            raise IndexError(f"Index {index} out of range for ModelOutputReg")

    def __len__(self):
        """Return the number of available outputs."""
        return 6


HEAD_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "leaky_relu": torch.nn.LeakyReLU,
}


class ModelHead(torch.nn.Module):
    """Classification/regression head on top of the pretrained backbone.

    Configurable via config.head_layers/config.head_activation (set from
    [FINETUNE] head_layers/head_activation in _init_huggingface_model) so its
    capacity and activation can be matched to mlp/mlp_reg's [MODEL]
    layers/activation for a fair embeddings-vs-finetuning comparison.
    Defaults (a single hidden layer sized to the backbone's own hidden_size,
    tanh) preserve the original hardcoded architecture.
    """

    def __init__(self, config, input_dim=None):
        super().__init__()

        head_layers = getattr(config, "head_layers", None) or [config.hidden_size]
        activation_name = getattr(config, "head_activation", "tanh")
        if activation_name not in HEAD_ACTIVATIONS:
            raise ValueError(
                f"unknown head_activation: {activation_name}; "
                f"expected one of {sorted(HEAD_ACTIVATIONS)}"
            )
        activation_cls = HEAD_ACTIVATIONS[activation_name]

        dims = [input_dim or config.hidden_size] + list(head_layers)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Dropout(config.final_dropout))
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(activation_cls())
        layers.append(torch.nn.Dropout(config.final_dropout))
        layers.append(torch.nn.Linear(dims[-1], config.num_labels))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features, **kwargs):
        return self.net(features)


BACKBONE_MODEL_CLASSES = {
    "wav2vec2": Wav2Vec2Model,
    "wavlm": WavLMModel,
    "hubert": HubertModel,
}


class Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        if not hasattr(config, "add_adapter"):
            setattr(config, "add_adapter", False)

        super().__init__(config)

        backbone_cls = BACKBONE_MODEL_CLASSES.get(
            getattr(config, "model_type", None), Wav2Vec2Model
        )
        self.wav2vec2 = backbone_cls(config)
        # "meanvar" pooling concatenates mean and variance along the
        # feature dim, doubling what the head actually receives - the head
        # can't just assume config.hidden_size the way "mean" pooling allows.
        self.pooling_mode = getattr(config, "pooling", "mean")
        pooled_dim = (
            config.hidden_size * 2 if self.pooling_mode == "meanvar" else config.hidden_size
        )
        self.head = ModelHead(config, input_dim=pooled_dim)
        self.is_classifier = config.is_classifier
        # post_init() (not the lower-level init_weights() it calls) is the
        # documented hook for subclasses - required as of transformers 5.x,
        # which moved bookkeeping (e.g. tied-weight-key collection) into it.
        self.post_init()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def pooling(
        self,
        hidden_states,
        attention_mask,
    ):
        if attention_mask is None:  # For evaluation with batch_size==1
            mean = torch.mean(hidden_states, dim=1)
            if self.pooling_mode == "meanvar":
                var = torch.var(hidden_states, dim=1, unbiased=False)
                return torch.cat([mean, var], dim=-1)
            return mean
        else:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            mask = torch.reshape(
                attention_mask,
                (-1, attention_mask.shape[-1], 1),
            )
            masked_hidden_states = hidden_states * mask
            attention_sum = torch.sum(attention_mask, dim=1)

            epsilon = 1e-6  # to avoid division by zero and numerical instability
            denom = torch.reshape(attention_sum, (-1, 1)) + epsilon
            mean = torch.sum(masked_hidden_states, dim=1) / denom

            if self.pooling_mode == "meanvar":
                mean_sq = torch.sum((hidden_states**2) * mask, dim=1) / denom
                # Clamp against tiny negative values from floating-point
                # cancellation in E[x^2] - E[x]^2 (true variance is >= 0).
                var = torch.clamp(mean_sq - mean**2, min=0.0)
                return torch.cat([mean, var], dim=-1)

            return mean

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
        if not self.training and self.is_classifier:
            # Regression logits have num_labels==1, so softmax over that
            # single-element dimension always returns exactly 1.0 regardless
            # of the input - silently turning every eval/test regression
            # prediction into the same constant, no matter what the model
            # actually learned.
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


class EmotionVecConfig:
    """Configuration class for emotion2vec models."""

    def __init__(
        self,
        num_labels,
        is_classifier=True,
        sampling_rate=16000,
        final_dropout=0.1,
        model_name=None,
        **kwargs,
    ):
        self.num_labels = num_labels
        self.is_classifier = is_classifier
        self.sampling_rate = sampling_rate
        self.final_dropout = final_dropout
        self.model_name = model_name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json_string(self):
        """Convert config to JSON string for HuggingFace compatibility."""
        import json

        config_dict = {
            "num_labels": self.num_labels,
            "is_classifier": self.is_classifier,
            "sampling_rate": self.sampling_rate,
            "final_dropout": self.final_dropout,
        }
        for key, value in self.__dict__.items():
            if key not in config_dict:
                config_dict[key] = value
        return json.dumps(config_dict, indent=2)


class Emotion2vecModel(torch.nn.Module):
    """Wrapper class for emotion2vec finetuning."""

    def __init__(self, emotion2vec_backbone, config):
        super().__init__()
        self.emotion2vec_backbone = emotion2vec_backbone
        self.config = config
        self.is_classifier = config.is_classifier

        # Determine embedding dimension based on model variant (hardcoded)
        embedding_dim = self._get_embedding_dim_by_model()
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(config.final_dropout),
            torch.nn.Linear(embedding_dim, config.num_labels),
        )

    def _get_embedding_dim_by_model(self):
        """Get embedding dimension based on model variant."""
        model_name = getattr(self.config, "model_name", "")

        # Large models have 1024 dimensions
        if "large" in model_name.lower():
            return 1024
        # Base, seed, and other models have 768 dimensions
        else:
            return 768

    def forward(self, input_values, labels=None, **kwargs):
        embeddings = self._extract_embeddings(input_values)

        logits = self.head(embeddings)

        if not self.training and self.is_classifier:
            logits = torch.softmax(logits, dim=1)

        if self.is_classifier:
            return ModelOutput(logits=logits)
        else:
            return ModelOutputReg(logits=logits)

    def _extract_embeddings(self, input_values):
        batch_embeddings = []
        device = next(self.parameters()).device  # Get the device of the model
        for audio_tensor in input_values:
            embedding = self._process_single_audio(audio_tensor)
            # Ensure embedding is on the same device as the model
            embedding = embedding.to(device)
            batch_embeddings.append(embedding)
        return torch.stack(batch_embeddings)

    def _process_single_audio(self, audio_tensor):
        import tempfile
        import soundfile as sf

        signal_np = audio_tensor.squeeze().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, signal_np, self.config.sampling_rate)

            try:
                res = self.emotion2vec_backbone.generate(
                    tmp_file.name, granularity="utterance", extract_embedding=True
                )

                if isinstance(res, list) and len(res) > 0:
                    embeddings = res[0].get("feats", None)
                    if embeddings is not None:
                        if isinstance(embeddings, list):
                            embeddings = np.array(embeddings)
                        return torch.tensor(embeddings.flatten(), dtype=torch.float32)

                # Fallback based on model type
                model_name = getattr(self.config, "model_name", "")
                if "large" in model_name.lower():
                    return torch.zeros(1024, dtype=torch.float32)
                else:
                    return torch.zeros(768, dtype=torch.float32)
            finally:
                os.unlink(tmp_file.name)

    def predict(self, signal):
        """Predict method for compatibility with nkululeko prediction pipeline."""
        if isinstance(signal, np.ndarray):
            signal_tensor = torch.from_numpy(signal).unsqueeze(0)
        else:
            signal_tensor = signal.unsqueeze(0) if signal.dim() == 1 else signal

        with torch.no_grad():
            result = self(signal_tensor)

        if self.is_classifier:
            logits = result.logits
        else:
            logits = result.logits

        return logits.detach().cpu().numpy()[0]
