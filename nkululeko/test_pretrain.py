# test_pretrain.py
import argparse
import configparser
import json
import os.path

import audeer
import audiofile
import audmetric
import datasets
import numpy as np
import pandas as pd
import torch
import transformers

import nkululeko.experiment as exp
import nkululeko.glob_conf as glob_conf
import nkululeko.models.finetune_model as fm
from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


def doit(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "test_pretrain"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    model_root = util.get_path("model_dir")
    log_root = audeer.mkdir("log")
    torch_root = audeer.path(model_root, "torch")

    metrics_gender = {
        "UAR": audmetric.unweighted_average_recall,
        "ACC": audmetric.accuracy,
    }

    sampling_rate = 16000
    max_duration_sec = 8.0

    model_path = "facebook/wav2vec2-large-robust-ft-swbd-300h"
    num_layers = None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    batch_size = 16
    accumulation_steps = 4
    # create dataset

    dataset = {}
    target_name = glob_conf.target
    data_sources = {
        "train": pd.DataFrame(expr.df_train[target_name]),
        "dev": pd.DataFrame(expr.df_test[target_name]),
    }

    for split in ["train", "dev"]:
        df = data_sources[split]
        df[target_name] = df[target_name].astype("float")

        y = pd.Series(
            data=df.itertuples(index=False, name=None),
            index=df.index,
            dtype=object,
            name="labels",
        )

        y.name = "targets"
        df = y.reset_index()
        df.start = df.start.dt.total_seconds()
        df.end = df.end.dt.total_seconds()

        print(f"{split}: {len(df)}")

        ds = datasets.Dataset.from_pandas(df)
        dataset[split] = ds

    dataset = datasets.DatasetDict(dataset)

    # load pre-trained model
    le = glob_conf.label_encoder
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    target_mapping = {k: int(v) for k, v in mapping.items()}
    target_mapping_reverse = {value: key for key, value in target_mapping.items()}

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        num_labels=len(target_mapping),
        label2id=target_mapping,
        id2label=target_mapping_reverse,
        finetuning_task=target_name,
    )
    if num_layers is not None:
        config.num_hidden_layers = num_layers
    setattr(config, "sampling_rate", sampling_rate)
    setattr(config, "data", util.get_data_name())

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
    processor = transformers.Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    assert processor.feature_extractor.sampling_rate == sampling_rate

    model = fm.Model.from_pretrained(
        model_path,
        config=config,
    )
    model.freeze_feature_extractor()
    model.train()

    # training

    def data_collator(data):

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
            if max_duration_sec is not None:
                duration = min(duration, max_duration_sec)
            signal, _ = audiofile.read(
                file,
                offset=offset,
                duration=duration,
            )
            signals.append(signal.squeeze())

        input_values = processor(
            signals,
            sampling_rate=sampling_rate,
            padding=True,
        )
        batch = processor.pad(
            input_values,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(targets)

        return batch

    def compute_metrics(p: transformers.EvalPrediction):

        truth_gender = p.label_ids[:, 0].astype(int)
        preds = p.predictions
        preds_gender = np.argmax(preds, axis=1)

        scores = {}

        for name, metric in metrics_gender.items():
            scores[f"gender-{name}"] = metric(truth_gender, preds_gender)

        scores["combined"] = scores["gender-UAR"]

        return scores

    targets = pd.DataFrame(dataset["train"]["targets"])
    counts = targets[0].value_counts().sort_index()
    train_weights = 1 / counts
    train_weights /= train_weights.sum()

    print(train_weights)

    criterion_gender = torch.nn.CrossEntropyLoss(
        weight=torch.Tensor(train_weights).to("cuda"),
    )

    class Trainer(transformers.Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
        ):

            targets = inputs.pop("labels").squeeze()
            targets_gender = targets.type(torch.long)

            outputs = model(**inputs)
            logits_gender = outputs[0].squeeze()

            loss_gender = criterion_gender(logits_gender, targets_gender)

            loss = loss_gender

            return (loss, outputs) if return_outputs else loss

    num_steps = len(dataset["train"]) // (batch_size * accumulation_steps) // 5
    num_steps = max(1, num_steps)
    print(num_steps)

    training_args = transformers.TrainingArguments(
        output_dir=model_root,
        logging_dir=log_root,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=5.0,
        fp16=True,
        save_steps=num_steps,
        eval_steps=num_steps,
        logging_steps=num_steps,
        learning_rate=1e-4,
        save_total_limit=2,
        metric_for_best_model="combined",
        greater_is_better=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=processor.feature_extractor,
        callbacks=[transformers.integrations.TensorBoardCallback()],
    )
    if False:
        trainer.train()
        trainer.save_model(torch_root)

    modelnew = fm.Model.from_pretrained(
        torch_root,
        config=config,
    )
    print(f"loaded new model type{type(modelnew)}")
    import audiofile

    signal, _ = audiofile.read("./test.wav", always_2d=True)
    result = modelnew.predict(signal)
    print(result)

    print("DONE")


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"
    doit(config_file)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
