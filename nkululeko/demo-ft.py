import argparse
import configparser
import os

from sklearn import pipeline
from transformers import pipelines

from nkululeko.utils.util import get_exp_dir

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="A file that should be processed (16kHz mono wav)")
# read config from ini file
parser.add_argument("--config", default="exp.ini", help="The base configuration")


args = parser.parse_args()
file = args.file
config = configparser.ConfigParser()

# get exp dir from config [EXP][root][name] + models + run_0 + torch
config.read(args.config)
exp_dir = get_exp_dir("model_path")

# exp_dir = get_exp_dir("model_path")
model_path = os.path.join(exp_dir, "model")
pipe = pipelines("audio-classification", model=model_path)


print(pipeline(file))
