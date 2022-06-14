# feats_transformer.py

from util import Util
from featureset import Featureset
import os
import pandas as pd
import os
import glob_conf 
import numpy as np
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import torch
import audeer
import audiofile
import datasets 

class TransformerFeats(Featureset):
    """Class to extract prepare sample for Convnet input needed by huggingface transformers """

    def __init__(self, name, data_df):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.sampling_rate = 16000
        store = self.util.get_path('store')
        storage = f'{store}{self.name}'
        self.model_root = audeer.mkdir(storage)
        self.model_path = self.util.config_val('FEATS', 'wav2vec.model', 'wav2vec2-large-robust-ft-swbd-300h')
        self.num_layers = 6

        self.model_initialized = False



    def init_model(self):
        # load model
        self.util.debug('loading wav2vec model...')
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path).to(self.device)
        print(f'intialized wav22vec model on {self.device}')
        self.model.eval()
        self.model_initialized = True


    def extract(self):
        """Extract the features or load them from disk if present."""
        dataset = datasets.load_dataset(
            'csv',
            data_files=data_files,
            delimiter=',',
            cache_dir=data_root,
        )
    