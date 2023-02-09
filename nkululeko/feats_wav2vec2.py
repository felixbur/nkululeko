# feats_wav2vec2.py

from nkululeko.util import Util
from nkululeko.featureset import Featureset
import os
import pandas as pd
import os
import nkululeko.glob_conf as glob_conf
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import torch

import audiofile

class Wav2vec2(Featureset):
    """Class to extract wav2vec2 embeddings (https://huggingface.co/facebook/wav2vec2-large-robust-ft-swbd-300h)"""

    def __init__(self, name, data_df):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.model_initialized = False



    def init_model(self):
        # load model
        self.util.debug('loading wav2vec model...')
        model_path = self.util.config_val('FEATS', 'wav2vec.model', 'wav2vec2-large-robust-ft-swbd-300h')
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path).to(self.device)
        print(f'intialized vec model on {self.device}')
        self.model.eval()
        self.model_initialized = True


    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug('extracting wav2vec2 embeddings, this might take a while...')
            emb_series = pd.Series(index = self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(self.data_df.index.to_list()):
                signal, sampling_rate = audiofile.read(file, offset=start.total_seconds(), duration=(end-start).total_seconds(), always_2d=True)
                #signal, sampling_rate = audiofile.read(audio_path, always_2d=True)
                emb = self.get_embeddings(signal, sampling_rate)
                emb_series[idx] = emb
                if idx%10==0:
                    self.util.debug(f'Wav2vec2: {idx} of {length} done')
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index) 
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted wav2vec2 embeddings')
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(f'got nan: {self.df.shape} {self.df.isnull().sum().sum()}')


    def get_embeddings(self, signal, sampling_rate):
        r"""Extract embeddings from raw audio signal."""
        with torch.no_grad():
            # run through processor to normalize signal
            # always returns a batch, so we just get the first entry
            # then we put it on the device
            y = self.processor(signal, sampling_rate=sampling_rate)
            y = y['input_values'][0]
            y = torch.from_numpy(y).to(self.device)
        
            # run through model
            # first entry contains hidden state
            y = self.model(y)[0]
        
            # pool result and convert to numpy
            y = torch.mean(y, dim=1)
            y = y.detach().cpu().numpy()
    
        return y.flatten()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr)
        return feats