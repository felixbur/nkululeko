# feats_trill.py
from numpy.core.numeric import tensordot
from featureset import Featureset
import pandas as pd
from util import Util 
import glob_conf
import audiofile as af
import os
import tensorflow as tf
# Import TF 2.X and make sure we're running eager.
assert tf.executing_eagerly()
import tensorflow_hub as hub

class TRILLset(Featureset):
    """A feature extractor for the Google TRILL embeddings"""
    """https://ai.googleblog.com/2020/06/improving-speech-representations-and.html"""

    def __init__(self, name, data_df):
        self.name = name
        self.data_df = data_df
        self.util = Util()
#        self.module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')
        model_path = self.util.config_val('FEATS', 'model', \
            'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')
        self.module = hub.load(model_path)

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        try:
            extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        except KeyError:
            extract = False
        if extract or not os.path.isfile(storage):
            self.util.debug('extracting TRILL embeddings, this might take a while...')
            emb_series = pd.Series(index = self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, file in enumerate(self.data_df.index.get_level_values(0)):
                emb = self.getEmbeddings(file)
                emb_series[idx] = emb
                if idx%10==0:
                    self.util.debug(f'TRILL: {idx} of {length} done')
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index) 
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted TRILL embeddings')
            self.df = pd.read_pickle(storage)

    def embed_wav(self, wav): 
        if len(wav.shape) > 1:
            wav = tf.reduce_mean(wav, axis=0)
            
        emb_dict = self.module(samples=wav, sample_rate=tf.constant(16000))
        return emb_dict['embedding']

    def getEmbeddings(self, file):
        wav = af.read(file)[0]
        wav = tf.convert_to_tensor(wav)
        emb_short = self.embed_wav(wav)
        # you get one embedding per frame, we use the mean for all the frames
        emb_short = emb_short.numpy().mean(axis=0)
        return emb_short
