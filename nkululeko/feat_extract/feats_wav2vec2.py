# feats_wav2vec2.py

import os

import nkululeko.glob_conf as glob_conf
import pandas as pd
import torch
import torchaudio
from nkululeko.feat_extract.featureset import Featureset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# import audiofile
# import torchaudio


class Wav2vec2(Featureset):
    """Class to extract wav2vec2 embeddings """

    def __init__(self, name, data_df, feat_type):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.util.config_val('MODEL', 'device', cuda)
        self.model_initialized = False
        self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug('loading wav2vec model...')
        model_path = self.util.config_val(
            'FEATS', 'wav2vec.model', f'facebook/{self.feat_type}')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path).to(self.device)
        print(f'intialized Wav2vec model on {self.device}')
        self.model.eval()
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        extract = self.util.config_val(
            'FEATS', 'needs_feature_extraction', False)
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                'extracting wav2vec2 embeddings, this might take a while...')
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(self.data_df.index.to_list()):
                signal, sampling_rate = torchaudio.load(file,
                    frame_offset=start.total_seconds()*16000,
                    num_frames=(end - start).total_seconds()*16000)
                assert sampling_rate == 16000
                # if sampling_rate != 16000:
                #     if idx == 0:
                #         self.util.debug(
                #             f"resampling {self.feat_type} to 16kHz. Will slow down the process."
                #         )
                #     resampler = torchaudio.transforms.Resample(
                #         sampling_rate, 16000)
                #     signal = resampler(signal)
                #     sampling_rate = 16000
                # signal, sampling_rate = audiofile.read(file, offset=start.total_seconds(), duration=(end-start).total_seconds(), always_2d=True)
                # signal, sampling_rate = audiofile.read(audio_path, always_2d=True)
                emb = self.get_embeddings(signal, sampling_rate, file)
                emb_series[idx] = emb
                if idx % 10 == 0:
                    self.util.debug(f'Wav2vec2: {idx} of {length} done')
            self.df = pd.DataFrame(
                emb_series.values.tolist(), index=self.data_df.index)
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
                self.util.error(
                    f'got nan: {self.df.shape} {self.df.isnull().sum().sum()}')

    def get_embeddings(self, signal, sampling_rate, file):
        r"""Extract embeddings from raw audio signal."""
        try:
            with torch.no_grad():
                # run through processor to normalize signal
                # always returns a batch, so we just get the first entry
                # then we put it on the device
                y = self.processor(signal, sampling_rate=sampling_rate)
                y = y['input_values'][0]
                y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
                # print(y.shape)
                # run through model
                # first entry contains hidden state
                y = self.model(y)[0]

                # pool result and convert to numpy
                y = torch.mean(y, dim=1)
                y = y.detach().cpu().numpy()
        except RuntimeError as re:
            print(str(re))
            self.util.error(f'couldn\'t extract file: {file}')

        return y.flatten()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, 'no file')
        return feats
