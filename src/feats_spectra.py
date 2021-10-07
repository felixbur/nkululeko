# opensmileset.py
from featureset import Featureset
from cacheddataset import CachedDataset
import os
import pandas as pd
from util import Util 
import audsp
import audeer
import audiofile 
import numpy as np
import torch 
import audtorch
import glob_conf

class Spectraloader(Featureset):

    def __init__(self, name, data_df):
        """Constructor setting the name"""
        Featureset.__init__(self, name, data_df)
        self.random_crop_length=500
        self.num_bands = 64
        self.freq_low = 50
        self.freq_high = 8000
        self.scale = audsp.define.AuditorySpectrumScale.MEL
        self.win_dur = 32
        self.hop_dur = 10
        self.power = False
        self.center = False
        self.reflect = False
        self.unit='ms'
        self.audspec = audsp.AuditorySpectrum(
            num_bands=self.num_bands,
            freq_low=self.freq_low,
            freq_high=self.freq_high,
            scale=self.scale,
        )
        self.spectrogram = audsp.Spectrogram(
            sampling_rate=16000,
            win_dur=self.win_dur,
            hop_dur=self.hop_dur,
            center=self.center,
            power=self.power,
            reflect=self.reflect,
            audspec=self.audspec,
            unit=self.unit,
        )
        self.transform = audsp.Log(
            self.spectrogram,
            c=1e-7
        )

        
    def make_feats(self):
        store = self.util.get_path('store')
        self.feats_dir = audeer.mkdir(store+self.name)
        try:
            extract = glob_conf.config['DATA']['needs_feature_extraction']
        except KeyError:
            extract = False

        if extract or not os.path.isfile(os.path.join(self.feats_dir, 'index.pkl')):
            self.util.debug('extracting spectra, this might take a while...')
            is_multi_index = False
            if isinstance(self.data_df.index, pd.MultiIndex):
                is_multi_index = True
                index = self.data_df.index
                filenames = [] 
                # print(self.data_df.head(1))
                for counter, (file, start, end) in audeer.progress_bar(
                    enumerate(index), 
                    total=len(index), 
                    desc='Extraction'
                ):
                    offset = start.total_seconds()
                    if end != end:
                        duration = None
                    else:
                        duration = (end-start).total_seconds()
                    signal, fs = audiofile.read(
                        file,
                        offset=offset,
                        duration=duration,
                        always_2d=True
                    )
                    spects = self.transform(signal, fs)
                    filename = os.path.join(self.feats_dir, '{:08}.npy'.format(counter))
                    np.save(filename, spects)
                    filenames.append(filename)
            else:
                filenames = []
                for counter, file in audeer.progress_bar(
                    enumerate(self.data_df.index), 
                    total=len(self.data_df.index), 
                    desc='Extraction'
                ):
                    signal, fs = audiofile.read(
                        file)
                    spects = self.transform(signal, fs)
                    filename = os.path.join(self.feats_dir, '{:08}.npy'.format(counter))
                    np.save(filename, spects)
                    filenames.append(filename)
            data = pd.DataFrame(
                index=self.data_df.index,
                data=filenames,
                columns=['filename']
            )
            data.to_pickle(os.path.join(self.feats_dir, 'index.pkl'))
            data.to_csv(os.path.join(self.feats_dir, 'index.csv'))
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else: 
            self.util.debug('spectra already extracted')

    def get_loader(self):
        target = glob_conf.config['DATA']['target']
        feats = pd.read_pickle(self.feats_dir+'/index.pkl')
        dataset = CachedDataset(
            features=feats,
            df=self.data_df,
            target_column=target,
            transform=audtorch.transforms.RandomCrop(self.random_crop_length)
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=True,
            num_workers=5
        )    
        return loader