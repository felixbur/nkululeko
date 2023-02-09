import nkululeko.glob_conf as glob_conf
from nkululeko.util import Util
import numpy as np
import sounddevice as sd
import audiofile

class Demo_predictor():


    def __init__(self, model, file, feature_extractor, label_encoder):
        """Constructor setting up name and configuration"""
        self.model = model
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.sr = 16000
        self.target = glob_conf.config['DATA']['target']
        self.util = Util()
        self.file = file

    def run_demo(self):
        signal, sr = None, 0
        if self.file is not None:
            sig, sr = audiofile.read(self.file)
            print(f'predicting file: {self.file}, len: {len(sig)} bytes, sampling rate: {sr}')
            self.predict_signal(sig, sr)
        else:
            while True:
                signal = self.record_audio(3)
                self.predict_signal(signal, self.sr)
    #            self.play_audio(signal)

    def predict_signal(self, signal, sr):
        features = self.feature_extractor.extract_sample(signal, sr)
        result_dict = self.model.predict_sample(features)
        keys = result_dict.keys()
        dict_2 = {}
        for i, k in enumerate(keys):
            ak = np.array(int(k)).reshape(1)
            lab = self.label_encoder.inverse_transform(ak)[0]
            dict_2[lab] = f'{result_dict[k]:.3f}'
        print(dict_2)


    def record_audio(self, seconds): 
        print("recording ...") 
        y = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1) 
        sd.wait() 
        y = y.T 
        return y 

    def play_audio(self, signal): 
        print("playback ...") 
        sd.play(signal.T, self.sr)
        status = sd.wait()