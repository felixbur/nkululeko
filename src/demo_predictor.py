import sounddevice
import audiofile
import glob_conf
from util import Util
from model import Model


class Demo_predictor():


    def __init__(self, model, feature_extractor, label_encoder):
        """Constructor setting up name and configuration"""
        self.model = model
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.sr = 16000
        self.target = glob_conf.config['DATA']['target']
        self.util = Util()

    def run_demo(self):
        while True:
            signal = self.record_audio(3)
            features = self.feature_extractor.extract_sample(signal, self.sr)
            features = features.to_numpy()
            result = self.model.predict_sample(features)
            result = self.label_encoder.inverse_transform(result)[0]
            print(f'{self.target}: {result}')

    def record_audio(self, seconds): 
        print("recording ...") 
        y = sounddevice.rec(int(seconds * self.sr), samplerate=self.sr, channels=1) 
        sounddevice.wait() 
        y = y.T 
        return y 