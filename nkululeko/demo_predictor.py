import pandas as pd
import numpy as np
import sounddevice as sd
import audiofile
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


class Demo_predictor:
    def __init__(self, model, file, is_list, feature_extractor, label_encoder, outfile):
        """Constructor setting up name and configuration"""
        self.model = model
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.is_list = is_list
        self.sr = 16000
        self.target = glob_conf.config["DATA"]["target"]
        self.util = Util("demo_predictor")
        self.file = file
        self.outfile = outfile

    def run_demo(self):
        signal, sr = None, 0
        if self.file is not None:
            if not self.is_list:
                sig, sr = audiofile.read(self.file)
                print(
                    f"predicting file: {self.file}, len: {len(sig)} bytes,"
                    f" sampling rate: {sr}"
                )
                self.predict_signal(sig, sr)
            else:
                df_res = pd.DataFrame()
                with open(self.file) as f:
                    first = True
                    for index, line in enumerate(f):
                        # first line might be "file"
                        if self.file.endswith(".csv") and first:
                            first = False
                        else:
                            sig, sr = audiofile.read(line.strip())
                            print(f"predicting file {index}: {line.strip()}")
                            res_dict = self.predict_signal(sig, sr)
                            df_tmp = pd.DataFrame(res_dict, index=[line.strip()])
                            df_res = pd.concat([df_res, df_tmp], ignore_index=False)
                    df_res = df_res.set_index(df_res.index.rename("file"))
                    if self.outfile is not None:
                        df_res.to_csv(self.outfile)
                    else:
                        self.util.debug(df_res)
        else:
            while True:
                signal = self.record_audio(3)
                self.predict_signal(signal, self.sr)

    #            self.play_audio(signal)

    def predict_signal(self, signal, sr):
        features = self.feature_extractor.extract_sample(signal, sr)
        scale_feats = self.util.config_val("FEATS", "scale", False)
        if scale_feats:
            # standard normalize the input features
            features = (features - features.mean()) / (features.std())
        features = np.nan_to_num(features)
        result_dict = self.model.predict_sample(features)
        dict_2 = {}
        if self.util.exp_is_classification():
            keys = result_dict.keys()
            if self.label_encoder is not None:
                for i, k in enumerate(keys):
                    ak = np.array(int(k)).reshape(1)
                    lab = self.label_encoder.inverse_transform(ak)[0]
                    dict_2[lab] = f"{result_dict[k]:.3f}"
                dict_2["predicted"] = max(dict_2, key=dict_2.get)
                print(dict_2)
                return dict_2
            else:
                print(result_dict)
                return result_dict
        else:
            # experiment is regression and returns one estimation
            dict_2["predicted"] = result_dict[0]
            print(dict_2)
            return dict_2

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
