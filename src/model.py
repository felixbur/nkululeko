# model.py

class Model:
    name = ''
    config = None
    df_train, df_test, feats_train, feats_test = None, None, None, None

    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        self.config = config
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
