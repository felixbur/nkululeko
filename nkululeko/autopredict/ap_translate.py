"""A translator for text.

Currently based on google translate.
"""

from nkululeko.utils.util import Util


class TextTranslator:
    """Translator.

    translate text with the google translate model
    """

    def __init__(self, df, util=None):
        self.df = df
        if util is not None:
            self.util = util
        else:
            # create a new util instance
            # this is needed to access the config and other utilities
            # in the autopredict module
            self.util = Util("translator")
            
        self.language = self.util.config_val("PREDICT", "target_language", "en")
        from nkululeko.autopredict.google_translator import GoogleTranslator
        self.translator = GoogleTranslator(
            language=self.language,
            util=self.util,
        )

    def predict(self, split_selection):
        self.util.debug(f"translating text for {split_selection} samples")
        df = self.translator.translate_index(
            self.df
        )
        return_df = self.df.copy()
        return_df[self.language] = df[self.language].values
        return return_df
