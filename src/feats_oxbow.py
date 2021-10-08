# feats_oxbow.py

from util import Util
from featureset import Featureset
from feats_opensmile import Opensmileset


class Openxbow(Featureset):
    """Class to extract openXBOW processed opensmile features (https://github.com/openXBOW)"""

    def process(self):
        self.feats.to_csv('tmp.csv')