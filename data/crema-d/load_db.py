import os

import audb


# set download directory to current
cwd = os.getcwd()
audb.config.CACHE_ROOT = cwd

# load the latest version of the data
db = audb.load("crema-d", format="wav", sampling_rate=16000, mixdown=True)
