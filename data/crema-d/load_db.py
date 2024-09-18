import os

import audb

# set download directory to current
cwd = os.getcwd()
audb.config.CACHE_ROOT = cwd

# load the latest version of the data
db = audb.load("crema-d", version="1.3.0", verbose=True)
