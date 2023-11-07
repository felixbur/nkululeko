"""
Code copyright by Uwe Reichel
"""

import numpy as np
from split_utils import binning, optimize_traindevtest_split

np.random.seed(42)
y = np.random.rand(10)

# intrinsic binning by equidistant percentiles
yci = binning(y, nbins=3)

# extrinsic binning by explicit lower boundaries
yce = binning(y, lower_boundaries=[0, 0.3, 0.8])

print("yci:", yci)
print("yce:", yce)

"""
 yci: [0 2 2 1 0 0 0 2 1 2]
 yce: [1 2 1 1 0 0 0 2 1 1]

 now yci or yce can be used for stratification, e.g.
stratify_on = {"target": yci, ...}
... = optimize_traindevtest_split(..., y=y, stratify_on=stratify_on, ...)
"""
