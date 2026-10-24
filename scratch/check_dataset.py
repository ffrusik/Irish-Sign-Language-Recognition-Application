import sys
sys.path.insert(0, "src")

import numpy as np
from dataset import load_all

X, y, sessions, names = load_all()
print(X.shape)                                     # expected: (90, 30, 191)
print(y.shape)                                     # expected: (90,)
print(names)                                       # expected: ['hello', 'how_are_you', 'sorry']
print(np.unique(sessions, return_counts=True))     # expected: S1, S2, S3, 30 each
print(np.unique(y, return_counts=True))           # expected: classes 0, 1, 2, 30 each