from pathlib import Path

import numpy as np

LANDMARKS_DIR = Path("data/processed/landmarks")

def load_all():
  # returns X (N, 30, 191), y (N,), sessions (N,), class names
  names = sorted(p.name for p in LANDMARKS_DIR.iterdir() if p.is_dir())
  X, y, sessions = [], [], []
  for label, sign in enumerate(names):
    for path in sorted((LANDMARKS_DIR / sign).glob("*.npy")):
      X.append(np.load(path))                  # one repetition: (30, 191)
      y.append(label)                          # class number of this sign
      sessions.append(path.stem.split("_")[2]) # "S1", "S2" or "S3"
  return np.array(X), np.array(y), np.array(sessions), names