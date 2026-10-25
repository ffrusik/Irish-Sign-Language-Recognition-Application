import sys
sys.path.insert(0, "src")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset import load_all
from features import COLS

X, y, sessions, names = load_all()

def evaluate(X_part):
  # mean accuracy over three runs, each holding out one whole session
  X_flat = X_part.reshape(len(X_part), -1)
  scores = []
  for held_out in ["S1", "S2", "S3"]:
    test = sessions == held_out
    train = ~test
    scaler = StandardScaler().fit(X_flat[train])
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(X_flat[train]), y[train])
    pred = model.predict(scaler.transform(X_flat[test]))
    scores.append((pred == y[test]).mean())
  return np.mean(scores)

experiments = {
  "hands only": COLS["hands"],
  "body only": COLS["body"],
  "face only": COLS["face"],
  "hands + body": np.r_[COLS["hands"], COLS["body"]],
  "hands + body + face": np.arange(191),
}

for name, cols in experiments.items():
  print(f"{name}: {evaluate(X[:, :, cols]):.2f}")