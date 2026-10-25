import sys
sys.path.insert(0, "src")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from dataset import load_all

X, y, sessions, names = load_all()
X_flat = X.reshape(len(X), -1)   # (90, 30, 191) -> (90, 5730)

# hold out one whole session as the test set, three times (S1, S2, S3)
for held_out in ["S1", "S2", "S3"]:
  test = sessions == held_out
  train = ~test

  # standardize using statistics of the training part only
  scaler = StandardScaler().fit(X_flat[train])
  model = LogisticRegression(max_iter=1000)
  model.fit(scaler.transform(X_flat[train]), y[train])

  pred = model.predict(scaler.transform(X_flat[test]))
  accuracy = (pred == y[test]).mean()
  print(f"test = {held_out}: accuracy {accuracy:.2f}")
  print(confusion_matrix(y[test], pred))