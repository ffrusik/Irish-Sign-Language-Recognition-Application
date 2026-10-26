import sys

import numpy as np
import torch
from torch import nn

from dataset import load_all
from model import SignGRU

from features import COLS

GROUPS = {
  "hands": COLS["hands"],
  "hands_body": np.r_[COLS["hands"], COLS["body"]],
  "hands_face": np.r_[COLS["hands"], COLS["face"]],
  "all": np.arange(191),
  "hands_body_mouth": np.r_[COLS["hands"], COLS["body"], COLS["brows"], COLS["mouth_jaw"]],
  "hands_body_eyes": np.r_[COLS["hands"], COLS["body"], COLS["eyes"]],
  "hands_body_mouth_eyes": np.r_[COLS["hands"], COLS["body"], COLS["brows"], COLS["mouth_jaw"], COLS["eyes"]],
}
group = sys.argv[1]   # which feature set to train on
cols = GROUPS[group]

X, y, sessions, names = load_all()
X = X[:, :, cols]

final = len(sys.argv) > 2 and sys.argv[2] == "final"

test = sessions == "S3"   # hold out one whole session for testing
train = np.ones(len(sessions), dtype=bool) if final else ~test

# standardize every feature using statistics of the training part only
mean = X[train].mean(axis=(0, 1))          # (191,)  or (n_features,) depending on the group
std = X[train].std(axis=(0, 1)) + 1e-6     # (191,) or (n_features,) depending on the group
# X = (X - mean) / std
X = np.clip((X - mean) / std, -5, 5)

X_train = torch.tensor(X[train], dtype=torch.float32)
y_train = torch.tensor(y[train], dtype=torch.long)
X_test = torch.tensor(X[test], dtype=torch.float32)
y_test = torch.tensor(y[test], dtype=torch.long)

DROPOUT = 0.3
FACE_DROP = 0.3 # use face features only with this probability during training (to avoid overfitting to the face and rely on hands more)
face_cols = np.r_[COLS["brows"], COLS["mouth_jaw"], COLS["eyes"]]
face_mask = torch.tensor(np.isin(cols, face_cols))   # True for face columns in the chosen feature set
model = SignGRU(n_features=X.shape[2], n_classes=len(names), dropout=DROPOUT)
loss_fn = nn.CrossEntropyLoss()

logits = model(X_train)
loss = loss_fn(logits, y_train)
print(logits.shape)   # expected: torch.Size([60, 3])
print(loss.item())    # expected: about 1.1
print(X[train].mean(axis=(0, 1))[:5])
print(X[train].std(axis=(0, 1))[:5])

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

for epoch in range(100):
  model.train()
  optimizer.zero_grad()

  drop = (torch.rand(len(X_train)) < FACE_DROP)[:, None, None]   # (N, 1, 1): which examples lose the face
  x = torch.where(drop & face_mask, torch.zeros_like(X_train), X_train)
  loss = loss_fn(model(x), y_train)

  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    model.eval()
    with torch.no_grad():
      test_logits = model(X_test)
      test_loss = loss_fn(test_logits, y_test).item()
      # test_pred = model(X_test).argmax(dim=1)
      test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
    print(epoch, round(loss.item(), 3), round(test_loss, 3), round(test_acc, 2))

torch.save({
  "model": model.state_dict(),
  "mean": torch.tensor(mean, dtype=torch.float32),
  "std": torch.tensor(std, dtype=torch.float32),
  "names": names,
  "n_features": X.shape[2],
  "cols": cols.tolist(),
}, f"models/sign_gru_{group}{'_final' if final else ''}.pt")