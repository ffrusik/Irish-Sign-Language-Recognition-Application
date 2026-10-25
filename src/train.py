import numpy as np
import torch
from torch import nn

from dataset import load_all
from model import SignGRU

X, y, sessions, names = load_all()
test = sessions == "S3"   # hold out one whole session for testing
train = ~test

# standardize every feature using statistics of the training part only
mean = X[train].mean(axis=(0, 1))          # (191,)
std = X[train].std(axis=(0, 1)) + 1e-6     # (191,)
X = (X - mean) / std

X_train = torch.tensor(X[train], dtype=torch.float32)
y_train = torch.tensor(y[train], dtype=torch.long)
X_test = torch.tensor(X[test], dtype=torch.float32)
y_test = torch.tensor(y[test], dtype=torch.long)

model = SignGRU(n_features=X.shape[2], n_classes=len(names))
loss_fn = nn.CrossEntropyLoss()

logits = model(X_train)
loss = loss_fn(logits, y_train)
print(logits.shape)   # expected: torch.Size([60, 3])
print(loss.item())    # expected: about 1.1
print(X[train].mean(axis=(0, 1))[:5])
print(X[train].std(axis=(0, 1))[:5])

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
  optimizer.zero_grad()
  loss = loss_fn(model(X_train), y_train)
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    with torch.no_grad():
      test_pred = model(X_test).argmax(dim=1)
      test_acc = (test_pred == y_test).float().mean().item()
    print(epoch, round(loss.item(), 3), round(test_acc, 2))

torch.save({
  "model": model.state_dict(),
  "mean": torch.tensor(mean, dtype=torch.float32),
  "std": torch.tensor(std, dtype=torch.float32),
  "names": names,
  "n_features": X.shape[2],
}, "models/sign_gru.pt")