import sys
sys.path.insert(0, "src")

import torch
from model import SignGRU

model = SignGRU(n_features=191, n_classes=3)
x = torch.randn(8, 30, 191)   # fake batch: 8 repetitions, 30 frames, 191 features
print(model(x).shape)         # expected: torch.Size([8, 3])
print(sum(p.numel() for p in model.parameters()))  # expected: 49539