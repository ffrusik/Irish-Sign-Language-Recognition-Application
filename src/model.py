import torch
from torch import nn

class SignGRU(nn.Module):
  def __init__(self, n_features, n_classes, hidden=64, dropout=0.0):
    super().__init__()
    self.gru = nn.GRU(n_features, hidden, batch_first=True)
    self.dropout = nn.Dropout(dropout)
    self.head = nn.Linear(hidden, n_classes)

  def forward(self, x):
    # x: (batch, time, features)
    _, h = self.gru(x)       # h: (num_layers, batch, hidden)
    return self.head(self.dropout(h[-1]))  # (batch, n_classes)