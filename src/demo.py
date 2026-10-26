import sys
from pathlib import Path

import numpy as np
import torch

from model import SignGRU

def load_model(path="models/sign_gru.pt"):
  checkpoint = torch.load(path)
  model = SignGRU(checkpoint["n_features"], len(checkpoint["names"]))
  model.load_state_dict(checkpoint["model"])
  model.eval()
  return model, checkpoint

def predict(model, checkpoint, sequence):
  if "cols" in checkpoint:
    sequence = sequence[:, checkpoint["cols"]]   # keep only the columns the model was trained on

  # sequence: numpy array (30, 191) -> probabilities for every sign
  # x = (sequence - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()
  x = np.clip((sequence - checkpoint["mean"].numpy()) / checkpoint["std"].numpy(), -5, 5)
  x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 30, 191)
  with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
  return probs

if __name__ == "__main__":
  path = Path(sys.argv[1])
  model, checkpoint = load_model()
  probs = predict(model, checkpoint, np.load(path))
  for name, p in zip(checkpoint["names"], probs):
    print(f"{name}: {p:.2f}")
  print("Answer:", checkpoint["names"][probs.argmax()])
  print("True label (folder name):", path.parent.name)