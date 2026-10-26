import sys
sys.path.insert(0, "src")

import numpy as np
import torch

from video_utils import get_frame_data
from features import video_to_matrix, fill_gaps, COLS

# mean/std of the PHONE data, saved by the model trained on all 191 features
checkpoint = torch.load("models/sign_gru.pt")
mean = checkpoint["mean"].numpy()
std = checkpoint["std"].numpy()

matrix = fill_gaps(video_to_matrix(get_frame_data("data/raw/webcam_test.mp4")))
z = (matrix - mean) / std   # how many "spreads" each value is away from the phone data

for name, cols in COLS.items():
  print(name, round(np.abs(z[:, cols].mean(axis=0)).mean(), 2))

shift = z.mean(axis=0)
worst = np.argsort(-np.abs(shift))[:10]
print(worst)
print(shift[worst].round(1))