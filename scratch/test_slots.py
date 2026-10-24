import sys
sys.path.insert(0, "src")

import csv
import numpy as np
from pathlib import Path

from features import resample, video_to_matrix, fill_gaps
from video_utils import get_frame_data

frames = get_frame_data("data/raw/videos/how_are_you/P00_HowAreYou_S1.mp4")
print(len(frames))                                # expected: 787
print(sum(f is not None for _, _, f in frames))   # frames with a face found
print(len(frames[100][2]))                        # expected: 52

matrix = video_to_matrix(frames)
print(matrix.shape)   # expected: (787, 191)
print(matrix[40][130:])     # frame 40 (no hands): hand parts are zeros, body part is not


with open("data/meta/segments.csv", newline="") as f:
  rows = list(csv.DictReader(f))

# take the first repetition of this video
row = next(r for r in rows if r["video"].endswith("P00_HowAreYou_S1.mp4"))
start, end = int(row["start"]), int(row["end"])
print(start, end)

rep = matrix[start:end]
print(rep.shape)                # (end - start, 138)
print(resample(rep).shape)      # expected: (30, 191)
print(resample(rep)[:, 136:138].T)  # visibility flags over 30 frames

flags = rep[:, 136:138]   # last two columns: right, left visibility

def gap_lengths(column):
  # lengths of runs of zeros (hand missing) strictly between two visible frames
  gaps = []
  run = 0
  seen = False  # have we already seen the hand in this repetition?
  for value in column:
    if value == 1:
      if seen and run > 0:
        gaps.append(run)
      seen = True
      run = 0
    else:
      run += 1
  return gaps

print("right gaps:", gap_lengths(flags[:, 0]))
print("left gaps: ", gap_lengths(flags[:, 1]))


filled = fill_gaps(rep)
flags = filled[:, 136:138]
print("right gaps:", gap_lengths(flags[:, 0]))
print("left gaps: ", gap_lengths(flags[:, 1]))


files = sorted(Path("data/processed/landmarks").rglob("*.npy"))
print(len(files))                       # expected: 90
print(np.load(files[0]).shape)          # expected: (30, 138)
print(any(np.isnan(np.load(f)).any() for f in files))  # expected: False