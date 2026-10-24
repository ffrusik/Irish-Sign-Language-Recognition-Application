import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from video_utils import get_frame_data
from features import video_to_matrix, fill_gaps, resample

SEQ_LEN = 30  # every repetition becomes exactly 30 frames
OUT_DIR = Path("data/processed/landmarks")

def main():
  # group repetitions by video, so each video is run through MediaPipe only once
  by_video = defaultdict(list)
  with open("data/meta/segments.csv", newline="") as f:
    for row in csv.DictReader(f):
      by_video[row["video"]].append(row)

  for video, rows in by_video.items():
    frames = get_frame_data(video)                    # hands + pose for every frame
    matrix = fill_gaps(video_to_matrix(frames))       # (T, 191), short dropouts filled
    for row in rows:
      start, end = int(row["start"]), int(row["end"])
      sequence = resample(matrix[start:end], SEQ_LEN)  # (30, 191)
      name = f"{Path(video).stem}_rep{int(row['rep']):02d}.npy"
      out_path = OUT_DIR / row["sign"] / name
      out_path.parent.mkdir(parents=True, exist_ok=True)
      np.save(out_path, sequence)
    print(video, len(rows), "repetitions saved")

if __name__ == "__main__":
  main()