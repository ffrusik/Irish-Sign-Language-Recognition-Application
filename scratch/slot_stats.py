import sys
sys.path.insert(0, "src")

import csv
from video_utils import get_slots_per_frame

PAD = 5  # same padding that find_segments adds on both sides of a repetition

with open("data/meta/segments.csv") as f:
  rows = list(csv.DictReader(f))

slots_by_video = {}
for row in rows:
  video = row["video"]
  if video not in slots_by_video:
    slots_by_video[video] = get_slots_per_frame(video)

print("sign rep frames right% left% any%")
for row in rows:
  start, end = int(row["start"]), int(row["end"])
  frames = slots_by_video[row["video"]][start + PAD : end - PAD + 1]
  n = len(frames)
  right = sum(s["Right"] is not None for s in frames)
  left = sum(s["Left"] is not None for s in frames)
  any_hand = sum(s["Right"] is not None or s["Left"] is not None for s in frames)
  print(row["sign"], row["rep"], n, round(100 * right / n), round(100 * left / n), round(100 * any_hand / n))