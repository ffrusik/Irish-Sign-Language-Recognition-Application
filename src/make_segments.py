from pathlib import Path
import csv
import matplotlib.pyplot as plt 
from video_utils import get_wrist_heights, find_segments

rows = []
for path in sorted(Path("data/raw/videos").rglob("*.mp4")):
  ys = get_wrist_heights(str(path))
  segments = find_segments(ys)
  for i, (start, end) in enumerate(segments):
    rows.append({
      "video": path.as_posix(),
      "sign": path.parent.name,
      "rep": i + 1,
      "start": start,
      "end": end,
    })

Path("data/meta").mkdir(parents=True, exist_ok=True)
with open("data/meta/segments.csv", "w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=["video", "sign", "rep", "start", "end"])
  writer.writeheader()
  writer.writerows(rows)

print(len(rows))