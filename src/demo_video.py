import sys

from video_utils import get_frame_data, get_wrist_heights, find_segments
from features import video_to_matrix, fill_gaps, resample
from demo import load_model, predict

video = sys.argv[1]
model, checkpoint = load_model()

segments = find_segments(get_wrist_heights(video))              # where the signs are
matrix = fill_gaps(video_to_matrix(get_frame_data(video)))      # (T, 191)

for i, (start, end) in enumerate(segments, 1):
  probs = predict(model, checkpoint, resample(matrix[start:end], 30))
  name = checkpoint["names"][probs.argmax()]
  print(f"{i}: frames {start}-{end}: {name} ({probs.max():.2f})")