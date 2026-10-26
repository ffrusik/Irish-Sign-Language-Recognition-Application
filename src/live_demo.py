import time
from collections import deque

import cv2

from video_utils import FrameProcessor
from features import video_to_matrix, fill_gaps, resample
from demo import load_model, predict

MAX_GAP = 8    # frames without hands that still belong to the same sign
MIN_LEN = 15   # shorter movements are ignored (noise)
PAD = 5        # extra frames kept before and after the sign
SEQ_LEN = 30
THRESHOLD = 0.7

def main():
  # model, checkpoint = load_model()
  model, checkpoint = load_model("models/sign_gru_hands_body_mouth_final.pt")
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  processor = FrameProcessor()
  start = time.time()
  last_ts = -1

  before = deque(maxlen=PAD)  # the last frames before a sign starts
  buffer = []                 # frames of the current sign
  recording = False
  missing = 0                 # consecutive frames without hands
  answer = ""
  show_points = True

  while True:
    ok, frame = cap.read()
    if not ok:
      break
    timestamp_ms = max(int((time.time() - start) * 1000), last_ts + 1)
    last_ts = timestamp_ms
    data = processor.process(frame, timestamp_ms)  # (slots, pose, face)
    slots = data[0]
    hands_visible = slots["Right"] is not None or slots["Left"] is not None

    if hands_visible:
      if not recording:
        recording = True
        buffer = list(before)  # start with a few frames from before the hands appeared
      buffer.append(data)
      missing = 0
    elif recording:
      buffer.append(data)
      missing += 1
      if missing > MAX_GAP:  # hands are gone for long enough: the sign is over
        recording = False
        buffer = buffer[:len(buffer) - (missing - PAD)]  # keep only PAD frames after the sign
        if len(buffer) >= MIN_LEN + 2 * PAD:
          matrix = fill_gaps(video_to_matrix(buffer))
          probs = predict(model, checkpoint, resample(matrix, SEQ_LEN))

          best = probs.max().item()
          name = checkpoint["names"][probs.argmax()]
          answer = f"{name} ({best:.2f})" if best >= THRESHOLD else f"? {name} {best:.2f}"
        buffer = []
        before.clear()
    else:
      before.append(data)  # idle: remember the most recent frames

    # display points
    if show_points:
      h, w = frame.shape[:2]
      for hand in slots.values():
        if hand is not None:
          for x, y, _ in hand:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
      pose = data[1]
      if pose is not None:
        for i in (0, 11, 12, 13, 14): # nose, shoulders, elbows
          cv2.circle(frame, (int(pose[i].x * w), int(pose[i].y * h)), 6, (255, 0, 0), -1)   # body points: blue

    shown = cv2.flip(frame, 1)  # mirror only what we display
    cv2.putText(shown, "REC" if recording else "idle", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
    cv2.putText(shown, answer, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.imshow("ISL demo", shown)

    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
    if key == ord("d"):
      show_points = not show_points

  cap.release()
  processor.close()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()