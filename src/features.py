import numpy as np

def hand_to_vector(points, body):
  # points: array (21, 3) or None; body: (center, scale) from body_frame, or None
  # returns 65 numbers: hand shape (63) + wrist position relative to the body (2)
  if points is None:
    return np.zeros(65)
  wrist = points[0]
  size = np.linalg.norm(points[9] - wrist)  # distance wrist -> base of middle finger
  if size < 1e-6:
    return np.zeros(65)
  shape = (points - wrist) / size
  if body is None:
    position = np.zeros(2)
  else:
    center, scale = body
    position = (wrist[:2] - center) / scale
  return np.concatenate([shape.flatten(), position])

def pose_to_vector(pose, body):
  # nose and both elbows relative to the body: 3 points x (x, y) = 6 numbers
  if pose is None or body is None:
    return np.zeros(6)
  center, scale = body
  points = [pose[0], pose[13], pose[14]]  # nose, left elbow, right elbow
  return np.concatenate([(np.array([p.x, p.y]) - center) / scale for p in points])

def face_to_vector(face):
  # face: list of 52 blendshape scores or None
  # returns 53 numbers: 52 scores + a "face found" flag
  if face is None:
    return np.zeros(53)
  return np.concatenate([face, [1.0]])

def frame_to_vector(slots, pose, face, body):
  # one frame -> 191 numbers
  right = hand_to_vector(slots["Right"], body)
  left = hand_to_vector(slots["Left"], body)
  body_part = pose_to_vector(pose, body)
  flags = np.array([slots["Right"] is not None, slots["Left"] is not None], dtype=float)
  face_part = face_to_vector(face)
  return np.concatenate([right, left, body_part, flags, face_part])

def body_frame(pose):
  # pose: list of 33 landmarks or None
  # returns (center, scale): midpoint between shoulders and shoulder width, or None
  if pose is None:
    return None
  left, right = pose[11], pose[12]
  center = np.array([(left.x + right.x) / 2, (left.y + right.y) / 2])
  scale = np.hypot(left.x - right.x, left.y - right.y)
  if scale < 1e-6:
    return None
  return center, scale

def video_to_matrix(frames):
  # frames: list of (slots, pose, face) for the whole video
  # returns an array (number of frames, 191)
  rows = []
  body = None  # last known body frame
  for slots, pose, face in frames:
    new_body = body_frame(pose)
    if new_body is not None:
      body = new_body  # if the body is lost for a frame, keep using the last known one
    rows.append(frame_to_vector(slots, pose, face, body))
  return np.array(rows)

def resample(matrix, length=30):
  # matrix: (T, 191) -> (length, 191)
  # pick `length` frames evenly spread over the repetition
  indices = np.linspace(0, len(matrix) - 1, length).round().astype(int)
  return matrix[indices]

def fill_gaps(matrix, max_gap=3):
  # linearly fill short dropouts of a hand between two visible frames
  matrix = matrix.copy()  # do not modify the caller's array
  # (columns of the hand vector, column of its visibility flag)
  hands = ((slice(0, 65), 136), (slice(65, 130), 137))
  for block, flag in hands:
    visible = np.where(matrix[:, flag] == 1)[0]  # indices of frames with this hand
    for a, b in zip(visible[:-1], visible[1:]):  # neighbouring visible frames
      gap = b - a - 1  # number of missing frames between them
      if 0 < gap <= max_gap:
        for k in range(1, gap + 1):
          t = k / (gap + 1)  # how far we are from frame a to frame b (0..1)
          matrix[a + k, block] = (1 - t) * matrix[a, block] + t * matrix[b, block]
          matrix[a + k, flag] = 1
  return matrix

# column indices inside one frame vector (191 numbers)
COLS = {
  "hands": np.r_[0:130, 136:138],  # both hands (shape + position) and their visibility flags
  "body": np.r_[130:136],          # nose and elbows
  "face": np.r_[138:191],          # 52 blendshapes + "face found" flag
  "brows": np.r_[139:144],          # 5 blendshapes: browDown x2, browInnerUp, browOuterUp x2
  "eyes": np.r_[147:161],           # 14 blendshapes: blink, look, squint, wide
  "mouth_jaw": np.r_[161:188],      # 27 blendshapes: 4 jaw + 23 mouth
}