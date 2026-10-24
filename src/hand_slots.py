import numpy as np

def hands_to_slots(result):
  # returns {"Right": array (21, 3) or None, "Left": array (21, 3) or None}
  hands = []
  for landmarks, handed in zip(result.hand_landmarks, result.handedness):
    points = np.array([[p.x, p.y, p.z] for p in landmarks])
    hands.append((handed[0].category_name, points))

  # if both hands got the same label, fix it by position:
  # in a non-mirrored video the hand on the image-left is the signer's right hand
  if len(hands) == 2 and hands[0][0] == hands[1][0]:
    hands.sort(key=lambda h: h[1][0][0])   # sort by wrist x
    hands[0] = ("Right", hands[0][1])
    hands[1] = ("Left", hands[1][1])

  slots = {"Right": None, "Left": None}
  for label, points in hands:
    slots[label] = points
  return slots