import functools
from typing import Set

import cv2

from features_task.real import non_ml_features


def resolve_max_response(kps: Set[cv2.KeyPoint]) -> cv2.KeyPoint:
    return max(kps, key=lambda item: item.response)


orb: cv2.ORB = cv2.ORB_create()
detect_and_compute = functools.partial(orb.detectAndCompute, mask=None)

orb_downsampler = non_ml_features.DownsamplingFill(
    8, detect_and_compute, resolve_max_response
)

img = cv2.imread('../data/images/human/3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res = orb_downsampler.downsample(gray)

print(res)
