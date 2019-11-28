import functools
from typing import List

import cv2
import numpy as np

from features_task.real import non_ml_features


def resolve_max_response(keypoints: List[cv2.KeyPoint],
                         descriptors: List[np.ndarray]) -> np.ndarray:
    index = range(len(keypoints))
    max_index = max(index, key=lambda item: keypoints[item].response)
    return descriptors[max_index]


def resolve_first(keypoints: List[cv2.KeyPoint],
                  descriptors: List[np.ndarray]) -> np.ndarray:
    return descriptors[0]


orb: cv2.ORB = cv2.ORB_create()
detect_and_compute = functools.partial(orb.detectAndCompute, mask=None)

max_resp_downs = non_ml_features.DownsamplingFill(
    8, detect_and_compute, resolve_max_response
)

first_downs = non_ml_features.DownsamplingFill(
    8, detect_and_compute, resolve_first
)

img = cv2.imread('../data/images/human/3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

res_max = max_resp_downs.downsample(gray)
res_first = first_downs.downsample(gray)

print(res_max)
