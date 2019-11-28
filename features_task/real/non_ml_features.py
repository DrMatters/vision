import collections
from typing import Callable, Tuple, List, Set

import cv2
import numpy as np


class DownsamplingFill:
    def __init__(self, cell_side_len: int,
                 detect_and_compute: Callable[[np.ndarray], Tuple[List[cv2.KeyPoint], np.ndarray]],
                 resolve_multiple: Callable[[Set[cv2.KeyPoint]], cv2.KeyPoint]):
        self._cell_side = cell_side_len
        self._extract = detect_and_compute
        self._resolve = resolve_multiple

    def downsample(self, image: np.array):
        keypoints, descriptors = self._extract(image)
        cells_index_def = collections.defaultdict(set)
        keypoints_index = {}
        for index, keypoint in enumerate(keypoints):
            # save mapping from cv2.KeyPoint to it's index to find descriptor later
            keypoints_index[keypoint.pt + (keypoint.angle,)] = index
            cell_x = int(keypoint.pt[0] // self._cell_side)
            cell_y = int(keypoint.pt[1] // self._cell_side)
            cells_index_def[(cell_x, cell_y)].add(keypoint)

        grid_shape = np.asarray(image.shape) // self._cell_side
        # add one dimension for descriptor vector
        grid_shape = np.append(grid_shape, descriptors.shape[1])
        grid = np.zeros(grid_shape, dtype=np.uint8)

        for cell, cell_keypoints in cells_index_def.items():
            keypoint = self._resolve(cell_keypoints)
            index = keypoints_index[keypoint.pt + (keypoint.angle,)]
            descriptor = descriptors[index]
            grid[cell[1], cell[0]] = descriptor
        return grid
