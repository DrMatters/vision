from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from sklearn import model_selection

import sys
import os
import os.path as osp

from torchreid.data import VideoDataset


class Saivt_SoftBio(VideoDataset):
    dataset_dir = 'SAIVT-SoftBio'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, 'Uncontrolled')

        person_ids = [f.name for f in os.scandir(self.data_dir) if f.is_dir()]
        train_pers, test_pers = model_selection.train_test_split(person_ids,
                                                                 random_state=42)

        print(person_ids)
        # train = ...
        # query = ...
        # gallery = ...

        super(Saivt_SoftBio, self).__init__(train, query, gallery, **kwargs)

    def prepare_tracklets(self, persons):
        for pers_id, person in enumerate(persons):
            for