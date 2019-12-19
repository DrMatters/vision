from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
from scipy.io import loadmat

from torchreid.data.datasets import VideoDataset
from torchreid.utils import read_json, write_json


class SAIVT_SoftBio(VideoDataset):
    """SAIVT-SoftBio.

    Dataset statistics:
        - identities: ??.
        - tracklets: ??.
        - cameras: ??.
    """
    dataset_dir = 'SAIVT-SoftBio'

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS-VID')
        self.split_dir = osp.join(self.dataset_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_1_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.dataset_dir, 'i-LIDS-VID/sequences/cam2')

        required_files = [
            self.dataset_dir,
            self.data_dir,
            self.split_dir
        ]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'.format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        train = self.process_data(train_dirs, cam1=True, cam2=True)
        query = self.process_data(test_dirs, cam1=True, cam2=False)
        gallery = self.process_data(test_dirs, cam1=False, cam2=True)

        super(SAIVT_SoftBio, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets