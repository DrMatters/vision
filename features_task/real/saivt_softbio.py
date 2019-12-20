from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import pathlib

import cv2
import xmltodict
from sklearn import model_selection

from torchreid import data
from torchreid.data import VideoDataset


class Saivt_SoftBioImageReader(data.ImageReader):
    def read_image(self, path):
        # 1. get roi
        # 2. cut selected image
        path = pathlib.Path(path)
        directory = path.parents[0]
        camera_no = path.parents[0].stem[-2:]
        subject_no = path.parents[1].stem[-3:]
        im_no = int(path.stem[-4:])

        roi_info = self.get_roi(camera_no, directory, im_no, subject_no)
        l = int(roi_info['@l'])
        r = int(roi_info['@r'])
        t = int(roi_info['@t'])
        b = int(roi_info['@b'])

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cropped = image[t:b, l:r]
        cv2.imshow('crop', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return cropped

    @staticmethod
    def get_roi(camera_no, directory, im_no, subject_no):
        with open(directory / f'Sub{subject_no}-Cam{camera_no}-ROI.xml') as roi_data:
            roi_dict = xmltodict.parse(roi_data.read())
            for im in roi_dict['SoftBiometricDatabaseSubjectView']['ROI']:
                # todo: binsearch?
                cur_no = int(im['@image'][-5 - 4:][:4])
                if cur_no == im_no:
                    return im


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

        reader = Saivt_SoftBioImageReader()
        super(Saivt_SoftBio, self).__init__(train, query, gallery,
                                            image_reader=reader, **kwargs)

    # def prepare_tracklets(self, persons):
    #     for pers_id, person in enumerate(persons):
    #         for


if __name__ == "__main__":
    a = Saivt_SoftBioImageReader()
    a.read_image('/Volumes/drmatters/datasets/SAIVT-SoftBio/Uncontrolled/'
                 'Subject001/Camera01/SAIVT-SBD-Cam01-0365.jpeg')
