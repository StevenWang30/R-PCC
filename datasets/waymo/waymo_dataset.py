import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from datasets.dataset import DatasetTemplate


class WaymoDataset(DatasetTemplate):
    def __init__(self, datalist=None, dataset_cfg=None, channel_distribute_csv=None, use_radius_outlier_removal=False):
        super(WaymoDataset, self).__init__(datalist, dataset_cfg, channel_distribute_csv, use_radius_outlier_removal)
        print('dataset initialize finished.')

    def __getitem__(self, index):
        assert NotImplementedError

