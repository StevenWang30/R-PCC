import os
BASE_DIR = os.path.dirname(__file__)
from datasets.dataset import DatasetTemplate
from datasets.kitti.kitti_dataset import KittiDataset
from datasets.waymo.waymo_dataset import WaymoDataset
from datasets.nclt.nclt_dataset import NcltDataset
from datasets.hkust.hkust_dataset import HkustCampusDataset
from datasets.oxford.oxford_dataset import OxfordCampusDataset

__all__ = {
    'KITTI': KittiDataset,
    'KITTI_test': KittiDataset,
    # 'NuScenesDataset': ,
    'Waymo': WaymoDataset,
    'NCLT': NcltDataset,  # http://robots.engin.umich.edu/nclt/
    'HKUSTCampus': HkustCampusDataset,  # ftps://143.89.78.112/ramlab_data/HKUST_dataset/20201105full/ust_campus_2020-11-05-14-50-04_0.bag
    'Oxford': OxfordCampusDataset
}

__cfg__ = {
    'KITTI': os.path.join(BASE_DIR, 'kitti/cfg/Velodyne_HDL_64E.yaml'),
    'KITTI_test': os.path.join(BASE_DIR, 'kitti/cfg/Velodyne_HDL_64E_unofficial.yaml'),
    # 'NuScenesDataset': ,
    'Waymo': os.path.join(BASE_DIR, 'waymo/cfg/Waymo_LiDAR_64.yaml'),
    'NCLT': os.path.join(BASE_DIR, 'nclt/cfg/Velodyne_HDL_32E.yaml'),  # http://robots.engin.umich.edu/nclt/manuals/HDL-32E_manual.pdf
    'HKUSTCampus': os.path.join(BASE_DIR, 'hkust/cfg/Velodyne_VLP_16E.yaml'),
    'Oxford': os.path.join(BASE_DIR, 'oxford/cfg/Velodyne_HDL_32E.yaml'),
}

# if non-even distribution of vertical channels
__csv__ = {
    'KITTI': None,
    'KITTI_test': None,
    # 'NuScenesDataset': ,
    'Waymo': None,
    'NCLT': None,
    'HKUSTCampus': None,
    'Oxford': os.path.join(BASE_DIR, 'oxford/cfg/Velodyne_HDL_32E_vertical_channel_distribution.csv'),
}


def build_dataset(datalist=None, dataset_name='KITTI', use_radius_outlier_removal=False):
    dataset = __all__[dataset_name](
        datalist,
        __cfg__[dataset_name],
        __csv__[dataset_name],
        use_radius_outlier_removal
        )
    return dataset
