import os
BASE_DIR = os.path.dirname(__file__)
from dataset.dataset import DatasetTemplate
from dataset.datasets.kitti_dataset import KittiDataset
from dataset.datasets.nclt_dataset import NcltDataset
from dataset.datasets.hkust_dataset import HkustCampusDataset
from dataset.datasets.oxford_dataset import OxfordCampusDataset

__dataset__ = {
    'KITTI': KittiDataset,
    'KITTI_test': KittiDataset,
    # 'NuScenesDataset': ,
    # 'Waymo': WaymoDataset,
    'NCLT': NcltDataset,  # http://robots.engin.umich.edu/nclt/
    'HKUSTCampus': HkustCampusDataset,  # ftps://143.89.78.112/ramlab_data/HKUST_dataset/20201105full/ust_campus_2020-11-05-14-50-04_0.bag
    'Oxford': OxfordCampusDataset
}

__dataset_cfg__ = {
    'KITTI': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_64E.yaml'),
    'KITTI_test': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_64E_unofficial.yaml'),
    # 'Waymo': os.path.join(BASE_DIR, 'lidar_cfg/Waymo_LiDAR_64.yaml'),
    'NCLT': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_32E.yaml'),  # http://robots.engin.umich.edu/nclt/manuals/HDL-32E_manual.pdf
    'Oxford': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_32E.yaml'),
    'HKUSTCampus': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_VLP_16.yaml'),
}

# if LiDAR has non-even distribution of vertical channels
__dataset_csv__ = {
    'KITTI': None,
    'KITTI_test': None,
    # 'NuScenesDataset': ,
    # 'Waymo': None,
    'NCLT': None,
    'HKUSTCampus': None,
    'Oxford': None,  # os.path.join(BASE_DIR, 'lidar_cfg/example-Velodyne_HDL_32E_vertical_channel_distribution.csv'),
}

__lidar_cfg__ = {
    'VelodyneVLP16': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_VLP_16.yaml'),
    'Velodyne32E': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_32E.yaml'),
    'Velodyne64E': os.path.join(BASE_DIR, 'lidar_cfg/Velodyne_HDL_64E.yaml'),
}

__lidar_csv__ = {
    'VelodyneVLP16': None,
    'Velodyne32E': None,
    'Velodyne64E': None,
}


def build_dataset(datalist=None, dataset_name=None, lidar_type=None, use_radius_outlier_removal=False):
    # assert dataset_name is not None or lidar_type is not None, "Must set dataset name or LiDAR type."
    if dataset_name is not None:
        return __dataset__[dataset_name](
            datalist,
            __dataset_cfg__[dataset_name],
            __dataset_csv__[dataset_name],
            use_radius_outlier_removal
        )
    if lidar_type is not None:
        return DatasetTemplate(
            datalist,
            __lidar_cfg__[lidar_type],
            __lidar_csv__[lidar_type],
            use_radius_outlier_removal
        )
    else:
        return DatasetTemplate(datalist, dataset_cfg=None, use_radius_outlier_removal=use_radius_outlier_removal)
