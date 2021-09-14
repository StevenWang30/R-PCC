import numpy as np
from datasets.dataset import DatasetTemplate
import glob
import IPython
import concurrent.futures as futures
from tqdm import tqdm


class KittiDataset(DatasetTemplate):
    def __init__(self, datalist=None, dataset_cfg=None, channel_distribute_csv=None, use_radius_outlier_removal=False):
        super(KittiDataset, self).__init__(datalist, dataset_cfg, channel_distribute_csv, use_radius_outlier_removal)
        print('dataset initialize finished.')

    def preprocess_txt_to_bin(self, data_root='/data/KITTI_rawdata/city_unsync'):
        #############
        # when download the KITTI raw data (unsync), the velodyne data will be saved as txt, which load very slow.
        # preprocess the txt file to create the bin file for fast load.
        #############
        file_list = glob.glob(os.path.join(data_root, '*/*/*/velodyne_points/data/*.txt'))
        file_list.sort()

        def save_txt_to_bin(file):
            save_path = file.replace('/velodyne_points/data/', '/velodyne_points/data_bin/').replace('.txt', '.bin')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            point_cloud = np.loadtxt(file)
            point_cloud.astype(np.float32).tofile(save_path)

        # for file in file_list:
        #     save_txt_to_bin(file)
        with futures.ThreadPoolExecutor(4) as executor:
            list(tqdm(executor.map(save_txt_to_bin, file_list), total=len(file_list)))


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    # # preprocess to create bin data.
    # dataset = KittiDataset()
    # dataset.preprocess_txt_to_bin()

    # visualization
    from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds, visualize_points_vertical_angle_distribution
    # data_list = '../../data/test_64E_KITTI_city.txt'
    data_list = '../../data/test_64E_KITTI_city_unsync.txt'
    dataset_cfg = './cfg/Velodyne_HDL_64E.yaml'
    dataset = KittiDataset(data_list, dataset_cfg)
    for i in range(len(dataset)):
        pc, ri, ori_pc, file_path = dataset[i]
        # visualize_points_vertical_angle_distribution(ori_pc)
        compare_point_clouds(pc, ori_pc, vis_all=True, save=False, vis=True)

    # IPython.embed()
