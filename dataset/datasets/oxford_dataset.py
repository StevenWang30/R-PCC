import numpy as np
from dataset.dataset import DatasetTemplate
import glob
import IPython
import concurrent.futures as futures
from tqdm import tqdm


class OxfordCampusDataset(DatasetTemplate):
    def __init__(self, datalist=None, dataset_cfg=None, channel_distribute_csv=None, use_radius_outlier_removal=False):
        super(OxfordCampusDataset, self).__init__(datalist, dataset_cfg, channel_distribute_csv, use_radius_outlier_removal)
        print('dataset initialize finished.')

    def preprocess_pcd_to_bin(self, data_root='/data/Oxford_32E_dataset/point_cloud'):
        import open3d as o3d
        #############
        # when download the raw rosbag data, use <rosrun pcl_ros bag_to_pcd xxx.bag ./data/ > to transform rosbag to
        # pcd file. Then use this to transform pcd to bin.
        #############
        file_dir = glob.glob(os.path.join(data_root, '*'))
        file_dir.sort()

        for dir in file_dir:
            file_list = glob.glob(os.path.join(dir, 'velodyne_points/right/*.pcd'))
            file_list.sort()
            for (i, file) in enumerate(file_list):
                save_path = file.replace('velodyne_points/right', 'velodyne_points/right_bin')
                save_path = save_path.replace(save_path.split('/')[-1], '%010d.bin' % i)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                pcd = o3d.io.read_point_cloud(file)
                point_cloud = np.asarray(pcd.points)
                point_cloud = np.append(point_cloud, np.zeros((point_cloud.shape[0], 1)), axis=1)
                point_cloud.astype(np.float32).tofile(save_path)
                print('save to ', save_path)

        # file_list = glob.glob(os.path.join(data_root, '/*/velodyne_points/*.pcd'))
        # file_list.sort()
        #
        # def save_pcd_to_bin(file):
        #     save_path = file.replace('/velodyne_points/', '/velodyne_points_bin/').replace('.pcd', '.bin')
        #     if not os.path.exists(os.path.dirname(save_path)):
        #         os.makedirs(os.path.dirname(save_path))
        #     point_cloud = np.loadtxt(file)
        #     point_cloud.astype(np.float32).tofile(save_path)
        #
        # for file in file_list:
        #     preprocess_pcd_to_bin(file)
        # with futures.ThreadPoolExecutor(4) as executor:
        #     list(tqdm(executor.map(save_txt_to_bin, file_list), total=len(file_list)))


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    # # preprocess to create bin data.
    # dataset = OxfordCampusDataset()
    # dataset.preprocess_pcd_to_bin()

    # visualization
    from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds, visualize_points_vertical_angle_distribution
    data_list = '../../data/test_32E_Oxford_dataset.txt'
    dataset_cfg = './cfg/Velodyne_HDL_32E.yaml'
    dataset = OxfordCampusDataset(data_list, dataset_cfg)
    for i in range(len(dataset)):
        pc, ri, ori_pc, file_path = dataset[i]
        # visualize_points_vertical_angle_distribution(ori_pc)
        compare_point_clouds(pc, ori_pc, vis_all=True, save=False, vis=True)


    # https://oxford-robotics-institute.github.io/radar-robotcar-dataset/downloads

    # IPython.embed()
