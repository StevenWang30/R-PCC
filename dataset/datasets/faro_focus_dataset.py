import numpy as np
from dataset.dataset import DatasetTemplate
import glob
import IPython
import concurrent.futures as futures
from tqdm import tqdm


class FaroFocusDataset(DatasetTemplate):
    # MEMS dataset
    def __init__(self, datalist=None, dataset_cfg=None, channel_distribute_csv=None, use_radius_outlier_removal=False):
        super(FaroFocusDataset, self).__init__(datalist, dataset_cfg, channel_distribute_csv, use_radius_outlier_removal)
        print('dataset initialize finished.')

    def __getitem__(self, index):
        file_name = self.data_list[index]
        original_point_cloud = self.load_data(file_name)  # original point cloud
        if self.use_radius_outlier_removal:
            # slow. plz preprocess data.
            import open3d as o3d
            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(original_point_cloud)
            cloud_filtered, index = points_o3d.remove_radius_outlier(nb_points=3, radius=1)
            point_cloud = np.asarray(cloud_filtered.points)
        else:
            point_cloud = original_point_cloud
        range_image = self.PCTransformer.point_cloud_to_range_image(point_cloud)
        range_image = np.expand_dims(range_image, -1)
        point_cloud = self.PCTransformer.range_image_to_point_cloud(range_image)
        return point_cloud, range_image, original_point_cloud, file_name


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    # # preprocess to create bin data.
    # dataset = HkustCampusDataset()
    # dataset.preprocess_pcd_to_bin()

    from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds, visualize_points_vertical_angle_distribution
    
    data_list = '../../external_usage/data/faro_focus_datalist.txt'
    dataset_cfg = '../../dataset/lidar_cfg/Velodyne_VLP_16.yaml'
    dataset = FaroFocusDataset(data_list, dataset_cfg)
    pc = dataset.load_data(dataset.data_list[0])
    save_point_cloud_to_pcd(pc, 'a.pcd')

