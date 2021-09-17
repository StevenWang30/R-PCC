import numpy as np
from dataset.transformer import PCTransformer
import open3d as o3d
import struct


class DatasetTemplate:
    def __init__(self, datalist, dataset_cfg, channel_distribute_csv=None, use_radius_outlier_removal=False):
        self.data_list = []
        if datalist is not None:
            print('start load data list from ', datalist)
            for line in open(datalist, "r"):
                self.data_list.append(line.strip())

        # lidar param
        if dataset_cfg is not None:
            self.dataset_cfg = dataset_cfg
            self.PCTransformer = PCTransformer(dataset_cfg, channel_distribute_csv)
            self.transform_map = self.PCTransformer.transform_map

        self.use_radius_outlier_removal = use_radius_outlier_removal

    def __len__(self):
        return len(self.data_list)

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

    def load_data(self, file):
        # return N x 3 point cloud
        data_type = file.split('.')[-1]
        if data_type == 'txt':
            point_cloud = np.loadtxt(file)
        elif data_type == 'bin':
            point_cloud = np.fromfile(file, dtype=np.float32)
            point_cloud = point_cloud.reshape((-1, 4))
        elif data_type == 'npy' or data_type == 'npz':
            point_cloud = np.load(file)
        elif data_type == 'ply':
            pcd = o3d.io.read_point_cloud(file)
            point_cloud = np.asarray(pcd.points)
        elif data_type == 'pcd':
            pcd = o3d.io.read_point_cloud(file)
            point_cloud = np.asarray(pcd.points)
        else:
            assert False, 'File type not correct.'

        point_cloud = point_cloud[:, :3]
        return point_cloud

    def load_range_image_points_from_file(self, file):
        original_point_cloud = self.load_data(file)
        range_image = self.PCTransformer.point_cloud_to_range_image(original_point_cloud)
        range_image = np.expand_dims(range_image, -1)
        point_cloud = self.PCTransformer.range_image_to_point_cloud(range_image)
        return point_cloud, range_image, original_point_cloud

    def save_point_cloud_to_file(self, file, point_cloud, color=None):
        data_type = file.split('.')[-1]
        valid_idx = np.where(np.sum(point_cloud, -1) != 0)
        point_cloud = point_cloud[valid_idx]
        if data_type == 'txt':
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
            np.savetxt(file, point_cloud)
        elif data_type == 'bin':
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
            point_cloud.astype(np.float32).tofile(file)
        elif data_type == 'npy' or data_type == 'npz':
            point_cloud = np.concatenate((point_cloud, np.zeros((point_cloud.shape[0], 1))), -1)
            np.save(file, point_cloud)
        elif data_type == 'ply':
            point_cloud = point_cloud[:, :3]
            # Write header of .ply file
            with open(file, 'wb') as fid:
                fid.write(bytes('ply\n', 'utf-8'))
                fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
                fid.write(bytes('element vertex %d\n' % point_cloud.shape[0], 'utf-8'))
                fid.write(bytes('property float x\n', 'utf-8'))
                fid.write(bytes('property float y\n', 'utf-8'))
                fid.write(bytes('property float z\n', 'utf-8'))
                fid.write(bytes('end_header\n', 'utf-8'))

                # Write 3D points to .ply file
                for i in range(point_cloud.shape[0]):
                    fid.write(bytearray(struct.pack("fff", point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2])))
        elif data_type == 'pcd':
            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            if color is not None:
                color_vec = color[valid_idx]
                points_o3d.colors = o3d.utility.Vector3dVector(color_vec)
            o3d.io.write_point_cloud(file, points_o3d)
        else:
            assert False, 'File type not correct.'