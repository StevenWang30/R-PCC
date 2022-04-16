import numpy as np
import ops.fps.fps_utils as fps

import time
import IPython
import os
import open3d as o3d
from ops.cpp_modules import segment_utils_cpp
import torch


class PointCloudSegment:
    def __init__(self, transform_map, plane_num=1):
        self.plane_num = plane_num

        self.transform_map = transform_map

    def calc_cluster_residual_radius(self, point_cloud, cluster_param, cpu=True):
        h, w, c = point_cloud.shape
        assert c == 3
        if cpu:
            diff = np.reshape(point_cloud, (h, w, 1, c)) - np.reshape(cluster_param, (1, 1, cluster_param.shape[0], c))
            residual = np.linalg.norm(diff, 2, -1)
        else:
            diff = point_cloud.view(h, w, 1, c) - cluster_param.unsqueeze(0).unsqueeze(0)
            residual = torch.norm(diff, 2, -1)
        return residual

    def calc_cluster_residual_depth(self, range_image, cluster_param, cpu=True):
        h, w, c = range_image.shape
        assert c == 1
        if cpu:
            residual = np.reshape(range_image, (h, w, 1)) - np.reshape(np.linalg.norm(cluster_param, 2, -1), (1, 1, cluster_param.shape[0]))
        else:
            residual = range_image - torch.norm(cluster_param, 2, -1).unsqueeze(0).unsqueeze(0)
        return residual

    def calc_plane_residual_vertical(self, point_cloud, plane_param, cpu=True):
        '''
            plane_param: 4 (ax + by + cz + d)
            point to plane: d = |ax + by + cz + d| / sqrt(a**2 + b**2 + c**2)
            output: residual is same as distance   H x W
        '''
        if cpu:
            plane_param = np.expand_dims(np.expand_dims(plane_param, 0), 0)
            residual = np.abs(np.sum(point_cloud * plane_param[..., :3], -1) + plane_param[..., 3]) / \
                       np.linalg.norm(plane_param[:, :3], 2, -1)
        else:
            plane_param = plane_param.unsqueeze(0).unsqueeze(0)
            residual = torch.abs(torch.sum(point_cloud * plane_param[..., :3], -1) + plane_param[..., 3]) / \
                       torch.norm(plane_param[..., :3], 2, -1)
        return residual

    def calc_plane_residual_depth(self, range_image, plane_param, transform_map, cpu=True):
        '''
            plane_param: 4 (ax + by + cz + d)
            transform_map: [A, B, C] for {x, y, z} --> x = A * depth, y = B * depth, z = C * depth
            center to point r: range_image depth
            center to plane range r': aAr' + bBr' + cCr' + d = 0 --> r' = -d / (aA + bB + cC)
            residual: delta_r = |r - r'|
            use square of the residual as the loss
            output: residual is same as distance
        '''
        if cpu:
            plane_param = np.expand_dims(np.expand_dims(plane_param, 0), 0)  # 1, 1, 4
            r_plane = -plane_param[..., 3] / np.sum(plane_param[..., :3] * transform_map, -1)
            residual = range_image[..., 0] - r_plane
        else:
            plane_param = plane_param.unsqueeze(0).unsqueeze(0)
            r_plane = -plane_param[..., 3] / torch.sum(plane_param[..., :3] * transform_map, -1)
            residual = range_image[..., 0] - r_plane
        return residual

    @staticmethod
    def ransac_plane_segmentation(point_cloud, threshold=0.1, ransac_n=10, num_iterations=100):
        # o3d
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(point_cloud)
        coefficients, indices = points_o3d.segment_plane(distance_threshold=threshold,
                                                         ransac_n=ransac_n,
                                                         num_iterations=num_iterations)
        return indices, coefficients

    def plane_angle_validation(self, plane_model, scan_idx, angle_threshold_scan_with_normal):
        # angle between scan with plane norm
        scan_vector = self.transform_map[scan_idx]
        alpha = np.arccos(np.abs(np.sum(np.expand_dims(plane_model[:3], 0) * scan_vector, -1)) / \
                          np.linalg.norm(plane_model[:3]) * np.linalg.norm(scan_vector, ord=2, axis=-1))
        if alpha.max() > np.pi * (angle_threshold_scan_with_normal / 180):
            # the plane is too sharp with LiDAR (error will be very large)
            return False
        else:
            return True

    def segment(self, point_cloud, range_image, segment_cfg, cpu=True):
        assert self.transform_map is not None, "Must set transform_map first."
        if self.plane_num > 1:
            assert NotImplementedError

        # find ground plane
        pc_filter = point_cloud[np.where(point_cloud[..., 2] < -1.5)]
        if pc_filter.shape[0] > 5000:
            random_idx = np.random.choice(pc_filter.shape[0], 5000, replace=False)
            pc_filter = pc_filter[random_idx]
        if pc_filter.shape[0] < 800:
            pc_filter = point_cloud.reshape((-1, 3))

        _, ground_model = self.ransac_plane_segmentation(pc_filter)

        segment_method = segment_cfg['segment_method']
        assert segment_method in ['FPS', 'DBSCAN']
        ground_threshold = segment_cfg['ground_vertical_threshold']

        if segment_method == 'FPS':
            # FPS segmentation and clustering
            cluster_num = segment_cfg['cluster_num']

            if cpu:
                depth_dif = self.calc_plane_residual_vertical(point_cloud, ground_model)
                pc_left = point_cloud[np.where(depth_dif > ground_threshold)]
                center_idx = fps.furthest_point_sample(torch.from_numpy(np.expand_dims(pc_left, 0)).float().cuda(),
                                                       cluster_num)
                center_idx = center_idx[0].cpu().numpy()
                cluster_centers = pc_left[center_idx]

                # segment ground and cluster points
                ground_residual = self.calc_plane_residual_depth(range_image, ground_model, self.transform_map)
                cluster_residual_radius = self.calc_cluster_residual_radius(point_cloud, cluster_centers)  # Note that this residual is abs
                # cluster_residual = self.calc_cluster_residual_depth(range_image, cluster_centers)
                distance = np.concatenate((np.expand_dims(ground_residual, -1), cluster_residual_radius), -1)
                seg_idx = np.argmax(-np.abs(distance), -1)
            else:
                range_image_cuda = torch.from_numpy(range_image).float().cuda()
                point_cloud = torch.from_numpy(point_cloud).float().cuda()
                ground_model = torch.from_numpy(ground_model).float().cuda()
                transform_map = torch.from_numpy(self.transform_map).float().cuda()
                depth_dif = self.calc_plane_residual_vertical(point_cloud, ground_model, cpu=cpu)
                nonground_mask = depth_dif > ground_threshold
                nonground_points = (point_cloud * nonground_mask.unsqueeze(-1)).view(-1, 3)
                center_idx = fps.furthest_point_sample(nonground_points.unsqueeze(0), cluster_num)
                cluster_centers = nonground_points[center_idx[0].long()]

                ground_residual = self.calc_plane_residual_depth(range_image_cuda, ground_model, transform_map, cpu=cpu)
                cluster_residual_radius = self.calc_cluster_residual_radius(point_cloud, cluster_centers, cpu=cpu)
                distance = torch.cat((ground_residual.unsqueeze(-1), cluster_residual_radius), -1)
                _, seg_idx = torch.max(-distance.abs(), -1)
                ground_model = ground_model.cpu().numpy()
                seg_idx = seg_idx.cpu().numpy()
        elif segment_method == 'DBSCAN':
            eps = segment_cfg['DBSCAN_eps']
            # min_points = segment_cfg['MIN_POINTS']
            min_points = 10
            # default eps=1.0, min_points=10

            ground_residual = self.calc_plane_residual_depth(range_image, ground_model, self.transform_map)
            nonground_idx = np.where(np.abs(ground_residual) > 0.5)

            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(point_cloud[nonground_idx])
            labels = np.array(points_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
            labels += 2  # -1 to 1, 0 to 2. # for plane is 1
            seg_idx = np.zeros((range_image.shape[0], range_image.shape[1]))
            seg_idx[nonground_idx] = labels
            seg_idx = seg_idx.astype(np.int32)
        else:
            assert NotImplementedError

        seg_idx[np.where(seg_idx > 0)] += 1  # for ground points as class 0, zero points as class 1, nonzero points as 2 to max
        seg_idx[np.where(range_image[..., 0] == 0)] = 1
        return seg_idx, ground_model

    def cluster_modeling(self, point_cloud, range_image, seg_idx, model_cfg):
        # assert method in ['point', 'plane', 'cylinder', 'sphere']
        model_method = model_cfg['model_method']
        assert model_method in ['point', 'plane']
        cluster_models = []
        ###########
        # model parameters:
        # point: [0, 0, 0, mean_depth]
        # plane: [a, b, c, d], ax + by + cz + d = 0
        ###########
        if model_method == 'point':
            cluster_models = segment_utils_cpp.point_modeling(range_image, seg_idx)
            cluster_models = np.concatenate((np.zeros((cluster_models.shape[0], 3)), np.expand_dims(cluster_models, -1)), -1)
            cluster_models = cluster_models[1:]
        else:
            # python version
            for i in range(seg_idx.max() + 1):
                if i == 0:
                    # ground
                    continue
                if i == 1:
                    # nonzero points
                    cluster_models.append([0, 0, 0, 0.0])
                    continue

                idx = np.where(seg_idx == i)
                cur_range = range_image[idx]
                if model_method == 'point':
                    cluster_models.append([0, 0, 0, cur_range.mean()])
                elif model_method == 'plane':
                    angle_threshold_scan_with_normal = model_cfg['angle_threshold']  # degree
                    if idx[0].shape[0] < 30:
                        cluster_models.append([0, 0, 0, cur_range.mean()])
                    else:
                        cur_points = point_cloud[idx]
                        _, plane_model = self.ransac_plane_segmentation(cur_points,
                                                                        ransac_n=4,
                                                                        num_iterations=10)

                        # judge line with plane angle
                        if self.plane_angle_validation(plane_model, idx, angle_threshold_scan_with_normal):
                            cluster_models.append(list(plane_model))
                        else:
                            # the plane is too sharp with LiDAR (error will be very large)
                            cluster_models.append([0, 0, 0, cur_range.mean()])
        return np.asarray(cluster_models)

    def intra_predict(self, seg_idx, model_param):
        # # python version
        # range_image_pred = np.zeros((self.transform_map.shape[0], self.transform_map.shape[1]))
        # for i in range(seg_idx.max() + 1):
        #     idx = np.where(seg_idx == i)
        #     model = model_param[i]
        #     if np.sum(model[:3]) == 0:
        #         cluster_depth = model[3]
        #         range_image_pred[idx] = cluster_depth
        #     else:
        #         plane_param = np.expand_dims(np.expand_dims(model, 0), 0)  # 1, 1, 4
        #         r_plane = -plane_param[..., 3] / np.sum(plane_param[..., :3] * self.transform_map, -1)
        #         range_image_pred[idx] = r_plane[idx]
        # return np.expand_dims(range_image_pred, -1)
        return segment_utils_cpp.intra_predict(seg_idx, model_param, self.transform_map)
