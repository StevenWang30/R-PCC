import numpy as np
import IPython
from easydict import EasyDict
import yaml
import math
import csv
import os
from ops.cpp_modules import dataset_utils_cpp


class PCTransformer:
    def __init__(self, lidar_cfg=None, channel_distribute_csv=None):
        if channel_distribute_csv is not None:
            self.even_dist = False
            channel = []
            vertical_angle = []
            with open(channel_distribute_csv, "r") as fin:
                reader = csv.DictReader(fin)
                for r in reader:
                    channel.append(int(r['channel']))
                    vertical_angle.append(float(r['vertical_angle']))
            self.vertical_angle = np.radians(np.array(vertical_angle))
        else:
            self.even_dist = True

        with open(lidar_cfg, 'r') as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.load(f)
        lidar_config = EasyDict(config)
        self.horizontal_FOV = lidar_config.HORIZONTAL_FOV * (np.pi / 180)
        self.vertical_max = lidar_config.VERTICAL_ANGLE_MAX * (np.pi / 180)
        self.vertical_min = lidar_config.VERTICAL_ANGLE_MIN * (np.pi / 180)
        self.vertical_FOV = self.vertical_max - self.vertical_min
        self.H = lidar_config.RANGE_IMAGE_HEIGHT
        self.W = lidar_config.RANGE_IMAGE_WIDTH

        self.transform_map = self.create_transform_map()

    def create_transform_map(self):
        transform_map = np.zeros((self.H, self.W, 3))
        for h in range(self.H):
            for w in range(self.W):
                if self.even_dist:
                    altitude = self.vertical_FOV * (h / (self.H - 1)) + self.vertical_min
                else:
                    altitude = self.vertical_angle[h]
                azimuth = self.horizontal_FOV * (w / self.W)
                transform_map[h, w, 0] = math.cos(altitude) * math.cos(azimuth)  # * depth
                transform_map[h, w, 1] = math.cos(altitude) * math.sin(azimuth)  # * depth
                transform_map[h, w, 2] = math.sin(altitude)  # * depth
        print('Transform map creation finished.')
        return transform_map.astype(np.float32)

    def calculate_vertical_angle(self, points):
        return np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], 2, -1))

    def calculate_horizon_angle(self, points):
        return np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)

    def point_cloud_to_range_image(self, point_cloud):
        if self.even_dist:
            range_image = dataset_utils_cpp.point_cloud_to_range_image_even(point_cloud.astype(np.float32), self.H, self.W,
                                                                            self.horizontal_FOV, self.vertical_max,
                                                                            self.vertical_min)  # 0.006s
        else:
            # python version
            range_image = np.zeros((self.H, self.W), dtype=np.float32)

            # horizontal index h
            horizontal_angle = self.calculate_horizon_angle(point_cloud)
            col = np.rint(horizontal_angle / self.horizontal_FOV * self.W)
            col = col % self.W

            # vertical index w
            vertical_angle = self.calculate_vertical_angle(point_cloud)
            if self.even_dist:
                vertical_resolution = (self.vertical_max - self.vertical_min) / (self.H - 1)
                row = np.rint((vertical_angle - self.vertical_min) / vertical_resolution)
            else:
                vertical_angle_dif = np.expand_dims(self.vertical_angle, 0) - np.expand_dims(vertical_angle, 1)
                row = np.argmin(np.abs(vertical_angle_dif), -1)
            # print('row min: ', row.min(), ' row max: ', row.max())
            # print(np.bincount(row.astype(np.int32)))
            row[np.where(row >= self.H)] = self.H - 1
            row[np.where(row < 0)] = 0

            # depth
            depth = np.linalg.norm(point_cloud[:, :3], 2, -1)
            range_image[row.astype(np.int32), col.astype(np.int32)] = depth
        return range_image

    def range_image_to_point_cloud(self, range_image):
        if len(range_image.shape) == 2:
            point_cloud = np.expand_dims(range_image, -1) * self.transform_map
        elif len(range_image.shape) == 3:
            point_cloud = range_image * self.transform_map
        else:
            assert False
        return point_cloud
