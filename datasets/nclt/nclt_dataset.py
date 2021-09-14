import numpy as np
from datasets.dataset import DatasetTemplate
import struct
import glob
import IPython


class NcltDataset(DatasetTemplate):
    def __init__(self, datalist=None, dataset_cfg=None, channel_distribute_csv=None, use_radius_outlier_removal=False):
        super(NcltDataset, self).__init__(datalist, dataset_cfg, channel_distribute_csv, use_radius_outlier_removal)
        print('dataset initialize finished.')

    def preprocess_original_utf8_to_bin_file(self, data_root='/data/NCLT_Dataset_32E_LiDAR'):
        #############
        # dataset is downloaded from http://robots.engin.umich.edu/nclt/
        # original velodyne data is saved in utf-8 binary file type.
        # use this preprocess func to create the proper bin file for fast loading.
        #############
        file_dir = glob.glob(os.path.join(data_root, '*_vel'))
        file_dir.sort()

        for dir in file_dir:
            file_list = glob.glob(os.path.join(os.path.join(dir, '*/velodyne_sync/*.bin')))
            file_list.sort()
            for (i, file) in enumerate(file_list):
                save_path = file.replace('velodyne_sync', 'velodyne_sync_bin')
                save_path = save_path.replace(save_path.split('/')[-1], '%010d.bin' % i)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                point_cloud = self.load_original_utf8_data(file)
                point_cloud = np.append(point_cloud, np.zeros((point_cloud.shape[0], 1)), axis=1)
                point_cloud.astype(np.float32).tofile(save_path)
                print('save to ', save_path)

    def load_original_utf8_data(self, file):
        #############
        # Example code to read a velodyne_sync/[utime].bin file from download data.
        # For example: file = '/home/skwang/Downloads/2013-01-10_vel/2013-01-10/velodyne_sync/1357847238732390.bin'
        #############
        point_cloud = []
        with open(file, "rb") as f_bin:
            while True:
                x_str = f_bin.read(2)
                if len(x_str) == 0:  # eof
                    break
                x = struct.unpack('<H', x_str)[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]
                x, y, z = self.convert(x, y, z)
                point_cloud += [[x, y, z]]
        point_cloud = np.asarray(point_cloud)
        return point_cloud

    def convert(self, x_s, y_s, z_s):
        scaling = 0.005  # 5 mm
        offset = -100.0
        x = x_s * scaling + offset
        y = y_s * scaling + offset
        z = z_s * scaling + offset
        return x, y, z


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

    # # preprocess to create bin data.
    # dataset = NcltDataset()
    # dataset.preprocess_original_utf8_to_bin_file()

    # visualization
    from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds
    import time

    data_list = '../../data/test_32E_NCLT_dataset.txt'
    csv = './cfg/Velodyne_HDL_32E_vertical_channel_distribution.csv'
    dataset_cfg = './cfg/Velodyne_HDL_32E.yaml'
    dataset = NcltDataset(data_list, dataset_cfg, csv)
    for i in range(len(dataset)):
        t = time.time()
        print('\n\n')
        pc, ri, ori_pc, file_path = dataset[i]
        print('time cost: ', time.time() - t)
        compare_point_clouds(pc, ori_pc, vis_all=True, save=False, vis=True)

    # IPython.embed()