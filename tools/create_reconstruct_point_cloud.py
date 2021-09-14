import numpy as np
import argparse

import time
import IPython
from pathlib import Path
from utils.utils import np_size, sys_size, bit_size
import pickle
import io
import datetime
from utils.compress_utils import BasicCompressor, compress_plane_idx_map
from utils.compress_utils import compress_point_cloud, decompress_point_cloud, load_compressor_cfg
import copy
from utils.compress_utils import compress_point_cloud, decompress_point_cloud
from utils.segment_utils import PointCloudSegment
from datasets import build_dataset
from utils.compress_utils import QuantizationModule, extract_features
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
import copy

import glob
import concurrent.futures as futures
from tqdm import tqdm
from utils.visualize_utils import draw_qualitative_point_clouds
from utils.evaluate_metrics import calc_chamfer_distance, calc_point_to_point_plane_psnr


parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--data_dir', default='/data/KITTI_detection/training/velodyne_original')

parser.add_argument('--dataset', default='KITTI')
parser.add_argument('--compressor_yaml', default=os.path.join(BASE_DIR, 'cfgs/compressor.yaml'))

parser.add_argument('--basic_compressor', type=str, default=None, help='for manual setting.')
parser.add_argument('--accuracy', type=float, default=None, help='for manual setting.')
parser.add_argument('--segment_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--cluster_num', type=int, default=None, help='for manual setting.')
parser.add_argument('--DBSCAN_eps', type=float, default=None, help='for manual setting.')
parser.add_argument('--model_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--angle_threshold', type=float, default=None, help='for manual setting.')
parser.add_argument('--nonuniform', action='store_true', help='for manual setting.')

parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))


def main():
    compressor_cfg = load_compressor_cfg(args.compressor_yaml)
    accuracy = compressor_cfg['ACCURACY'] * 2
    basic_compressor = BasicCompressor(compressor_yaml=args.compressor_yaml)
    test_dataset = build_dataset(datalist=None, dataset_name=args.dataset)
    pc_seg = PointCloudSegment(test_dataset.transform_map)
    segment_cfg = {
        'segment_method': compressor_cfg['SEGMENT_METHOD'],
        'ground_vertical_threshold': compressor_cfg['GROUND_THRESHOLD'],
        'cluster_num': compressor_cfg['CLUSTER_NUM'],  # used in farthest point sampling
        'DBSCAN_eps': compressor_cfg['DBSCAN_EPS']  # used in DBSCAN
    }
    model_cfg = {
        'model_method': compressor_cfg['MODEL_METHOD'],
        'angle_threshold': compressor_cfg['PLANE_ANGLE_THRESHOLD'],
    }

    if args.basic_compressor is not None:
        basic_compressor.set_method(args.basic_compressor)
    if args.accuracy is not None:
        if args.accuracy > 0:
            accuracy = args.accuracy * 2
        else:
            accuracy = args.accuracy
    if args.segment_method is not None:
        segment_cfg['segment_method'] = args.segment_method
    if args.cluster_num is not None:
        segment_cfg['cluster_num'] = args.cluster_num
    if args.DBSCAN_eps is not None:
        segment_cfg['DBSCAN_eps'] = args.DBSCAN_eps
    if args.model_method is not None:
        model_cfg['model_method'] = args.model_method
    if args.angle_threshold is not None:
        model_cfg['angle_threshold'] = args.angle_threshold

    PCTransformer = test_dataset.PCTransformer

    file_list = glob.glob(os.path.join(args.data_dir, '*'))
    file_list.sort()

    def reconstruct_point_cloud(file):
        # file = '/data/KITTI_detection/training/velodyne/001120.bin'
        # file = '/data/KITTI_detection/training/velodyne_false/000002.bin'
        # print(file)
        # i = -1
        point_cloud, range_image, original_point_cloud = test_dataset.load_range_image_points_from_file(file)

        if accuracy == 0:
            # use range image point cloud without lossy
            point_cloud_rec = copy.copy(point_cloud)
        elif accuracy == -1:
            # use point cloud original from bin file
            point_cloud_rec = copy.copy(original_point_cloud)
        else:
            # use reconstructed point cloud
            seg_idx, ground_model = pc_seg.segment(point_cloud, range_image, segment_cfg, cpu=args.cpu)
            cluster_models = pc_seg.cluster_modeling(point_cloud, range_image, seg_idx, model_cfg)
            model_param = np.concatenate((ground_model.reshape(1, 4), cluster_models), 0)
            range_image_pred = pc_seg.intra_predict(seg_idx, model_param)
            # point_cloud_pred = PCTransformer.range_image_to_point_cloud(range_image_pred)
            residual = range_image - range_image_pred
            model_num = seg_idx.max() + 1

            QM = QuantizationModule(accuracy, uniform=(not args.nonuniform))
            residual_quantized, salience_score, quantize_level, key_point_map = RM.quantize_residual(residual,
                                                                                                     seg_idx,
                                                                                                     point_cloud,
                                                                                                     range_image)
            residual_rec = QM.dequantize_residual(residual_quantized, seg_idx, quantize_level)
            range_image_pred = pc_seg.intra_predict(seg_idx, model_param)
            range_image_rec = range_image_pred + residual_rec
            point_cloud_rec = PCTransformer.range_image_to_point_cloud(range_image_rec)

            range_dif = np.abs(range_image_rec - range_image)
            max_range_error = np.max(range_dif)
            # mean_range_error = np.mean(range_dif)
            # print('max range error: ', max_range_error)
            # print('mean range error: ', mean_range_error)
            if args.nonuniform:
                if max_range_error > accuracy + 0.06 + 0.0001:
                    print('reconstruction error..... plz check')
                    IPython.embed()
            else:
                if max_range_error > accuracy + 0.0001:
                    print('reconstruction error..... plz check')
                    IPython.embed()

        point_cloud_rec = point_cloud_rec[np.where(np.sum(point_cloud_rec, -1) != 0)]
        point_cloud_rec = np.concatenate((point_cloud_rec, np.zeros((point_cloud_rec.shape[0], 1))), -1)
        # IPython.embed()
        point_cloud_rec = point_cloud_rec.astype(np.float32)

        save_path = file.replace('/velodyne_original/', '/velodyne/')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        point_cloud_rec.tofile(save_path)

        # range_dif = np.abs(range_image_rec - range_image)
        # max_range_error = np.max(range_dif)
        # mean_range_error = np.mean(range_dif)
        # print('max range error: ', max_range_error)
        # print('mean range error: ', mean_range_error)
        # save_point_cloud_to_pcd(np.reshape(point_cloud_original, (-1, 3)), os.path.join('log_tmp', 'o.pcd'),
        #                         color=None, save=True)
        # save_point_cloud_to_pcd(np.reshape(point_cloud, (-1, 3)), os.path.join('log_tmp', 'p.pcd'),
        #                         color=None, save=True)
        # save_point_cloud_to_pcd(point_cloud_rec[:, :3], os.path.join('log_tmp', 'r.pcd'),
        #                         color=None, save=True)
        # IPython.embed()

    #     chamfer_dist_ours = calc_chamfer_distance(point_cloud, point_cloud_rec[..., :3])
    #     original_data, compressed_data = compress_point_cloud(basic_compressor, model_param,
    #                                                           seg_idx, residual_quantized,
    #                                                           point_cloud, range_image,
    #                                                           full=False)  # if full = True
    #     print('bpp: ', len(compressed_data['residual_quantized'] + \
    #                  compressed_data['contour_map'] + \
    #                  compressed_data['idx_sequence'] + \
    #                  compressed_data['plane_param']) * 8 / np.where(range_image != 0)[0].shape[0])
    #     # draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=True,
    #     #                      save_path=os.path.join(BASE_DIR, 'draw_ICRA_figures/qualitative_fig/ours_uniform_all.pcd'))
    #     # draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=False,
    #     #                      save_path=os.path.join(BASE_DIR, 'draw_ICRA_figures/qualitative_fig/ours_uniform.pcd'))
    #     # draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=False,
    #     #                      save=False, vis=True)
    #
    #     draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=True,
    #                          save_path=os.path.join(BASE_DIR, 'draw_ICRA_figures/qualitative_fig/ours_nonuniform_all.pcd'))
    #     draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=False,
    #                          save_path=os.path.join(BASE_DIR, 'draw_ICRA_figures/qualitative_fig/ours_nonuniform.pcd'))
    #     draw_qualitative_point_clouds(point_cloud, point_cloud_rec[:, :3], vis_all=False,
    #                          save=False, vis=True)
    #     IPython.embed()
    #
    #
    # reconstruct_point_cloud(file_list[0])
    # a=b
    if accuracy == 0 or accuracy == -1:
        with futures.ThreadPoolExecutor(12) as executor:
            list(tqdm(executor.map(reconstruct_point_cloud, file_list), total=len(file_list)))
    else:
        with futures.ThreadPoolExecutor(4) as executor:
            list(tqdm(executor.map(reconstruct_point_cloud, file_list), total=len(file_list)))
    # for file in tqdm(file_list):
    #     reconstruct_point_cloud(file)
    print('\n\nTransform finished.')


if __name__ == '__main__':
    main()
