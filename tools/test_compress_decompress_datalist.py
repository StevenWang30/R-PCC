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
from utils.utils import load_compressor_cfg
import copy
from utils.compress_utils import compress_point_cloud, decompress_point_cloud
from utils.segment_utils import PointCloudSegment
# from utils.chamfer_dist_utils import dist_chamfer
from utils.evaluate_metrics import calc_chamfer_distance, calc_point_to_point_plane_psnr
from datasets import build_dataset
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from utils.compress_utils import QuantizationModule, extract_features

import torch

parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--result_dir', default=os.path.join(BASE_DIR, 'experiment_results/temp_results'))
# Data related arguments
# parser.add_argument('--datalist', default='data/test_64E_KITTI_city.txt')
parser.add_argument('--datalist', default=os.path.join(BASE_DIR, 'data/test_64E_KITTI_city.txt'))
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

parser.add_argument('--max_eval_frames', type=int, default=-1, help='Only eval first N frames. When it == -1, eval all.')

# parser.add_argument('--eval_chamfer_distance', action='store_true')
#
parser.add_argument('--output_file', default=None)

parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))

scene = args.datalist.split('/')[-1]


time_test = False
def main():
    compressor_cfg = load_compressor_cfg(args.compressor_yaml)
    accuracy = compressor_cfg['ACCURACY'] * 2
    basic_compressor = BasicCompressor(compressor_yaml=args.compressor_yaml)
    test_dataset = build_dataset(datalist=args.datalist, dataset_name=args.dataset)
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
        accuracy = args.accuracy * 2
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

    compressed_size_list = []
    original_size_list = []
    point_num_list = []
    chamfer_distance_list = []
    f1_score_list = []
    point_to_point_psnr_list = []
    point_to_plane_psnr_list = []
    for i in range(len(test_dataset)):
        if args.max_eval_frames is not None and args.max_eval_frames != -1:
            if i >= args.max_eval_frames:
                break
        if time_test:
            print('\nTime cost:')
            t = time.time()
            t_init = time.time()
        point_cloud, range_image, original_point_cloud, file_name = test_dataset[i]
        t = time.time()
        nonzero_point_cloud = point_cloud[np.where(point_cloud[..., 0] != 0)]
        if time_test:
            print('nonzero...', time.time() - t)
        # point_cloud_original = test_dataset.get_original_data(i)
        # compare_point_clouds(point_cloud_original, point_cloud, vis_all=True, save_path=os.path.join('log_tmp', 'oa.pcd'))
        # visualize_left_points(point_cloud_original, point_cloud)

        # range_image_original = copy.copy(range_image)
        # point_num = range_image.shape[0] * range_image.shape[1]
        point_num = nonzero_point_cloud.shape[0]

        # point_cloud = test_dataset.range_image_to_point_cloud(range_image, lidar_param)
        if time_test:
            print('  load data: ', time.time() - t)
            torch.cuda.synchronize()
            t = time.time()

        # plane_param, residual, seg_idx = pc_seg.segment_cuda(point_cloud, range_image)
        seg_idx, ground_model = pc_seg.segment(point_cloud, range_image, segment_cfg, cpu=args.cpu)
        cluster_models = pc_seg.cluster_modeling(point_cloud, range_image, seg_idx, model_cfg)
        model_param = np.concatenate((ground_model.reshape(1, 4), cluster_models), 0)
        range_image_pred = pc_seg.intra_predict(seg_idx, model_param)
        # point_cloud_pred = PCTransformer.range_image_to_point_cloud(range_image_pred)
        residual = range_image - range_image_pred
        model_num = seg_idx.max() + 1


        QM = QuantizationModule(accuracy, uniform=(not args.nonuniform))
        residual_quantized, salience_score, quantize_level, key_point_map = QM.quantize_residual(residual,
                                                                                                 seg_idx,
                                                                                                 point_cloud,
                                                                                                 range_image)

        #
        # residual_collect = []
        # for m in range(model_num):
        #     residual_collect.extend(residual[..., 0][np.where(seg_idx == m)])
        # residual_collect = np.array(residual_collect).reshape(range_image.shape)
        # residual_collect_quantized = np.rint(residual_collect / accuracy).astype(np.uint16)

        # cluster_param_o, plane_param_o, residual_o, seg_idx_o = pc_seg.segment_with_cluster(point_cloud, range_image)
        if time_test:
            torch.cuda.synchronize()
            print('  all compress segment: ', time.time() - t)
            t = time.time()
        # IPython.embed()
        # print('plane param: ', plane_param)

        # residual = calc_residual(range_image, plane_param, cluster_param, seg_idx, lidar_param)
        #
        # print('calculate residual: ', time.time() - t)
        t = time.time()
        # start compress
        ground_idx = np.where(seg_idx == 0)
        nonground_idx = np.where(seg_idx > 0)
        ground_residual_quantized = np.rint(residual[ground_idx] / accuracy)
        cluster_residual_quantized = np.rint(residual[nonground_idx] / accuracy)

        original_data, compressed_data = compress_point_cloud(basic_compressor, model_param,
                                                              seg_idx, residual_quantized,
                                                              ground_residual_quantized, cluster_residual_quantized,
                                                              point_cloud, range_image,
                                                              full=False)  # if full = True, will compress point cloud, range image, and ground and non-grond points.
        if time_test:
            print('  basic compression: ', time.time() - t)
            t = time.time()

        bit_stream = compressed_data['residual_quantized'] + \
                     compressed_data['contour_map'] + \
                     compressed_data['idx_sequence'] + \
                     compressed_data['plane_param']

        if time_test:
            print('  save bit stream: ', time.time() - t)
            print('  total compression cost: ', time.time() - t_init, '\n')

        compressed_size_list.append(bit_size(bit_stream))
        original_size_list.append(np_size(nonzero_point_cloud))
        point_num_list.append(point_num)


        # reconstruct
        t = time.time()
        t_init = time.time()
        residual_quantized, seg_idx, plane_param = decompress_point_cloud(compressed_data, basic_compressor,
                                                                          model_num,
                                                                          test_dataset.transform_map.shape[0],
                                                                          test_dataset.transform_map.shape[1])
        # residual = residual_quantized * accuracy
        residual = QM.dequantize_residual(residual_quantized, seg_idx, quantize_level)
        if time_test:
            print('\n\nTime cost:\n  decompression: ', time.time() - t)
            t = time.time()

        range_image_pred = pc_seg.intra_predict(seg_idx, plane_param)
        range_image_rec = range_image_pred + residual
        point_cloud_rec = PCTransformer.range_image_to_point_cloud(range_image_rec)

        if time_test:
            print('  reconstruction: ', time.time() - t)
            print('  decompression pyand reconstruction: ', time.time() - t_init, '\n')

        range_dif = np.abs(range_image_rec - range_image)
        max_range_error = np.max(range_dif)
        mean_range_error = np.mean(range_dif)
        print('max range error: ', max_range_error)
        print('mean range error: ', mean_range_error)

        if max_range_error > accuracy + 0.06 + 0.01:
            print('reconstruction error..... plz check')
            IPython.embed()

        # chamfer_dist = calc_chamfer_distance(point_cloud_original, point_cloud_rec)
        # chamfer_dist_base = calc_chamfer_distance(point_cloud_original, point_cloud)
        chamfer_dist_ours = calc_chamfer_distance(point_cloud, point_cloud_rec, out=False)
        # chamfer_result = calc_chamfer_distance(point_cloud_rec, nonzero_point_cloud)
        point_to_point_result, point_to_plane_result = calc_point_to_point_plane_psnr(point_cloud, point_cloud_rec, out=False)

        chamfer_distance_list.append(chamfer_dist_ours['mean'])
        f1_score_list.append(chamfer_dist_ours['f_score'])
        point_to_point_psnr_list.append(point_to_point_result['psnr_mean'])
        point_to_plane_psnr_list.append(point_to_plane_result['psnr_mean'])

        print('\ncompose ', file_name)
        print('mean chamfer distance: ', chamfer_distance_list[-1])
        print('mean f1 score(0.1): ', f1_score_list[-1])
        print('mean p2point PSNR: ', point_to_point_psnr_list[-1])
        print('mean p2plane PSNR: ', point_to_plane_psnr_list[-1])
        print('compression rate: ',  original_size_list[-1] / compressed_size_list[-1])

        # # time test:
        # residual_quantized = np.rint(residual / accuracy).astype(np.int16)
        # t = time.time()
        # for i in range(100):
        #     res = basic_compressor.compress(residual_quantized)
        # print('t_python: ', time.time() - t)
        #
        # import os
        # np.save('a.npy', residual_quantized)
        # t = time.time()
        # for i in range(100):
        #     os.system('bzip2 -1 < a.npy > a.rb2')
        # print('t_c++: ', time.time() - t)

    print('\n\nTest finished.')
    frame_num = len(original_size_list)
    compression_rate_list = np.array(original_size_list) / np.array(compressed_size_list)
    bpp_list = np.array(compressed_size_list) * 8 / np.array(point_num_list)
    print('In scene %s, test %d point cloud frames, when accuracy is %.4f, the mean compression rate is: %.2f, the mean bpp is: %.2f' %
          (scene, frame_num, accuracy / 2, compression_rate_list.mean(), bpp_list.mean()))

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "a+") as f:
            f.write('Evaluation parameters...\n')
            for key, val in vars(args).items():
                f.write("{:16} {}".format(key, val))
                f.write('\n')
            f.write('\nAll data evaluation finished. \n')
            f.write('Scene: %s\n' % scene)
            f.write('Test %d point cloud frames\n' % frame_num)
            f.write('Accuracy: %.4f\n' % (accuracy / 2))
            f.write('Result:\n')
            f.write('    Ours method: (mean)\n')
            f.write('        compression rate: %.2f\n'
                    '        bpp: %.2f\n'
                    '        chamfer_distance: %.4f\n'
                    '        f1_score: %.4f\n'
                    '        point_to_point_psnr: %.2f\n'
                    '        point_to_plane_psnr: %.2f\n'
                    %
                    (compression_rate_list.mean(),
                     bpp_list.mean(),
                     np.array(chamfer_distance_list).mean(),
                     np.array(f1_score_list).mean(),
                     np.array(point_to_point_psnr_list).mean(),
                     np.array(point_to_plane_psnr_list).mean()))


if __name__ == '__main__':
    main()
