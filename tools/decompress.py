import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
import numpy as np
import argparse
import time
from utils.compress_utils import BasicCompressor
from utils.utils import load_compressor_cfg
from utils.compress_utils import compress_point_cloud, decompress_point_cloud
from utils.segment_utils import PointCloudSegment
from utils.evaluate_metrics import calc_chamfer_distance, calc_point_to_point_plane_psnr
from dataset import build_dataset
from utils.compress_utils import QuantizationModule, save_compressed_bitstream, read_compressed_bitstream


parser = argparse.ArgumentParser()
# Data related arguments
parser.add_argument('--input', help='input compressed bitstream.')
parser.add_argument('--output', help='output reconstructed point cloud.')
parser.add_argument('--lidar', help='lidar type of this point cloud collection.')

parser.add_argument('--compressor_yaml', default=os.path.join(BASE_DIR, 'cfgs/compressor.yaml'))

parser.add_argument('--basic_compressor', type=str, default=None, help='for manual setting.')
parser.add_argument('--accuracy', type=float, default=None, help='for manual setting.')
parser.add_argument('--segment_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--cluster_num', type=int, default=None, help='for manual setting.')
parser.add_argument('--DBSCAN_eps', type=float, default=None, help='for manual setting.')
parser.add_argument('--model_method', type=str, default=None, help='for manual setting.')
parser.add_argument('--angle_threshold', type=float, default=None, help='for manual setting.')
parser.add_argument('--nonuniform', action='store_true', help='for manual setting.')

parser.add_argument('--eval', action='store_true',
                    help='please set path of original point cloud to evaluate the reconstruction quality.')
parser.add_argument('--original_point_cloud', default=None)

parser.add_argument('--cpu', action='store_true', help='set --cpu to make it run on CPU only if lack of GPU.')
args = parser.parse_args()
print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))


def decompress():
    compressor_cfg = load_compressor_cfg(args.compressor_yaml)
    accuracy = compressor_cfg['accuracy'] * 2
    basic_compressor = BasicCompressor(compressor_yaml=args.compressor_yaml)
    dataset = build_dataset(lidar_type=args.lidar)
    segment_cfg = {
        'segment_method': compressor_cfg['segment_method'],
        'ground_vertical_threshold': compressor_cfg['ground_threshold'],
        'cluster_num': compressor_cfg['cluster_num'],  # used in farthest point sampling
        'DBSCAN_eps': compressor_cfg['DBSCAN_eps']  # used in DBSCAN
    }
    model_cfg = {
        'model_method': compressor_cfg['modeling_method'],
        'angle_threshold': compressor_cfg['plane_angle_threshold'],
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

    model_num = segment_cfg['cluster_num'] + 1
    PCTransformer = dataset.PCTransformer

    if args.nonuniform:
        uniform = False
    else:
        if compressor_cfg['compress_framework'] == 'uniform':
            uniform = True
        else:
            uniform = False

    # reconstruct
    compressed_data = read_compressed_bitstream(args.input, uniform=uniform)
    residual_quantized, seg_idx, salience_level, plane_param = decompress_point_cloud(compressed_data, basic_compressor,
                                                                                      model_num,
                                                                                      dataset.transform_map.shape[0],
                                                                                      dataset.transform_map.shape[1])
    if uniform:
        QM = QuantizationModule(accuracy)
    else:
        QM = QuantizationModule(accuracy, uniform=False,
                                level_kp_num=tuple(compressor_cfg['level_key_point_num']),
                                level_dacc=tuple(compressor_cfg['level_delta_acc']),
                                ground_salience_level=compressor_cfg['ground_salience_level'],
                                feature_region=compressor_cfg['feature_region'],
                                segments=compressor_cfg['segments'],
                                sharp_num=compressor_cfg['sharp_num'],
                                less_sharp_num=compressor_cfg['less_sharp_num'],
                                flat_num=compressor_cfg['flat_num'])
    residual = QM.dequantize_residual(residual_quantized, seg_idx, salience_level)

    pc_seg = PointCloudSegment(dataset.transform_map)
    range_image_pred = pc_seg.intra_predict(seg_idx, plane_param)
    range_image_rec = range_image_pred + residual
    point_cloud_rec = PCTransformer.range_image_to_point_cloud(range_image_rec)

    dataset.save_point_cloud_to_file(args.output, point_cloud_rec)

    print('\nDecompression finished.')
    print(args.output.split('.')[-1], 'file save in ', args.output)

    if args.eval:
        assert args.original_point_cloud is not None, 'If want to evaluate the reconstruction quality, must set the original point cloud file path first.'
        print('\nStart evaluation...')
        point_cloud, range_image, original_point_cloud = \
            dataset.load_range_image_points_from_file(args.original_point_cloud)
        n_points = np.where(range_image != 0)[0].shape[0]

        range_dif = np.abs(range_image_rec - range_image)
        max_depth_error = np.max(range_dif)
        mean_depth_error = np.mean(range_dif)

        if uniform:
            if max_depth_error > accuracy + 0.00001:
                print('Does the the uniform or non-uniform compression framework matches the compress processing?')
                AssertionError('reconstruction error..... please check')
        if not uniform:
            if max_depth_error > accuracy + 0.06 + 0.00001:
                print('Does the the uniform or non-uniform compression framework matches the compress processing?')
                AssertionError('reconstruction error..... please check')

        chamfer_dist_ours = calc_chamfer_distance(point_cloud, point_cloud_rec, out=False)
        point_to_point_result, point_to_plane_result = calc_point_to_point_plane_psnr(point_cloud, point_cloud_rec, out=False)

        compressed_bit_size = os.path.getsize(args.input) * 8

        print('\nCompared with ', args.original_point_cloud)
        print('    BPP: ', compressed_bit_size / n_points)
        print('    Compression Ratio: ', (n_points * 32 * 3) / compressed_bit_size)
        print('    Depth Error (mean): ', mean_depth_error)
        print('    Depth Error (max): ', max_depth_error)
        print('    Chamfer Distance (mean): ', chamfer_dist_ours['mean'])
        print('    F1 score (threshold=0.02): ', chamfer_dist_ours['f_score'])
        print('    Point-to-Point PSNR (r=59.7): ', point_to_point_result['psnr_mean'])
        print('    Point-to-Plane PSNR (r=59.7): ', point_to_plane_result['psnr_mean'])


if __name__ == '__main__':
    decompress()
