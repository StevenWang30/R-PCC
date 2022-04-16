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
import IPython


parser = argparse.ArgumentParser()
# Data related arguments
parser.add_argument('--input', help='single frame input for static compression.')
parser.add_argument('--output', help='output bitstream.')
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

parser.add_argument('--eval', action='store_true', help='evaluate the reconstruction quality.')

parser.add_argument('--cpu', action='store_true', help='set --cpu to make it run on CPU only if lack of GPU.')
args = parser.parse_args()
print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))


def compress():
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

    pc_seg = PointCloudSegment(dataset.transform_map)
    if not args.cpu:
        # for time cost
        point_cloud, range_image, original_point_cloud = dataset.load_range_image_points_from_file(args.input)
        pc_seg.segment(point_cloud, range_image, segment_cfg, cpu=args.cpu)

    t_init = time.time()
    point_cloud, range_image, original_point_cloud = dataset.load_range_image_points_from_file(args.input)
    nonzero_point_cloud = point_cloud[np.where(point_cloud[..., 0] != 0)]
    point_num = nonzero_point_cloud.shape[0]
    t_load_data = time.time()

    seg_idx, ground_model = pc_seg.segment(point_cloud, range_image, segment_cfg, cpu=args.cpu)
    t_segmentation = time.time()

    cluster_models = pc_seg.cluster_modeling(point_cloud, range_image, seg_idx, model_cfg)
    model_param = np.concatenate((ground_model.reshape(1, 4), cluster_models), 0)
    t_modeling = time.time()

    range_image_pred = pc_seg.intra_predict(seg_idx, model_param)
    residual = range_image - range_image_pred
    t_intra_pred = time.time()

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
    residual_quantized, salience_level, key_point_map = QM.quantize_residual(residual,
                                                                             seg_idx,
                                                                             point_cloud,
                                                                             range_image)
    t_quantization = time.time()

    # if full = True, will compress point cloud, range image, and ground and non-grond points.
    original_data, compressed_data = compress_point_cloud(basic_compressor, model_param,
                                                          seg_idx, salience_level, residual_quantized,
                                                          point_cloud, range_image,
                                                          full=False)

    t_basic_compressor = time.time()

    save_compressed_bitstream(args.output, compressed_data, uniform=uniform)
    t_save = time.time()

    print('\nCompression finished.')
    print('binary bitstream save in ', args.output)

    print('\nTime Cost:')
    print('    Load data: ', t_load_data - t_init)
    print('    Segmentation module: ', t_segmentation - t_load_data)
    print('    Modeling module: ', t_modeling - t_segmentation)
    print('    Intra-prediction module: ', t_intra_pred - t_modeling)
    print('    Quantization module: ', t_quantization - t_intra_pred)
    print('    Basic compressor module (', basic_compressor.method_name, '): ', t_basic_compressor - t_quantization)
    print('    Save binary file: ', t_save - t_basic_compressor)
    print('    Total time cost: ', t_save - t_init)
    print('    Total time cost without loading data: ', t_save - t_load_data)

    compressed_bit_size = os.path.getsize(args.output) * 8
    print('\nCompression Results: ')
    print('    Compression ratio: ', (point_num * 32 * 3) / compressed_bit_size)
    print('    BPP: ', compressed_bit_size / point_num)
    print('\n')

    if args.eval:
        # reconstruct
        compressed_data = read_compressed_bitstream(args.output, uniform=uniform)
        residual_quantized, seg_idx, salience_level, plane_param = decompress_point_cloud(compressed_data, basic_compressor,
                                                                                          model_num,
                                                                                          dataset.transform_map.shape[0],
                                                                                          dataset.transform_map.shape[1])
        QM = QuantizationModule(accuracy, uniform=uniform)
        residual = QM.dequantize_residual(residual_quantized, seg_idx, salience_level)

        range_image_pred = pc_seg.intra_predict(seg_idx, plane_param)
        range_image_rec = range_image_pred + residual
        point_cloud_rec = PCTransformer.range_image_to_point_cloud(range_image_rec)

        range_dif = np.abs(range_image_rec - range_image)
        max_depth_error = np.max(range_dif)
        mean_depth_error = np.mean(range_dif)

        if uniform:
            if max_depth_error > accuracy + 0.00001:
                AssertionError('Reconstruction error... Please check...')
        if not uniform:
            if max_depth_error > accuracy + 0.06 + 0.00001:
                AssertionError('Reconstruction error... Please check...')

        chamfer_dist_ours = calc_chamfer_distance(point_cloud, point_cloud_rec, out=False)
        point_to_point_result, point_to_plane_result = calc_point_to_point_plane_psnr(point_cloud, point_cloud_rec, out=False)

        print('\nReconstruction quality: ')
        print('    Depth Error (mean): ', mean_depth_error)
        print('    Depth Error (max): ', max_depth_error)
        print('    Chamfer Distance (mean): ', chamfer_dist_ours['mean'])
        print('    F1 score (threshold=0.02): ', chamfer_dist_ours['f_score'])
        print('    Point-to-Point PSNR (r=59.7): ', point_to_point_result['psnr_mean'])
        print('    Point-to-Plane PSNR (r=59.7): ', point_to_plane_result['psnr_mean'])


if __name__ == '__main__':
    compress()
